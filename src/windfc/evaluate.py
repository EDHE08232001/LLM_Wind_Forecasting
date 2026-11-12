"""Scoring: metrics rows appended to ``results/metrics.csv`` and per-run prediction files.

Every prediction vector is aligned with ``task.target_index(split)`` and is in kW.
Each (split, subset) pair gives one row, where ``subset`` is ``all`` (every hour)
or ``clean`` (only the hours that pass the paper's cleaning rules, i.e. the rows
the paper evaluated on).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .metrics import regression_metrics, rmse, skill_score
from .splits import ForecastTask
from .utils import append_rows, git_commit, now, results_dir

METRIC_COLUMNS = [
    "timestamp", "git_commit", "config", "model", "family", "variant", "horizon", "split", "subset", "n",
    "mae_kw", "mse_kw", "rmse_kw", "r2", "r2_pearson", "mae_scaled", "mse_scaled", "rmse_scaled",
    "skill_vs_persistence", "interval_coverage", "notes",
]


def evaluate_predictions(
    task: ForecastTask,
    split: str,
    y_pred: np.ndarray,
    model_name: str,
    cfg: Config,
    family: str = "",
    variant: str = "",
    notes: str = "",
    interval: tuple[np.ndarray, np.ndarray] | None = None,
    save: bool = True,
    verbose: bool = True,
) -> list[dict]:
    """Score one model on one split. ``interval`` is an optional (q10, q90) pair in kW."""
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    y_true = task.y(split)
    if len(y_pred) != len(y_true):
        raise ValueError(f"{model_name}: {len(y_pred)} predictions for {len(y_true)} targets ({split})")
    clean = task.is_clean(split)
    persistence = task.persistence(split)
    rows: list[dict] = []
    for subset in cfg["evaluation"].get("subsets", ["all", "clean"]):
        mask = np.ones(len(y_true), dtype=bool) if subset == "all" else clean
        if mask.sum() < 2:
            continue
        m = regression_metrics(y_true[mask], y_pred[mask], task.target_range)
        m["skill_vs_persistence"] = skill_score(m["rmse_kw"], rmse(y_true[mask], persistence[mask]))
        if interval is not None:
            lo, hi = interval
            m["interval_coverage"] = float(np.mean((y_true[mask] >= lo[mask]) & (y_true[mask] <= hi[mask])))
        rows.append({
            "timestamp": now(), "git_commit": git_commit(), "config": cfg.get("name", ""), "model": model_name,
            "family": family, "variant": variant, "horizon": task.horizon, "split": split, "subset": subset,
            "notes": notes, **m,
        })
        if verbose:
            print(
                f"  [{task.name:>3} {split:>4} {subset:>5}] {model_name:<32} MAE={m['mae_kw']:7.1f} kW  "
                f"RMSE={m['rmse_kw']:7.1f} kW  R2={m['r2']:.4f}  MAE_s={m['mae_scaled']:.4f}  "
                f"skill={m['skill_vs_persistence']:+.3f}",
                flush=True,
            )
    if save:
        out = results_dir(cfg)
        append_rows(out / "metrics.csv", [{c: r.get(c) for c in METRIC_COLUMNS} for r in rows])
        save_predictions(task, split, y_pred, model_name, out / "predictions", interval)
    return rows


def prediction_path(pred_dir: Path, task_name: str, model_name: str, split: str) -> Path:
    safe = model_name.replace("/", "_").replace(" ", "_")
    return pred_dir / f"{task_name}__{safe}__{split}.parquet"


def save_predictions(task: ForecastTask, split: str, y_pred: np.ndarray, model_name: str, pred_dir: Path,
                     interval: tuple[np.ndarray, np.ndarray] | None = None) -> Path:
    pred_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "target_time": task.target_index(split),
        "y_true": task.y(split),
        "y_pred": np.asarray(y_pred, dtype=np.float64),
        "persistence": task.persistence(split),
        "wind_speed": task.wind_speed(split),
        "is_clean": task.is_clean(split),
    })
    if interval is not None:
        df["q10"], df["q90"] = interval
    path = prediction_path(pred_dir, task.name, model_name, split)
    df.to_parquet(path, index=False)
    return path
