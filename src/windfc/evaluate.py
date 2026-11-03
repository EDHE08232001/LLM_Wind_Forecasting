"""Evaluation harness: common model interface, metrics logging, prediction files.

Every model implements :class:`Forecaster` (``fit(task)`` then
``predict(task, split)`` returning kW predictions aligned with
``task.target_index(split)``). :func:`run_forecaster` fits, predicts the val
and test splits, and appends one metrics row per (split, subset) to
``results/metrics.csv`` together with the config name, seed and git commit.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .metrics import binned_errors, regression_metrics, rmse, skill_score
from .splits import ForecastTask
from .utils import append_rows, git_commit, now, results_dir

METRIC_COLUMNS = [
    "timestamp", "git_commit", "config", "model", "family", "protocol", "horizon", "split", "subset",
    "seed", "n", "mae_kw", "mse_kw", "rmse_kw", "r2", "mae_scaled", "mse_scaled", "rmse_scaled",
    "skill_vs_persistence", "fit_seconds", "notes",
]


class Forecaster:
    """Common interface. Subclasses set ``name`` and ``family``."""

    name: str = "model"
    family: str = "generic"
    notes: str = ""

    def fit(self, task: ForecastTask) -> Forecaster:  # pragma: no cover - interface
        return self

    def predict(self, task: ForecastTask, split: str) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError


def evaluate_predictions(
    task: ForecastTask,
    split: str,
    y_pred: np.ndarray,
    model_name: str,
    cfg: Config,
    family: str = "",
    seed: int | None = None,
    fit_seconds: float | None = None,
    notes: str = "",
    save: bool = True,
    verbose: bool = True,
) -> list[dict]:
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    y_true = task.y(split)
    if len(y_pred) != len(y_true):
        raise ValueError(f"{model_name}: {len(y_pred)} predictions for {len(y_true)} targets ({split})")
    rows: list[dict] = []
    subsets = ["all"] if task.protocol == "A" else list(cfg["protocol_b"].get("eval_subsets", ["all"]))
    clean = task.is_clean(split)
    persistence = task.persistence(split) if task.protocol == "B" else None
    for subset in subsets:
        mask = np.ones(len(y_true), dtype=bool) if subset == "all" else clean
        if mask.sum() == 0:
            continue
        m = regression_metrics(y_true[mask], y_pred[mask], task.target_range)
        if persistence is not None:
            m["skill_vs_persistence"] = skill_score(m["rmse_kw"], rmse(y_true[mask], persistence[mask]))
        row = {
            "timestamp": now(),
            "git_commit": git_commit(),
            "config": cfg.get("name", ""),
            "model": model_name,
            "family": family,
            "protocol": task.protocol,
            "horizon": task.horizon,
            "split": split,
            "subset": subset,
            "seed": seed,
            "fit_seconds": fit_seconds,
            "notes": notes,
            **m,
        }
        rows.append(row)
        if verbose:
            skill = m.get("skill_vs_persistence")
            print(
                f"  [{task.name:>6} {split:>4} {subset:>5}] {model_name:<28} "
                f"MAE={m['mae_kw']:8.1f} kW  RMSE={m['rmse_kw']:8.1f} kW  R2={m['r2']:.4f}  "
                f"MAE_s={m.get('mae_scaled', np.nan):.4f}"
                + (f"  skill={skill:+.3f}" if skill is not None else ""),
                flush=True,
            )
    if save:
        out = results_dir(cfg)
        append_rows(out / "metrics.csv", rows)
        save_predictions(task, split, y_pred, model_name, seed, out / "predictions")
    return rows


def prediction_path(pred_dir: Path, protocol_name: str, model_name: str, split: str, seed: int | None) -> Path:
    safe = model_name.replace("/", "_").replace(" ", "_")
    return pred_dir / f"{protocol_name}__{safe}__{split}__seed{seed if seed is not None else 'na'}.parquet"


def save_predictions(task: ForecastTask, split: str, y_pred: np.ndarray, model_name: str, seed: int | None,
                     pred_dir: Path) -> Path:
    pred_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "target_time": task.target_index(split),
            "y_true": task.y(split),
            "y_pred": np.asarray(y_pred, dtype=np.float64),
            "wind_speed": task.wind_speed(split),
            "is_clean": task.is_clean(split),
        }
    )
    if task.protocol == "B":
        df["persistence"] = task.persistence(split)
    path = prediction_path(pred_dir, task.name, model_name, split, seed)
    df.to_parquet(path, index=False)
    return path


def run_forecaster(model: Forecaster, task: ForecastTask, cfg: Config, seed: int | None = None,
                   splits: tuple[str, ...] = ("val", "test"), save: bool = True) -> list[dict]:
    t0 = time.time()
    model.fit(task)
    fit_s = time.time() - t0
    rows: list[dict] = []
    for split in splits:
        if split == "val" and not task.has_val:
            continue
        pred = model.predict(task, split)
        rows += evaluate_predictions(task, split, pred, model.name, cfg, family=model.family, seed=seed,
                                     fit_seconds=round(fit_s, 2), notes=model.notes, save=save)
    return rows


def binned_report(task: ForecastTask, split: str, y_pred: np.ndarray, cfg: Config) -> pd.DataFrame:
    return binned_errors(task.y(split), y_pred, task.wind_speed(split), cfg["metrics"]["wind_bins"])
