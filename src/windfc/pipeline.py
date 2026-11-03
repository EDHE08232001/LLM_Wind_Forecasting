"""High-level stages shared by the numbered scripts and ``scripts/run_all.py``."""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from .config import Config, resolve_path
from .data import build_hourly_dataset, load_processed
from .device import free_device_memory, get_device
from .evaluate import evaluate_predictions, run_forecaster
from .models.baselines import build_baseline
from .splits import ForecastTask
from .train import NetForecaster
from .utils import banner, checkpoints_dir, results_dir, save_run_config


def stage_prepare(cfg: Config) -> dict:
    banner("Phase 1: data pipeline")
    df, report = build_hourly_dataset(cfg)
    for k, v in report.items():
        print(f"  {k:>28}: {v}")
    print(f"  final frame: {df.shape[0]} rows x {df.shape[1]} cols, clean rows={int(df['is_clean'].sum())}")
    return report


def _tasks(cfg: Config, df: pd.DataFrame, protocols: list[str]) -> list[ForecastTask]:
    tasks = []
    for p in protocols:
        if p.upper() == "A":
            tasks.append(ForecastTask(df, cfg, "A"))
        else:
            for h in cfg["protocol_b"]["horizons"]:
                tasks.append(ForecastTask(df, cfg, "B", int(h)))
    return tasks


def stage_baselines(cfg: Config, protocols: list[str] = ("A", "B"), models: list[str] | None = None) -> list[dict]:
    banner("Phase 3: baselines")
    df = load_processed(cfg)
    device = str(get_device(cfg.get("device", "auto")))
    models = models or list(cfg["baselines"]["models"])
    rows = []
    save_run_config(cfg, results_dir(cfg) / "runs", f"baselines_{int(time.time())}")
    for task in _tasks(cfg, df, list(protocols)):
        print(task.describe())
        for name in models:
            if task.protocol == "A" and name in ("persistence", "seasonal_naive"):
                continue  # not defined for same-hour regression on shuffled rows
            model = build_baseline(name, cfg, device=device, seed=int(cfg.get("seed", 42)))
            rows += run_forecaster(model, task, cfg, seed=int(cfg.get("seed", 42)))
    return rows


def stage_paper(cfg: Config, protocols: list[str] = ("A", "B"), models: list[str] | None = None,
                seeds: list[int] | None = None) -> list[dict]:
    banner("Phase 4: paper networks (ANN / CNN / RNN / LSTM)")
    df = load_processed(cfg)
    tcfg = cfg["train"]
    models = models or list(tcfg["models"])
    ckpt = checkpoints_dir(cfg) / "paper_nets"
    rows = []
    save_run_config(cfg, results_dir(cfg) / "runs", f"paper_{int(time.time())}")
    for task in _tasks(cfg, df, list(protocols)):
        print(task.describe())
        if task.protocol == "A":
            run_seeds = [int(cfg["protocol_a"]["seed"])]  # paper: fixed 100 epochs, no validation
        else:
            run_seeds = list(seeds) if seeds is not None else [int(s) for s in tcfg["seeds_protocol_b"]]
        for arch in models:
            for seed in run_seeds:
                model = NetForecaster(arch, tcfg, device=cfg.get("device", "auto"), seed=seed,
                                      use_early_stopping=task.protocol == "B", checkpoint_dir=ckpt)
                rows += run_forecaster(model, task, cfg, seed=seed)
                model.release()
            free_device_memory()
    return rows


def stage_chronos(cfg: Config, variants: list[str] | None = None, pipe=None) -> list[dict]:
    banner("Phase 5a: Chronos-2 zero-shot")
    from .models.chronos import ChronosForecaster, load_pipeline

    df = load_processed(cfg)
    ccfg = cfg["chronos"]
    variants = variants or list(ccfg["variants"])
    pipe = pipe or load_pipeline(ccfg["model_id"], cfg.get("device", "auto"))
    tasks = _tasks(cfg, df, ["B"])
    rows = []
    for variant in variants:
        model = ChronosForecaster(pipe, cfg, variant=variant, tag="zeroshot")
        for task in tasks:
            print(task.describe())
            rows += run_forecaster(model, task, cfg, seed=None)
            _save_intervals(cfg, model, task)
        model.clear_cache()
    del pipe
    free_device_memory()
    return rows


def stage_finetune(cfg: Config, method: str | None = None, grid: bool = True) -> list[dict]:
    banner("Phase 5b: Chronos-2 fine-tuning")
    from .models.chronos import ChronosForecaster, finetune_chronos, grid_search_finetune

    df = load_processed(cfg)
    fcfg = cfg["finetune"]
    if method:
        cfg["finetune"]["method"] = method
    out_root = checkpoints_dir(cfg) / "chronos2" / fcfg.get("method", "lora")
    device = cfg.get("device", "auto")
    seed = int(cfg.get("seed", 42))
    if grid:
        pipe, info, grid_rows = grid_search_finetune(df, cfg, device=device, seed=seed, out_root=out_root)
    else:
        ctx, lr = int(fcfg["context_lengths"][0]), float(fcfg["learning_rates"][0])
        pipe, info = finetune_chronos(df, cfg, ctx, lr, device=device, seed=seed,
                                      out_dir=out_root / f"ctx{ctx}_lr{lr:g}")
        grid_rows = [{k: v for k, v in info.items() if k != "history"}]
    res = results_dir(cfg)
    pd.DataFrame(grid_rows).to_csv(res / "tables" / f"finetune_grid_{info['method']}.csv", index=False)
    (res / "tables" / f"finetune_best_{info['method']}.json").write_text(json.dumps(info, indent=2, default=float))
    verdict = "beats" if info["best_val_rmse_kw"] < info["zero_shot_val_rmse_kw"] else "does NOT beat"
    print(f"[finetune] best {info.get('tag', '')}: val RMSE {info['best_val_rmse_kw']:.1f} kW "
          f"({verdict} zero-shot {info['zero_shot_val_rmse_kw']:.1f} kW)")

    model = ChronosForecaster(pipe, cfg, variant=info["variant"], context_length=info["context_length"],
                              tag=info["method"])
    model.notes += f"; best_step={info['best_step']}; lr={info['learning_rate']}"
    rows = []
    for task in _tasks(cfg, df, ["B"]):
        rows += run_forecaster(model, task, cfg, seed=seed)
        _save_intervals(cfg, model, task)
    del pipe, model
    free_device_memory()
    return rows


def _save_intervals(cfg: Config, model, task: ForecastTask) -> None:
    """Store the 10-90% prediction interval and report its empirical coverage."""
    out = results_dir(cfg) / "predictions"
    for key, (lo, hi) in getattr(model, "last_intervals", {}).items():
        split = key.rsplit("_", 1)[-1]
        y = task.y(split)
        cov = float(np.mean((y >= lo) & (y <= hi)))
        pd.DataFrame({"target_time": task.target_index(split), "q10": lo, "q90": hi}).to_parquet(
            out / f"{task.name}__{model.name}__{split}__intervals.parquet", index=False
        )
        print(f"  [{task.name} {split}] {model.name} 80% interval coverage = {cov:.3f}")
    model.last_intervals = {}


def ensure_processed(cfg: Config) -> None:
    if not resolve_path(cfg["paths"]["processed"]).exists():
        stage_prepare(cfg)


__all__ = [
    "stage_prepare", "stage_baselines", "stage_paper", "stage_chronos", "stage_finetune",
    "ensure_processed", "evaluate_predictions",
]
