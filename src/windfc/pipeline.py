"""The pipeline stages shared by the numbered scripts and ``scripts/run_all.py``."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import Config, resolve_path
from .data import build_hourly_dataset, load_processed
from .device import free_device_memory, get_device
from .evaluate import evaluate_predictions
from .splits import ForecastTask
from .utils import banner, checkpoints_dir, results_dir, save_run_config

EVAL_SPLITS = ("val", "test")


def stage_prepare(cfg: Config) -> dict:
    banner("Data pipeline")
    df, report = build_hourly_dataset(cfg)
    for k, v in report.items():
        print(f"  {k:>28}: {v}")
    print(f"  final frame: {df.shape[0]} rows x {df.shape[1]} cols, clean rows={int(df['is_clean'].sum())}")
    return report


def ensure_processed(cfg: Config) -> None:
    if not resolve_path(cfg["paths"]["processed"]).exists():
        stage_prepare(cfg)


def _tasks(cfg: Config, df: pd.DataFrame) -> list[ForecastTask]:
    return [ForecastTask(df, cfg, int(h)) for h in cfg["evaluation"]["horizons"]]


def stage_references(cfg: Config) -> list[dict]:
    """Two no-training references that put the Chronos-2 numbers in context."""
    banner("References (no training): persistence and the theoretical power curve")
    rows = []
    for task in _tasks(cfg, load_processed(cfg)):
        print(task.describe())
        for split in EVAL_SPLITS:
            rows += evaluate_predictions(task, split, task.persistence(split), "persistence", cfg,
                                         family="reference", notes="power at the forecast origin")
            rows += evaluate_predictions(task, split, task.power_curve(split), "power_curve", cfg,
                                         family="reference", variant="paper_inputs",
                                         notes="manufacturer curve at the target hour's wind speed")
    return rows


def _evaluate_chronos(cfg: Config, model, tasks: list[ForecastTask]) -> list[dict]:
    rows = []
    for task in tasks:
        for split in EVAL_SPLITS:
            pred, interval = model.predict(task, split)
            rows += evaluate_predictions(task, split, pred, model.name, cfg, family=model.family,
                                         variant=model.variant, notes=model.notes, interval=interval)
    model.clear_cache()
    return rows


def stage_zeroshot(cfg: Config, variants: list[str] | None = None) -> list[dict]:
    banner("Chronos-2 zero-shot")
    from .chronos import ChronosForecaster, load_pipeline

    df = load_processed(cfg)
    save_run_config(cfg, results_dir(cfg) / "runs", "zeroshot")
    pipe = load_pipeline(cfg["chronos"]["model_id"], cfg.get("device", "auto"))
    tasks = _tasks(cfg, df)
    rows = []
    for variant in variants or list(cfg["chronos"]["variants"]):
        rows += _evaluate_chronos(cfg, ChronosForecaster(pipe, cfg, variant=variant, tag="zeroshot"), tasks)
    del pipe
    free_device_memory()
    return rows


def adapter_root(cfg: Config, method: str, variant: str) -> Path:
    return checkpoints_dir(cfg) / "chronos2" / method / variant


def stage_finetune(cfg: Config, method: str | None = None, variants: list[str] | None = None,
                   grid: bool = True) -> list[dict]:
    """Fine-tune one adapter per variant (grid on validation), then score the best one on the test months."""
    from .chronos import ChronosForecaster, grid_search_finetune, qlora_unavailable_reason

    df = load_processed(cfg)
    fcfg = cfg["finetune"]
    method = (method or fcfg.get("method", "lora")).lower()
    device = cfg.get("device", "auto")
    if method == "qlora":
        reason = qlora_unavailable_reason(get_device(device))
        if reason:
            print(f"[finetune] {reason}.\n[finetune] Falling back to LoRA (full-precision base weights).")
            method = "lora"
    banner(f"Chronos-2 fine-tuning ({method.upper()})")
    seed = int(cfg.get("seed", 42))
    res = results_dir(cfg)
    save_run_config(cfg, res / "runs", f"finetune_{method}")
    tasks = _tasks(cfg, df)
    rows = []
    for variant in variants or list(fcfg["variants"]):
        pipe, info, grid_rows = grid_search_finetune(df, cfg, variant, method=method, device=device, seed=seed,
                                                     out_root=adapter_root(cfg, method, variant), grid=grid)
        stem = f"{method}_{variant}"
        pd.DataFrame(grid_rows).to_csv(res / "tables" / f"finetune_grid_{stem}.csv", index=False)
        (res / "tables" / f"finetune_best_{stem}.json").write_text(json.dumps(info, indent=2, default=float))
        gain = 100 * (1 - info["best_val_rmse_kw"] / info["zero_shot_val_rmse_kw"])
        print(f"[finetune] best for {variant}: {info['tag']} at step {info['best_step']}, validation RMSE "
              f"{info['best_val_rmse_kw']:.1f} kW vs {info['zero_shot_val_rmse_kw']:.1f} kW zero-shot "
              f"({gain:+.1f}%)" + ("" if info["best_step"] else "  -> fine-tuning did not help; zero-shot kept"))
        model = ChronosForecaster(pipe, cfg, variant=variant, context_length=info["context_length"], tag=method)
        model.notes += f"; lr={info['learning_rate']:g}; best_step={info['best_step']}/{info['num_steps']}"
        rows += _evaluate_chronos(cfg, model, tasks)
        del pipe, model
        free_device_memory()
    return rows


def find_adapters(cfg: Config) -> list[Path]:
    """The best saved adapter (lowest validation RMSE) for every method x variant."""
    root = checkpoints_dir(cfg) / "chronos2"
    best: dict[tuple[str, str], tuple[float, Path]] = {}
    for meta in sorted(root.glob(f"*/*/*/{'windfc_finetune.json'}")):
        info = json.loads(meta.read_text())
        key = (info["method"], info["variant"])
        if key not in best or info["best_val_rmse_kw"] < best[key][0]:
            best[key] = (info["best_val_rmse_kw"], meta.parent)
    return [p for _, p in best.values()]


def stage_evaluate_saved(cfg: Config, adapters: list[str] | None = None) -> list[dict]:
    """Re-score saved adapters without training (all best adapters, or the given directories)."""
    banner("Chronos-2: evaluating saved adapters")
    from .chronos import ChronosForecaster, load_finetuned

    paths = [resolve_path(a) for a in adapters] if adapters else find_adapters(cfg)
    if not paths:
        raise FileNotFoundError("No saved adapters found under checkpoints/. Run scripts/03_finetune.py first.")
    tasks = _tasks(cfg, load_processed(cfg))
    rows = []
    for path in paths:
        pipe, info = load_finetuned(path, cfg.get("device", "auto"))
        print(f"[evaluate] {path}  (validation RMSE {info['best_val_rmse_kw']:.1f} kW)")
        model = ChronosForecaster(pipe, cfg, variant=info["variant"], context_length=info["context_length"],
                                  tag=info["method"])
        model.notes += f"; lr={info['learning_rate']:g}; best_step={info['best_step']}/{info['num_steps']}"
        rows += _evaluate_chronos(cfg, model, tasks)
        del pipe, model
        free_device_memory()
    return rows
