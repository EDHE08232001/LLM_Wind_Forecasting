"""Demo / inference pipeline: forecast the test period with a few models and plot a sample week.

Uses saved checkpoints when they exist (paper nets from 03_replicate_paper.py, Chronos-2
adapters from 05_finetune_chronos.py), otherwise trains a quick version on the fly.
Nothing is written to results/metrics.csv; outputs go to results/demo/ (git-ignored).

    python scripts/demo.py --synthetic                  # no downloads needed (fake data)
    python scripts/demo.py                              # real data (after 00/01 scripts)
    python scripts/demo.py --horizon 24 --chronos       # + Chronos-2 zero-shot
    python scripts/demo.py --chronos-adapter checkpoints/chronos2/lora/ctx512_lr0.0001
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from _common import ROOT, base_arg_parser, load_config

from windfc.config import resolve_path
from windfc.data import load_processed
from windfc.device import describe_environment, free_device_memory, get_device
from windfc.evaluate import evaluate_predictions
from windfc.models.baselines import Persistence, XGBoostModel
from windfc.pipeline import ensure_processed
from windfc.report import _plt, model_color
from windfc.splits import ForecastTask
from windfc.synthetic import make_synthetic
from windfc.train import NetForecaster, load_net_checkpoint, predict_array


def find_chronos_adapter(cfg) -> Path | None:
    root = resolve_path(cfg["paths"]["checkpoints_dir"]) / "chronos2"
    metas = sorted(root.glob("*/*/windfc_finetune.json")) if root.exists() else []
    if not metas:
        return None
    import json

    best = min(metas, key=lambda p: json.loads(p.read_text())["best_val_rmse_kw"])
    return best.parent


def main() -> None:
    parser = base_arg_parser(__doc__)
    parser.add_argument("--synthetic", action="store_true", help="use synthetic data (configs/smoke.yaml)")
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--arch", default="lstm", choices=["ann", "cnn", "rnn", "lstm"])
    parser.add_argument("--quick-epochs", type=int, default=15, help="epochs if no checkpoint exists")
    parser.add_argument("--chronos", action="store_true", help="also run Chronos-2 zero-shot (downloads ~0.5 GB once)")
    parser.add_argument("--chronos-adapter", default=None, help="fine-tuned Chronos-2 dir ('auto' = best saved)")
    parser.add_argument("--week", default=None, help="start of the plotted week, e.g. '2018-11-12 00:00'")
    args = parser.parse_args()
    if args.synthetic and args.config.endswith("default.yaml"):
        args.config = "configs/smoke.yaml"
    cfg = load_config(args.config, args.overrides)
    print(describe_environment())

    if args.synthetic:
        scada = resolve_path(cfg["paths"]["scada_csv"])
        if not scada.exists():
            make_synthetic(scada.parent, year=int(cfg["data"]["year"]), seed=int(cfg["seed"]))
    ensure_processed(cfg)
    df = load_processed(cfg)
    task = ForecastTask(df, cfg, "B", args.horizon)
    print(task.describe())

    preds: dict[str, np.ndarray] = {}
    preds["persistence"] = Persistence().predict(task, "test")

    xgb = XGBoostModel(cfg["baselines"]["xgboost"], device=str(get_device(cfg.get("device", "auto"))))
    xgb.cfg = {**xgb.cfg, "grid": {k: v[:1] for k, v in xgb.cfg["grid"].items()}}  # one grid point = fast
    preds["xgboost"] = xgb.fit(task).predict(task, "test")

    ckpt = resolve_path(cfg["paths"]["checkpoints_dir"]) / "paper_nets" / f"{task.name}__{args.arch}__seed0.pt"
    if ckpt.exists():
        print(f"[demo] loading {ckpt.relative_to(ROOT) if ckpt.is_relative_to(ROOT) else ckpt}")
        model, meta = load_net_checkpoint(ckpt, cfg.get("device", "auto"))
        x, _ = task.windows("test")
        preds[args.arch] = task.unscale_y(predict_array(model, x, next(model.parameters()).device))
    else:
        print(f"[demo] no checkpoint for {args.arch}; quick training for {args.quick_epochs} epochs")
        tcfg = {**cfg["train"], "epochs": args.quick_epochs, "batch_size": 128, "save_checkpoints": False}
        net = NetForecaster(args.arch, tcfg, device=cfg.get("device", "auto"), seed=0)
        preds[args.arch] = net.fit(task).predict(task, "test")
        net.release()

    if args.chronos or args.chronos_adapter:
        from windfc.models.chronos import ChronosForecaster, load_finetuned, load_pipeline

        if args.chronos:
            pipe = load_pipeline(cfg["chronos"]["model_id"], cfg.get("device", "auto"))
            m = ChronosForecaster(pipe, cfg, variant="past_covariates", tag="zeroshot")
            preds[m.name] = m.predict(task, "test")
            del pipe, m
            free_device_memory()
        if args.chronos_adapter:
            path = find_chronos_adapter(cfg) if args.chronos_adapter == "auto" else Path(args.chronos_adapter)
            if path is None or not Path(path).exists():
                print("[demo] no fine-tuned Chronos-2 adapter found; run scripts/05_finetune_chronos.py")
            else:
                pipe, info = load_finetuned(Path(path), cfg.get("device", "auto"))
                m = ChronosForecaster(pipe, cfg, variant=info["variant"], context_length=info["context_length"],
                                      tag=info["method"])
                preds[m.name] = m.predict(task, "test")
                del pipe, m
                free_device_memory()

    print("\nTest-set metrics (not logged):")
    for name, p in preds.items():
        evaluate_predictions(task, "test", p, name, cfg, save=False)

    # Plot a sample week.
    out = resolve_path("results/demo")
    out.mkdir(parents=True, exist_ok=True)
    idx = task.target_index("test")
    start = pd.Timestamp(args.week or cfg["report"]["sample_week_start"])
    if not ((idx >= start) & (idx < start + pd.Timedelta(days=7))).any():
        start = idx[0]
    w = (idx >= start) & (idx < start + pd.Timedelta(days=7))
    plt = _plt()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(idx[w], task.y("test")[w], color="#0b0b0b", lw=2, label="actual")
    for name, p in preds.items():
        ax.plot(idx[w], p[w], lw=1.5, label=name, color=model_color(name), ls="--" if name == "persistence" else "-")
    ax.set_ylabel("Active power (kW)")
    ax.set_title(f"Demo: {args.horizon}-hour-ahead forecasts, week of {start:%Y-%m-%d}")
    ax.legend(ncol=5, fontsize=8, loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    path = out / f"demo_week_h{args.horizon}.png"
    fig.savefig(path, dpi=150)
    pd.DataFrame({"target_time": idx, "y_true": task.y("test"), **{f"pred_{k}": v for k, v in preds.items()}}) \
        .to_csv(out / f"demo_predictions_h{args.horizon}.csv", index=False)
    print(f"\n[demo] figure -> {path}\n[demo] predictions -> {out / f'demo_predictions_h{args.horizon}.csv'}")


if __name__ == "__main__":
    main()
