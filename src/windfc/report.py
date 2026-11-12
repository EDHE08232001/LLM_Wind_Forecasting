"""Tables and figures built from ``results/metrics.csv`` and ``results/predictions/``.

Outputs under ``results/``:

* ``tables/main_results.{csv,md}``      - every model x horizon x subset on the test months, in kW
* ``tables/paper_comparison.{csv,md}``  - the paper's Table 5 next to our models, in the paper's units
* ``tables/slides_results.md``          - the slide table with its TBD row filled in
* ``tables/finetune_summary.{csv,md}``  - what the LoRA / QLoRA runs selected
* ``tables/bootstrap.csv``              - paired block-bootstrap CIs (fine-tuned vs zero-shot, vs persistence)
* ``tables/binned_errors.csv``          - error by wind-speed bin (the paper's weak spot: above 13 m/s)
* ``figures/*.png``                     - paper comparison, error by wind bin, power curve, sample week,
                                          fine-tuning curves
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .metrics import binned_errors, bootstrap_rmse_diff
from .paper import PAPER_BEST, PAPER_CITATION, PAPER_TABLE5
from .utils import banner, results_dir

# One fixed colour per model group, so a group looks the same in every figure.
GROUP_COLORS = {"paper": "#1baf7a", "reference": "#8a8984", "zeroshot": "#2a78d6", "finetuned": "#eb6834"}
GROUP_LABELS = {"paper": "Paper (Karaman 2023)", "reference": "Reference, no training",
                "zeroshot": "Chronos-2 zero-shot", "finetuned": "Chronos-2 fine-tuned"}
TEXT, TEXT2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
VARIANT_LABELS = {"univariate": "power history only", "past_covariates": "past covariates",
                  "paper_inputs": "paper inputs"}
KEY_COLS = ["model", "horizon", "split", "subset"]


def model_group(model: str) -> str:
    if model.startswith("chronos2_zeroshot"):
        return "zeroshot"
    if model.startswith("chronos2_"):
        return "finetuned"
    return "reference"


def model_label(model: str) -> str:
    if model == "persistence":
        return "Persistence"
    if model == "power_curve":
        return "Theoretical power curve"
    _, tag, variant = model.split("_", 2)
    how = {"zeroshot": "zero-shot", "lora": "+ LoRA", "qlora": "+ QLoRA"}.get(tag, tag)
    return f"Chronos-2 {how} ({VARIANT_LABELS.get(variant, variant)})"


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def to_markdown(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    def fmt(v):
        if isinstance(v, float | np.floating):
            return "" if np.isnan(v) else format(v, floatfmt)
        return str(v)

    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(fmt(r[c]) for c in cols) + " |" for _, r in df.iterrows()]
    return "\n".join(lines) + "\n"


def write_table(df: pd.DataFrame, path_stem: Path, floatfmt: str = ".4f", note: str = "") -> None:
    df.to_csv(path_stem.with_suffix(".csv"), index=False)
    path_stem.with_suffix(".md").write_text(to_markdown(df, floatfmt) + (f"\n{note}\n" if note else ""))


def load_metrics(res: Path) -> pd.DataFrame:
    path = res / "metrics.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run the experiment scripts first.")
    m = pd.read_csv(path)
    return m.sort_values("timestamp").drop_duplicates(KEY_COLS, keep="last")  # the latest run wins


def load_predictions(res: Path, task_name: str, split: str = "test") -> dict[str, pd.DataFrame]:
    return {p.name.split("__")[1]: pd.read_parquet(p)
            for p in sorted((res / "predictions").glob(f"{task_name}__*__{split}.parquet"))}


def paper_comparison(test: pd.DataFrame) -> pd.DataFrame:
    """Paper Table 5 followed by our test-set rows, all in the paper's min-max-scaled units."""
    paper = PAPER_TABLE5.rename(columns={"model": "model"}).assign(
        source="paper", setting="same-hour inputs, random 75/25 split", horizon_h=np.nan, r2_pearson=np.nan)
    paper["rows"] = "clean"
    ours = test.assign(source="ours", horizon_h=test["horizon"].astype(float), rows=test["subset"],
                       setting="chronological split, test = Nov-Dec")
    ours = ours.assign(model=ours["model"].map(model_label))
    cols = ["source", "model", "setting", "horizon_h", "rows", "mae_scaled", "mse_scaled", "rmse_scaled", "r2",
            "r2_pearson"]
    ours = ours.sort_values(["rows", "horizon_h", "rmse_scaled"], ascending=[False, True, True])
    return pd.concat([paper[cols], ours[cols]], ignore_index=True)


def pick_headline(test: pd.DataFrame, cfg: Config) -> pd.Series | None:
    """The fine-tuned row that goes next to the paper's (falls back to zero-shot if none exists)."""
    h = cfg["report"]["headline"]
    sub = test[(test["variant"] == h["variant"]) & (test["horizon"] == int(h["horizon"]))
               & (test["subset"] == h["subset"]) & (test["family"] == "chronos2")]
    tuned = sub[~sub["model"].str.contains("zeroshot")]
    pool = tuned if len(tuned) else sub
    return None if pool.empty else pool.sort_values("rmse_kw").iloc[0]


def slides_table(test: pd.DataFrame, cfg: Config) -> str:
    rows = [{"Model": r["model"] + (" (best in paper)" if r["model"] == PAPER_BEST else ""), "MAE": r["mae_scaled"],
             "MSE": r["mse_scaled"], "RMSE": r["rmse_scaled"], "R2": r["r2"]} for _, r in PAPER_TABLE5.iterrows()]
    head = pick_headline(test, cfg)
    note = "No Chronos-2 result found yet for the headline setting."
    if head is not None:
        rows.append({"Model": model_label(head["model"]), "MAE": head["mae_scaled"], "MSE": head["mse_scaled"],
                     "RMSE": head["rmse_scaled"], "R2": head["r2"]})
        h = cfg["report"]["headline"]
        note = (
            f"Paper rows: {PAPER_CITATION} (random 75/25 split, cleaned rows, min-max-scaled errors).\n"
            f"Our row: {head['model']}, horizon {h['horizon']} h, test months (chronological split), "
            f"'{h['subset']}' rows, errors divided by the training min-max range of active power.\n"
            "The split differs (ours is chronological and harder), and Chronos-2 also sees the power history, "
            "which the paper's networks did not. See tables/paper_comparison.md for every variant and horizon."
        )
    return "## Results on the test set (slide table)\n\n" + to_markdown(pd.DataFrame(rows)) + "\n" + note + "\n"


def finetune_summary(tables: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(tables.glob("finetune_best_*.json")):
        i = json.loads(p.read_text())
        rows.append({
            "method": i["method"], "variant": i["variant"], "context_h": i["context_length"],
            "learning_rate": i["learning_rate"], "best_step": i["best_step"], "num_steps": i["num_steps"],
            "val_rmse_zero_shot_kw": i["zero_shot_val_rmse_kw"], "val_rmse_best_kw": i["best_val_rmse_kw"],
            "val_gain_pct": 100 * (1 - i["best_val_rmse_kw"] / i["zero_shot_val_rmse_kw"]),
            "trainable_params": i["trainable_params"], "total_params": i.get("total_params"),
            "train_seconds": i.get("train_seconds"), "device": i.get("device"),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT2, "xtick.color": TEXT2, "ytick.color": TEXT2,
        "text.color": TEXT, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.axisbelow": True,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
    })
    return plt


def _save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return path


def fig_paper_comparison(test: pd.DataFrame, cfg: Config, out: Path) -> Path | None:
    """Scaled RMSE and R^2: the paper's four networks and our models in the headline setting."""
    h = cfg["report"]["headline"]
    ours = test[(test["horizon"] == int(h["horizon"])) & (test["subset"] == h["subset"])]
    if ours.empty:
        return None
    bars = [(r["model"], "paper", r["rmse_scaled"], r["r2"]) for _, r in PAPER_TABLE5.iterrows()]
    bars += [(model_label(r["model"]), model_group(r["model"]), r["rmse_scaled"], r["r2"])
             for _, r in ours.sort_values("rmse_scaled", ascending=False).iterrows()]
    labels, groups = [b[0] for b in bars], [b[1] for b in bars]
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.42 * len(bars) + 1.6), sharey=True)
    for ax, col, title, fmt in ((axes[0], 2, "RMSE, min-max scaled (lower is better)", "{:.4f}"),
                                (axes[1], 3, "R² (higher is better)", "{:.3f}")):
        vals = np.array([b[col] for b in bars], dtype=float)
        y = np.arange(len(bars))[::-1]
        ax.barh(y, vals, height=0.62, color=[GROUP_COLORS[g] for g in groups])
        span = np.nanmax(np.abs(vals)) or 1.0
        for yi, v in zip(y, vals):
            ax.text(max(v, 0) + 0.012 * span, yi, fmt.format(v), va="center", fontsize=8.5, color=TEXT)
        ax.set_yticks(y, labels)
        ax.set_title(title)
        ax.set_xlim(min(0.0, np.nanmin(vals) * 1.1), span * 1.14)
        ax.grid(axis="y", visible=False)
    from matplotlib.patches import Patch

    fig.suptitle(f"Paper Table 5 vs Chronos-2: {h['horizon']} h ahead, '{h['subset']}' test hours (Nov-Dec)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0.07, 1, 1))  # leave a strip under the plots for the legend
    fig.legend(handles=[Patch(color=GROUP_COLORS[g], label=GROUP_LABELS[g]) for g in GROUP_COLORS if g in groups],
               loc="lower center", ncol=4, fontsize=9)
    path = out / "paper_comparison.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def fig_binned(binned: pd.DataFrame, task_name: str, models: list[str], out: Path) -> Path:
    plt = _plt()
    sub = binned[binned["task"] == task_name]
    bins = list(dict.fromkeys(sub["wind_bin"]))
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    width = 0.8 / max(len(models), 1)
    x = np.arange(len(bins))
    hatches = ["", "////", "....", "xxxx"]  # tells models of one group apart without relying on colour
    seen: dict[str, int] = {}
    for i, m in enumerate(models):
        g = model_group(m)
        k = seen.get(g, 0)
        seen[g] = k + 1
        vals = sub[sub["model"] == m].set_index("wind_bin").reindex(bins)["rmse_kw"].to_numpy()
        ax.bar(x + (i - (len(models) - 1) / 2) * width, vals, width * 0.9, color=GROUP_COLORS[g],
               hatch=hatches[k % len(hatches)], edgecolor=SURFACE, linewidth=1, label=model_label(m))
    counts = sub[sub["model"] == models[0]].set_index("wind_bin").reindex(bins)["n"]
    ax.set_xticks(x, [f"{b} m/s\n(n={int(n)})" for b, n in zip(bins, counts.fillna(0))])
    ax.set_ylabel("RMSE (kW)")
    ax.set_title(f"Test RMSE by wind speed at the target hour, {task_name[1:]} h ahead")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    return _save(fig, out / f"binned_rmse_{task_name}.png")


def fig_power_curve(d: pd.DataFrame, model: str, task_name: str, out: Path) -> Path:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    ax.scatter(d["wind_speed"], d["y_true"], s=9, color="#8a8984", alpha=0.55, label="Actual", linewidths=0)
    ax.scatter(d["wind_speed"], d["y_pred"], s=9, color=GROUP_COLORS[model_group(model)], alpha=0.55,
               label=model_label(model), linewidths=0)
    ax.set_xlabel("Wind speed at the target hour (m/s)")
    ax.set_ylabel("Active power (kW)")
    ax.set_title(f"Power curve, actual vs predicted ({task_name[1:]} h ahead, test)")
    ax.legend(loc="lower right", markerscale=2, fontsize=8)
    return _save(fig, out / f"power_curve_{task_name}.png")


def fig_sample_week(preds: dict[str, pd.DataFrame], models: list[str], band_model: str | None, task_name: str,
                    start: str, out: Path) -> Path:
    plt = _plt()
    first = next(iter(preds.values()))
    t0 = pd.Timestamp(start)
    if not ((first["target_time"] >= t0) & (first["target_time"] < t0 + pd.Timedelta(days=7))).any():
        t0 = first["target_time"].iloc[0]
    t1 = t0 + pd.Timedelta(days=7)
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    if band_model and "q10" in preds[band_model]:
        d = preds[band_model]
        w = (d["target_time"] >= t0) & (d["target_time"] < t1)
        ax.fill_between(d.loc[w, "target_time"], d.loc[w, "q10"], d.loc[w, "q90"], linewidth=0, alpha=0.18,
                        color=GROUP_COLORS[model_group(band_model)], label="10-90% interval")
    w = (first["target_time"] >= t0) & (first["target_time"] < t1)
    ax.plot(first.loc[w, "target_time"], first.loc[w, "y_true"], color=TEXT, lw=2, label="Actual")
    for m in models:
        d = preds[m]
        w = (d["target_time"] >= t0) & (d["target_time"] < t1)
        ax.plot(d.loc[w, "target_time"], d.loc[w, "y_pred"], color=GROUP_COLORS[model_group(m)], lw=1.6,
                ls="--" if model_group(m) == "reference" else "-", label=model_label(m))
    ax.set_ylabel("Active power (kW)")
    ax.set_title(f"Forecast vs actual, {task_name[1:]} h ahead, week of {t0:%d %b %Y}")
    ax.legend(ncol=2, loc="upper left", fontsize=8)
    fig.autofmt_xdate()
    return _save(fig, out / f"sample_week_{task_name}.png")


def fig_finetune_curves(tables: Path, ckpt_root: Path, out: Path) -> list[Path]:
    """Validation RMSE against training step for every grid point of every fine-tuning run."""
    plt = _plt()
    paths = []
    for best in sorted(tables.glob("finetune_best_*.json")):
        info = json.loads(best.read_text())
        runs = sorted((ckpt_root / "chronos2" / info["method"] / info["variant"]).glob("*/windfc_finetune.json"))
        runs = [json.loads(r.read_text()) for r in runs] or [info]
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        styles = ["-", "--", ":", "-."]
        for i, r in enumerate(runs):
            is_best = (r["context_length"], r["learning_rate"]) == (info["context_length"], info["learning_rate"])
            hist = pd.DataFrame(r["history"])
            ax.plot(hist["step"], hist["val_rmse_kw"], marker="o", ms=4, lw=2.2 if is_best else 1.4,
                    ls=styles[i % len(styles)], color=GROUP_COLORS["finetuned"] if is_best else "#8a8984",
                    label=f"context {r['context_length']} h, lr {r['learning_rate']:g}"
                          + ("  (selected)" if is_best else ""))
        ax.axhline(info["zero_shot_val_rmse_kw"], color=GROUP_COLORS["zeroshot"], lw=1.2, ls=(0, (1, 2)))
        ax.annotate("zero-shot (selected context)", (ax.get_xlim()[1], info["zero_shot_val_rmse_kw"]),
                    ha="right", va="bottom", fontsize=8, color=TEXT2)
        ax.set_xlabel("Training step")
        ax.set_ylabel("Validation RMSE (kW)")
        ax.ticklabel_format(axis="y", useOffset=False)
        name = {"lora": "LoRA", "qlora": "QLoRA"}.get(info["method"], info["method"])
        ax.set_title(f"{name} fine-tuning, {VARIANT_LABELS.get(info['variant'], info['variant'])}")
        ax.legend(fontsize=8)
        paths.append(_save(fig, out / f"finetune_curve_{info['method']}_{info['variant']}.png"))
    return paths


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------
def make_report(cfg: Config) -> dict[str, Path]:
    banner("Report")
    from .utils import checkpoints_dir

    res = results_dir(cfg)
    tables, figs = res / "tables", res / "figures"
    m = load_metrics(res)
    test = m[m["split"] == "test"].copy()
    written: dict[str, Path] = {}

    main = test.sort_values(["horizon", "subset", "rmse_kw"])[
        ["model", "variant", "horizon", "subset", "n", "mae_kw", "rmse_kw", "r2", "mae_scaled", "rmse_scaled",
         "skill_vs_persistence", "interval_coverage"]]
    write_table(main, tables / "main_results", ".3f")
    written["main"] = tables / "main_results.csv"
    print(to_markdown(main[main["subset"] == "all"].drop(columns="subset"), ".3f"))

    note = (
        f"Paper rows are quoted from {PAPER_CITATION}; its networks were not re-trained here.\n"
        "Scaled errors = kW errors divided by the training min-max range of active power (MSE by its square).\n"
        "r2 = 1 - SS_res/SS_tot. r2_pearson = squared correlation, which is how the paper's Eq. (2) defines R2.\n"
        "rows: clean = only hours passing the paper's cleaning rules; all = every hour.\n"
        "Only the 'paper inputs' models and the power curve use the paper's information (same-hour wind speed, "
        "theoretical power, weather). The paper used a random split; ours is chronological."
    )
    write_table(paper_comparison(test), tables / "paper_comparison", note=note)
    written["paper_comparison"] = tables / "paper_comparison.md"
    (tables / "slides_results.md").write_text(slides_table(test, cfg))
    written["slides"] = tables / "slides_results.md"

    ft = finetune_summary(tables)
    if not ft.empty:
        write_table(ft, tables / "finetune_summary", ".4g")

    binned_rows, boot_rows = [], []
    n_boot = int(cfg["report"].get("bootstrap_samples", 2000))
    for h in cfg["evaluation"]["horizons"]:
        tname = f"h{int(h)}"
        preds = load_predictions(res, tname)
        if not preds:
            continue
        for model, d in preds.items():
            b = binned_errors(d["y_true"], d["y_pred"], d["wind_speed"], cfg["evaluation"]["wind_bins"])
            b.insert(0, "model", model)
            b.insert(0, "task", tname)
            binned_rows.append(b)
            if model != "persistence":
                boot_rows.append({"task": tname, "model": model, "versus": "persistence",
                                  **bootstrap_rmse_diff(d["y_true"], d["y_pred"], d["persistence"], n_boot)})
            if model_group(model) == "finetuned":
                zero = "chronos2_zeroshot_" + model.split("_", 2)[2]
                if zero in preds:
                    boot_rows.append({"task": tname, "model": model, "versus": zero,
                                      **bootstrap_rmse_diff(d["y_true"], d["y_pred"], preds[zero]["y_pred"], n_boot)})
        ranked = [x for x in test[(test["horizon"] == int(h)) & (test["subset"] == "all")]
                  .sort_values("rmse_kw")["model"] if x in preds]
        chronos = [x for x in ranked if model_group(x) != "reference"]
        if not chronos:
            continue
        # shown: the best fine-tuned model (else the best zero-shot one), its zero-shot twin, persistence
        tuned = [x for x in chronos if model_group(x) == "finetuned"]
        best = tuned[0] if tuned else chronos[0]
        twin = "chronos2_zeroshot_" + best.split("_", 2)[2]
        shown = list(dict.fromkeys([best] + ([twin] if twin in preds else []) + ["persistence"]))
        binned = pd.concat(binned_rows, ignore_index=True)
        written[f"binned_{tname}"] = fig_binned(binned, tname, shown, figs)
        written[f"curve_{tname}"] = fig_power_curve(preds[best], best, tname, figs)
        fig_sample_week(preds, shown, best, tname, cfg["report"]["sample_week_start"], figs)
    if binned_rows:
        pd.concat(binned_rows, ignore_index=True).to_csv(tables / "binned_errors.csv", index=False)
    if boot_rows:
        pd.DataFrame(boot_rows).to_csv(tables / "bootstrap.csv", index=False)
    p = fig_paper_comparison(test, cfg, figs)
    if p:
        written["fig_paper"] = p
    fig_finetune_curves(tables, checkpoints_dir(cfg), figs)
    print((tables / "slides_results.md").read_text())
    print(f"Tables  -> {tables}\nFigures -> {figs}")
    return written
