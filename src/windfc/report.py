"""Tables and figures generated from results/metrics.csv and results/predictions/.

Outputs (under ``results/``):

* ``tables/main_results.{csv,md,tex}``    - every model x protocol x horizon (test split)
* ``tables/protocol_a_vs_b.{csv,md}``     - how much the random split flatters the paper nets
* ``tables/paper_comparison.md``          - our protocol-A numbers next to paper Table 5
* ``tables/bootstrap_vs_persistence.csv`` - paired block-bootstrap CI on RMSE difference
* ``tables/binned_errors.csv``            - error by wind-speed bin (focus > 13 m/s)
* ``tables/slides_results.md``            - the slide table, filled from metrics.csv
* ``figures/*.png``                       - error-by-wind-bin, power curve, sample week, skill
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .metrics import binned_errors, bootstrap_rmse_diff
from .utils import banner, results_dir

PAPER_TABLE5 = pd.DataFrame(
    {
        "model": ["ann", "cnn", "rnn", "lstm"],
        "paper_mae": [0.0245, 0.0235, 0.0218, 0.0209],
        "paper_mse": [0.0062, 0.0055, 0.0043, 0.0038],
        "paper_rmse": [0.0787, 0.0742, 0.0656, 0.0614],
        "paper_r2": [0.9301, 0.9378, 0.9514, 0.9574],
    }
)

# Validated categorical palette (dataviz reference instance, light mode), assigned by
# a fixed per-model map so a model keeps its colour in every figure.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TEXT, TEXT2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
ACTUAL_COLOR = "#0b0b0b"

KEY_COLS = ["model", "protocol", "horizon", "split", "subset", "seed"]
SHOW = ["mae_kw", "rmse_kw", "r2", "mae_scaled", "rmse_scaled", "skill_vs_persistence"]


# ---------------------------------------------------------------------------
# Table helpers (no tabulate / jinja2 dependency)
# ---------------------------------------------------------------------------
def to_markdown(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    def fmt(v):
        if isinstance(v, float | np.floating):
            return "" if np.isnan(v) else format(v, floatfmt)
        return str(v)

    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt(r[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def to_latex(df: pd.DataFrame, floatfmt: str = ".4f", caption: str = "") -> str:
    def fmt(v):
        if isinstance(v, float | np.floating):
            return "--" if np.isnan(v) else format(v, floatfmt)
        return str(v).replace("_", r"\_").replace("%", r"\%")

    cols = list(df.columns)
    out = [r"\begin{table}[ht]", r"\centering", r"\begin{tabular}{" + "l" * 3 + "r" * (len(cols) - 3) + "}",
           r"\hline", " & ".join(c.replace("_", r"\_") for c in cols) + r" \\", r"\hline"]
    for _, r in df.iterrows():
        out.append(" & ".join(fmt(r[c]) for c in cols) + r" \\")
    out += [r"\hline", r"\end{tabular}"]
    if caption:
        out.append(r"\caption{" + caption + "}")
    out.append(r"\end{table}")
    return "\n".join(out) + "\n"


def write_table(df: pd.DataFrame, path_stem: Path, caption: str = "", latex: bool = False) -> None:
    df.to_csv(path_stem.with_suffix(".csv"), index=False)
    path_stem.with_suffix(".md").write_text(to_markdown(df))
    if latex:
        path_stem.with_suffix(".tex").write_text(to_latex(df, caption=caption))


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_metrics(res: Path) -> pd.DataFrame:
    path = res / "metrics.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run the experiment scripts first.")
    m = pd.read_csv(path)
    m["seed"] = m["seed"].fillna(-1).astype(int)
    m = m.sort_values("timestamp").drop_duplicates(KEY_COLS, keep="last")  # latest run wins
    return m


def aggregate_seeds(m: pd.DataFrame) -> pd.DataFrame:
    group = ["model", "family", "protocol", "horizon", "split", "subset"]
    agg = m.groupby(group, dropna=False)[SHOW + ["n"]].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg = agg.reset_index()
    agg["n_seeds"] = m.groupby(group, dropna=False)["seed"].nunique().to_numpy()
    return agg


def load_predictions(res: Path, task_name: str, split: str = "test") -> dict[str, pd.DataFrame]:
    """One prediction frame per model (seed-averaged when several seeds exist)."""
    out: dict[str, list[pd.DataFrame]] = {}
    for p in sorted((res / "predictions").glob(f"{task_name}__*__{split}__seed*.parquet")):
        model = p.name.split("__")[1]
        out.setdefault(model, []).append(pd.read_parquet(p))
    merged = {}
    for model, frames in out.items():
        base = frames[0].copy()
        if len(frames) > 1:
            base["y_pred"] = np.mean([f["y_pred"].to_numpy() for f in frames], axis=0)
        merged[model] = base
    return merged


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
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
        "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
    })
    return plt


GROUP_ORDER = ["linear", "xgboost", "ann", "cnn", "rnn", "lstm", "chronos2_zeroshot", "chronos2_finetuned"]
GREY = "#8a8984"


def display_group(model: str) -> str:
    if model.startswith("chronos2_zeroshot"):
        return "chronos2_zeroshot"
    if model.startswith("chronos2_"):
        return "chronos2_finetuned"
    return model


def model_color(model: str) -> str:
    """Fixed colour per model *group* (8 validated slots); baselines without a slot are grey."""
    g = display_group(model)
    return PALETTE[GROUP_ORDER.index(g)] if g in GROUP_ORDER else GREY


def pick_models(ranked: list[str], k: int = 5, with_persistence: bool = True) -> list[str]:
    """Best model of each group (at most k), so every plotted series has a distinct colour."""
    seen, out = set(), []
    for m in ranked:
        g = display_group(m)
        if g in seen or g not in GROUP_ORDER:
            continue
        seen.add(g)
        out.append(m)
        if len(out) == k:
            break
    if with_persistence and "persistence" in ranked:
        out.append("persistence")
    return out


def fig_binned(binned: pd.DataFrame, task_name: str, out: Path, models: list[str]) -> Path:
    plt = _plt()
    sub = binned[binned["task"] == task_name]
    bins = list(dict.fromkeys(sub["wind_bin"]))
    fig, ax = plt.subplots(figsize=(9, 4.2))
    width = 0.8 / max(len(models), 1)
    x = np.arange(len(bins))
    for i, m in enumerate(models):
        vals = sub[sub["model"] == m].set_index("wind_bin").reindex(bins)["rmse_kw"].to_numpy()
        ax.bar(x + (i - (len(models) - 1) / 2) * width, vals, width * 0.92, color=model_color(m), label=m,
               edgecolor=SURFACE, linewidth=1)
    ax.set_xticks(x, [f"{b} m/s" for b in bins])
    ax.set_ylabel("RMSE (kW)")
    ax.set_title(f"Test RMSE by wind-speed bin ({task_name})")
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=min(4, len(models)), loc="upper left", fontsize=8)
    fig.tight_layout()
    path = out / f"binned_rmse_{task_name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_power_curve(preds: dict[str, pd.DataFrame], model: str, task_name: str, out: Path, color: str) -> Path:
    plt = _plt()
    d = preds[model]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter(d["wind_speed"], d["y_true"], s=8, color="#b5b4ae", alpha=0.6, label="actual", linewidths=0)
    ax.scatter(d["wind_speed"], d["y_pred"], s=8, color=color, alpha=0.6, label=f"predicted ({model})",
               linewidths=0)
    ax.set_xlabel("Wind speed at target hour (m/s)")
    ax.set_ylabel("Active power (kW)")
    ax.set_title(f"Power curve: actual vs predicted ({task_name}, test)")
    ax.legend(loc="lower right", markerscale=2)
    fig.tight_layout()
    path = out / f"power_curve_{task_name}_{model}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_sample_week(preds: dict[str, pd.DataFrame], models: list[str], task_name: str, start: str,
                    out: Path) -> Path | None:
    plt = _plt()
    t0 = pd.Timestamp(start)
    t1 = t0 + pd.Timedelta(days=7)
    first = next(iter(preds.values()))
    win = (first["target_time"] >= t0) & (first["target_time"] < t1)
    if win.sum() == 0:
        t0 = first["target_time"].iloc[0]
        t1 = t0 + pd.Timedelta(days=7)
        win = (first["target_time"] >= t0) & (first["target_time"] < t1)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(first.loc[win, "target_time"], first.loc[win, "y_true"], color=ACTUAL_COLOR, lw=2, label="actual")
    for m in models:
        d = preds[m]
        w = (d["target_time"] >= t0) & (d["target_time"] < t1)
        ax.plot(d.loc[w, "target_time"], d.loc[w, "y_pred"], color=model_color(m), lw=1.6, label=m,
                ls="--" if m == "persistence" else "-")
    ax.set_ylabel("Active power (kW)")
    ax.set_title(f"Forecast vs actual, sample week ({task_name})")
    ax.legend(ncol=5, loc="upper left", fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    path = out / f"sample_week_{task_name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_skill(table: pd.DataFrame, out: Path) -> Path | None:
    plt = _plt()
    sub = table[(table["protocol"] == "B") & table["skill_vs_persistence_mean"].notna()]
    if sub.empty:
        return None
    horizons = sorted(sub["horizon"].unique())
    fig, axes = plt.subplots(1, len(horizons), figsize=(5.5 * len(horizons), 4), squeeze=False)
    for ax, h in zip(axes[0], horizons):
        s = sub[(sub["horizon"] == h) & (sub["model"] != "persistence")].sort_values("skill_vs_persistence_mean")
        # one series -> one colour; bars are identified by their axis labels
        ax.barh(s["model"], s["skill_vs_persistence_mean"], color=PALETTE[0],
                xerr=s["skill_vs_persistence_std"].fillna(0), ecolor=TEXT2, height=0.6)
        ax.axvline(0, color=TEXT2, lw=1)
        ax.set_title(f"Skill vs persistence, h={h}")
        ax.set_xlabel("1 - RMSE / RMSE_persistence")
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    path = out / "skill_vs_persistence.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------
def make_report(cfg: Config) -> dict[str, Path]:
    banner("Phase 6: report")
    res = results_dir(cfg)
    tables, figs = res / "tables", res / "figures"
    m = load_metrics(res)
    agg = aggregate_seeds(m)
    written: dict[str, Path] = {}

    # Main table: test split, all targets.
    main = agg[(agg["split"] == "test") & (agg["subset"] == "all")].copy()
    main = main.sort_values(["protocol", "horizon", "rmse_kw_mean"])
    cols = ["model", "protocol", "horizon", "n_seeds", "mae_kw_mean", "mae_kw_std", "rmse_kw_mean", "rmse_kw_std",
            "r2_mean", "mae_scaled_mean", "rmse_scaled_mean", "skill_vs_persistence_mean"]
    pretty = main[cols].rename(columns=lambda c: c.replace("_mean", ""))
    write_table(pretty, tables / "main_results", caption="Test-set results (Nov--Dec 2018 for protocol B).",
                latex=True)
    written["main"] = tables / "main_results.csv"
    print(to_markdown(pretty, ".3f"))

    clean = agg[(agg["split"] == "test") & (agg["subset"] == "clean")]
    if not clean.empty:
        write_table(clean[cols].rename(columns=lambda c: c.replace("_mean", "")), tables / "main_results_clean")

    # Protocol A vs B for the paper nets.
    nets = main[main["family"] == "paper_net"]
    if not nets.empty:
        ab = nets.pivot_table(index="model", columns=["protocol", "horizon"], values="r2_mean")
        ab.columns = [f"R2_{p}" + (f"_h{h}" if p == "B" else "") for p, h in ab.columns]
        ab = ab.reset_index()
        write_table(ab, tables / "protocol_a_vs_b")

    # Comparison with paper Table 5 (scaled units).
    pa = main[(main["protocol"] == "A") & (main["family"] == "paper_net")]
    if not pa.empty:
        mse_s = m[(m["protocol"] == "A") & (m["split"] == "test")].groupby("model")["mse_scaled"].mean()
        comp = PAPER_TABLE5.merge(
            pa[["model", "mae_scaled_mean", "rmse_scaled_mean", "r2_mean"]], on="model", how="left"
        ).rename(columns={"mae_scaled_mean": "ours_mae", "rmse_scaled_mean": "ours_rmse", "r2_mean": "ours_r2"})
        comp["ours_mse"] = comp["model"].map(mse_s)
        comp = comp[["model", "paper_mae", "ours_mae", "paper_mse", "ours_mse", "paper_rmse", "ours_rmse",
                     "paper_r2", "ours_r2"]]
        write_table(comp, tables / "paper_comparison")

    # Per-task: binned errors, bootstrap vs persistence, figures.
    binned_rows, boot_rows = [], []
    task_names = ["A"] + [f"B_h{int(h)}" for h in cfg["protocol_b"]["horizons"]]
    for tname in task_names:
        preds = load_predictions(res, tname)
        if not preds:
            continue
        for model, d in preds.items():
            b = binned_errors(d["y_true"], d["y_pred"], d["wind_speed"], cfg["metrics"]["wind_bins"])
            b.insert(0, "model", model)
            b.insert(0, "task", tname)
            binned_rows.append(b)
            if "persistence" in d and tname != "A" and model != "persistence":
                bs = bootstrap_rmse_diff(d["y_true"], d["y_pred"], d["persistence"],
                                         n_boot=int(cfg["report"].get("bootstrap_samples", 2000)))
                boot_rows.append({"task": tname, "model": model, **bs})
        binned = pd.concat(binned_rows, ignore_index=True)
        ranking = main[(main["protocol"] == tname[0]) &
                       (main["horizon"] == (int(tname.split("h")[1]) if tname != "A" else 0))]
        ranked = [x for x in ranking.sort_values("rmse_kw_mean")["model"] if x in preds]
        if ranked:
            written[f"binned_{tname}"] = fig_binned(binned, tname, figs, pick_models(ranked, k=6))
            best = pick_models(ranked, k=1, with_persistence=False)
            if best:
                written[f"curve_{tname}"] = fig_power_curve(preds, best[0], tname, figs, model_color(best[0]))
            if tname != "A":
                fig_sample_week(preds, pick_models(ranked, k=3), tname, cfg["report"]["sample_week_start"], figs)
    if binned_rows:
        pd.concat(binned_rows, ignore_index=True).to_csv(tables / "binned_errors.csv", index=False)
    if boot_rows:
        pd.DataFrame(boot_rows).to_csv(tables / "bootstrap_vs_persistence.csv", index=False)
    fig_skill(main, figs)

    # Slide table: protocol A paper nets + best Chronos fine-tuned (the "attention-based" row).
    slide = []
    for _, r in PAPER_TABLE5.iterrows():
        row = pa[pa["model"] == r["model"]]
        slide.append({"model": r["model"].upper(), "MAE": row["mae_scaled_mean"].mean() if len(row) else np.nan,
                      "MSE": float(mse_s.get(r["model"], np.nan)) if not pa.empty else np.nan,
                      "RMSE": row["rmse_scaled_mean"].mean() if len(row) else np.nan,
                      "R2": row["r2_mean"].mean() if len(row) else np.nan})
    slide_md = "## Paper protocol (A), scaled units\n\n" + to_markdown(pd.DataFrame(slide))
    ft = main[(main["family"] == "chronos2") & ~main["model"].str.contains("zeroshot")]
    if not ft.empty:
        slide_md += "\n## Attention-based model (Chronos-2 fine-tuned), protocol B\n\n" + to_markdown(
            ft[["model", "horizon", "mae_kw_mean", "rmse_kw_mean", "r2_mean", "mae_scaled_mean",
                "rmse_scaled_mean"]].rename(columns=lambda c: c.replace("_mean", "")))
    (tables / "slides_results.md").write_text(slide_md)
    print(f"\nTables -> {tables}\nFigures -> {figs}")
    return written
