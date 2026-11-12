"""End to end on synthetic data with the tiny offline model: references -> zero-shot -> LoRA -> report."""

import pandas as pd
import pytest

pytest.importorskip("chronos")

from windfc.config import resolve_path  # noqa: E402
from windfc.pipeline import stage_evaluate_saved, stage_finetune, stage_references, stage_zeroshot  # noqa: E402
from windfc.report import make_report  # noqa: E402


def test_pipeline_end_to_end(tiny_cfg, hourly):
    cfg = tiny_cfg
    rows = stage_references(cfg)
    pers = [r for r in rows if r["model"] == "persistence"]
    assert pers and all(abs(r["skill_vs_persistence"]) < 1e-12 for r in pers)
    # the synthetic power is the power curve times noise, so the curve must be a strong reference
    curve = next(r for r in rows if r["model"] == "power_curve" and r["split"] == "test" and r["subset"] == "clean")
    assert curve["r2"] > 0.9 and curve["r2_pearson"] >= curve["r2"]

    stage_zeroshot(cfg, ["univariate", "paper_inputs"])
    stage_finetune(cfg, variants=["paper_inputs"])
    res, ckpt = resolve_path(cfg["paths"]["results_dir"]), resolve_path(cfg["paths"]["checkpoints_dir"])
    assert len(list((ckpt / "chronos2" / "lora" / "paper_inputs").glob("*/adapter_model.safetensors"))) == 2
    rescored = stage_evaluate_saved(cfg)

    m = pd.read_csv(res / "metrics.csv")
    assert {"persistence", "power_curve", "chronos2_zeroshot_univariate", "chronos2_zeroshot_paper_inputs",
            "chronos2_lora_paper_inputs"} == set(m["model"])
    assert set(m["horizon"]) == {1, 24} and set(m["split"]) == {"val", "test"} and set(m["subset"]) == {"all", "clean"}
    assert m[["mae_kw", "rmse_kw", "r2", "mae_scaled", "mse_scaled", "rmse_scaled"]].notna().all().all()
    first = m[(m["model"] == "chronos2_lora_paper_inputs") & (m["split"] == "test") & (m["subset"] == "all")]
    for r in [r for r in rescored if r["split"] == "test" and r["subset"] == "all"]:
        run1 = first[first["horizon"] == r["horizon"]].iloc[0]  # the reloaded adapter reproduces the run
        assert r["rmse_kw"] == pytest.approx(run1["rmse_kw"], rel=1e-3)

    make_report(cfg)
    tables, figs = res / "tables", res / "figures"
    comp = pd.read_csv(tables / "paper_comparison.csv")
    assert comp[comp["source"] == "paper"]["model"].tolist() == ["ANN", "CNN", "RNN", "LSTM"]
    assert comp.loc[comp["model"] == "LSTM", "r2"].item() == 0.9574
    slide = (tables / "slides_results.md").read_text()
    assert "LSTM (best in paper)" in slide and "Chronos-2 + LoRA (paper inputs)" in slide
    boot = pd.read_csv(tables / "bootstrap.csv")
    assert "chronos2_zeroshot_paper_inputs" in set(boot["versus"])
    for name in ["paper_comparison.png", "binned_rmse_h1.png", "power_curve_h24.png", "sample_week_h1.png",
                 "finetune_curve_lora_paper_inputs.png"]:
        assert (figs / name).stat().st_size > 5000, name
    assert (tables / "finetune_summary.md").exists() and (tables / "main_results.md").exists()
