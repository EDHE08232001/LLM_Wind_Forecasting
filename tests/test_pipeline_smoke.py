"""End-to-end through the harness on synthetic data (CPU, a few seconds)."""

import pandas as pd

from windfc.config import resolve_path
from windfc.evaluate import run_forecaster
from windfc.models.baselines import LinearModel, Persistence, XGBoostModel
from windfc.report import make_report
from windfc.splits import ForecastTask
from windfc.train import NetForecaster, load_net_checkpoint


def test_harness_end_to_end(tmp_cfg, hourly):
    task = ForecastTask(hourly, tmp_cfg, "B", 1)
    rows = run_forecaster(Persistence(), task, tmp_cfg)
    assert {r["split"] for r in rows} == {"val", "test"}
    assert all(abs(r["skill_vs_persistence"]) < 1e-12 for r in rows)
    run_forecaster(LinearModel(), task, tmp_cfg)
    run_forecaster(XGBoostModel(tmp_cfg["baselines"]["xgboost"]), task, tmp_cfg)
    ckpt_dir = resolve_path(tmp_cfg["paths"]["checkpoints_dir"])
    net = NetForecaster("cnn", {**tmp_cfg["train"], "epochs": 2}, device="cpu", seed=0, checkpoint_dir=ckpt_dir,
                        verbose=False)
    run_forecaster(net, task, tmp_cfg, seed=0)
    model, meta = load_net_checkpoint(ckpt_dir / "B_h1__cnn__seed0.pt", "cpu")
    assert meta["horizon"] == 1

    task_a = ForecastTask(hourly, tmp_cfg, "A")
    run_forecaster(LinearModel(), task_a, tmp_cfg)

    res = resolve_path(tmp_cfg["paths"]["results_dir"])
    metrics = pd.read_csv(res / "metrics.csv")
    assert {"persistence", "linear", "xgboost", "cnn"} <= set(metrics["model"])
    assert metrics["mae_kw"].notna().all()
    out = make_report(tmp_cfg)
    assert (res / "tables" / "main_results.md").exists() and out["main"].exists()
    assert any((res / "figures").glob("*.png"))
