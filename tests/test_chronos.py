"""Chronos-2 tests. Input construction runs offline; the model tests download amazon/chronos-2
(~0.5 GB, cached in ~/.cache/huggingface) and are marked slow: `pytest -m slow`."""

import numpy as np
import pytest

chronos = pytest.importorskip("chronos")

from windfc.models.chronos import ChronosForecaster, build_inputs, finetune_chronos, load_pipeline  # noqa: E402
from windfc.splits import ForecastTask  # noqa: E402


def test_build_inputs_no_future_leak(tmp_cfg, hourly):
    origins = np.array([500, 900])
    inp = build_inputs(hourly, origins, "future_covariates", tmp_cfg, context_length=64, prediction_length=24)
    y = hourly["active_power"].to_numpy(dtype=np.float32)
    assert len(inp[0]["target"]) == 64 and inp[0]["target"][-1] == y[500]  # context ends at the origin
    fut = inp[1]["future_covariates"]
    assert set(fut) == set(tmp_cfg["chronos"]["future_covariates"])
    assert "active_power" not in fut and "wind_speed" not in fut  # target / wind never "known" in future
    np.testing.assert_allclose(fut["temperature"], hourly["temperature"].to_numpy()[901:925], rtol=1e-6)
    uni = build_inputs(hourly, origins, "univariate", tmp_cfg, 64, 24)
    assert set(uni[0]) == {"target"}


@pytest.mark.slow
def test_zero_shot_and_lora(tmp_cfg, hourly):
    pipe = load_pipeline(tmp_cfg["chronos"]["model_id"], "cpu")
    task = ForecastTask(hourly, tmp_cfg, "B", 24)
    model = ChronosForecaster(pipe, tmp_cfg, "univariate", context_length=64, verbose=False)
    pred = model.predict(task, "test")
    assert pred.shape == (len(task.target_index("test")),) and np.isfinite(pred).all()
    cfg = dict(tmp_cfg)
    cfg["finetune"] = {**tmp_cfg["finetune"], "num_steps": 2, "eval_every": 2, "val_max_origins": 8,
                       "batch_size": 16}
    ft, info = finetune_chronos(hourly, cfg, 64, 1e-4, variant="univariate", device="cpu", verbose=False)
    assert info["method"] == "lora" and info["trainable_params"] > 0
