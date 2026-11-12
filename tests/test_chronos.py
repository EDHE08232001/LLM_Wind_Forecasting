"""Chronos-2 plumbing on a tiny random model (offline), and the real model in the slow test."""

import json

import numpy as np
import pytest

pytest.importorskip("chronos")

from windfc.chronos import (  # noqa: E402
    ChronosForecaster,
    build_inputs,
    finetune_chronos,
    load_finetuned,
    load_pipeline,
    qlora_unavailable_reason,
    variant_columns,
)
from windfc.device import get_device  # noqa: E402
from windfc.splits import ForecastTask  # noqa: E402


def test_build_inputs_never_leak_the_target(tmp_cfg, hourly):
    origins = np.array([500, 900])
    y = hourly["active_power"].to_numpy(dtype=np.float32)
    inp = build_inputs(hourly, origins, "paper_inputs", tmp_cfg, context_length=64, prediction_length=24)
    assert len(inp[0]["target"]) == 64 and inp[0]["target"][-1] == y[500]  # the context ends at the origin
    fut = inp[1]["future_covariates"]
    assert set(fut) == set(tmp_cfg["chronos"]["covariates"]) and "active_power" not in fut
    np.testing.assert_allclose(fut["wind_speed"], hourly["wind_speed"].to_numpy()[901:925], rtol=1e-6)
    past = build_inputs(hourly, origins, "past_covariates", tmp_cfg, 64, 24)
    assert "future_covariates" not in past[0] and len(past[0]["past_covariates"]["wind_speed"]) == 64
    assert set(build_inputs(hourly, origins, "univariate", tmp_cfg, 64, 24)[0]) == {"target"}
    assert "active_power" not in variant_columns("paper_inputs", tmp_cfg)[0]


def test_known_covariates_change_the_forecast(tiny_cfg, hourly):
    """The model must actually receive the target-hour covariates in the paper_inputs variant."""
    pipe = load_pipeline(tiny_cfg["chronos"]["model_id"], "cpu")
    task = ForecastTask(hourly, tiny_cfg, 1)
    a, (lo, hi) = ChronosForecaster(pipe, tiny_cfg, "paper_inputs", verbose=False).predict(task, "test")
    b, _ = ChronosForecaster(pipe, tiny_cfg, "past_covariates", verbose=False).predict(task, "test")
    assert a.shape == (len(task.y("test")),) and np.isfinite(a).all()
    assert (lo <= a + 1e-6).all() and (a <= hi + 1e-6).all()
    assert not np.allclose(a, b)


@pytest.mark.parametrize("variant", ["paper_inputs", "univariate"])
def test_lora_finetune_save_and_reload(tiny_cfg, hourly, tmp_path, variant):
    pipe, info = finetune_chronos(hourly, tiny_cfg, 96, 1e-2, variant=variant, device="cpu", out_dir=tmp_path / "a",
                                  verbose=False)
    assert info["method"] == "lora" and 0 < info["trainable_params"] < 0.2 * info["total_params"]
    assert [h["step"] for h in info["history"]] == [0, 3, 6]
    assert info["best_val_rmse_kw"] <= info["zero_shot_val_rmse_kw"]  # step 0 takes part in the selection
    assert (tmp_path / "a" / "adapter_config.json").exists()  # only the adapter is saved, not the base model
    saved = json.loads((tmp_path / "a" / "adapter_config.json").read_text())
    assert set(saved["target_modules"]) == set(tiny_cfg["finetune"]["lora"]["target_modules"])

    task = ForecastTask(hourly, tiny_cfg, 24)
    before, _ = ChronosForecaster(pipe, tiny_cfg, variant, verbose=False).predict(task, "val")
    reloaded, info2 = load_finetuned(tmp_path / "a", "cpu")
    after, _ = ChronosForecaster(reloaded, tiny_cfg, variant, verbose=False).predict(task, "val")
    assert info2["variant"] == variant
    np.testing.assert_allclose(after, before, rtol=1e-3, atol=1e-2 * task.target_range)


def test_lora_training_changes_only_the_adapter(tiny_cfg, hourly):
    cfg = dict(tiny_cfg)
    cfg["finetune"] = {**tiny_cfg["finetune"], "select_best": False}  # keep the last step
    pipe, info = finetune_chronos(hourly, cfg, 96, 1e-2, variant="past_covariates", device="cpu", verbose=False)
    assert info["best_step"] == info["num_steps"]
    base = load_pipeline(tiny_cfg["chronos"]["model_id"], "cpu").model.state_dict()
    tuned = dict(pipe.model.named_parameters())
    moved = [k for k, v in tuned.items() if "lora_B" in k and float(v.detach().abs().sum()) > 0]
    assert moved, "LoRA B matrices start at zero and must have been updated"
    for name, value in tuned.items():
        key = name.replace("base_model.model.", "").replace(".base_layer", "")
        if "lora_" not in name and key in base:
            assert not value.requires_grad and (value == base[key]).all(), name


def test_qlora_falls_back_without_nvidia(tiny_cfg, hourly, capsys):
    if get_device("auto").type == "cuda" and qlora_unavailable_reason(get_device("auto")) is None:
        pytest.skip("QLoRA is available here; the fallback does not apply")
    assert "QLoRA needs an NVIDIA GPU" in qlora_unavailable_reason(get_device("cpu"))
    _, info = finetune_chronos(hourly, tiny_cfg, 96, 1e-3, method="qlora", variant="univariate", device="cpu",
                               verbose=False)
    assert info["method"] == "lora" and info["requested_method"] == "qlora"
    assert "Falling back to LoRA" in capsys.readouterr().out


@pytest.mark.slow
def test_real_chronos2_zero_shot_and_lora(tmp_cfg, hourly):
    """Downloads amazon/chronos-2 (about 0.5 GB, cached by Hugging Face)."""
    pipe = load_pipeline(tmp_cfg["chronos"]["model_id"], tmp_cfg["device"])
    task = ForecastTask(hourly, tmp_cfg, 24)
    pred, _ = ChronosForecaster(pipe, tmp_cfg, "paper_inputs", context_length=64, verbose=False).predict(task, "val")
    assert pred.shape == (len(task.y("val")),) and np.isfinite(pred).all()
    cfg = dict(tmp_cfg)
    cfg["finetune"] = {**tmp_cfg["finetune"], "num_steps": 2, "eval_every": 2, "val_max_origins": 8, "batch_size": 22}
    _, info = finetune_chronos(hourly, cfg, 64, 1e-4, variant="paper_inputs", device=tmp_cfg["device"], verbose=False)
    assert info["method"] == "lora" and info["total_params"] > 100_000_000
