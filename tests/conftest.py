"""Shared fixtures: synthetic raw data run through the real data pipeline, and a tiny Chronos-2.

The tiny model has the real Chronos-2 architecture (same classes, same module names, so LoRA
attaches the same way) but a few thousand random weights, so the fast tests need no download.
Its forecasts are meaningless; the tests only check the plumbing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from windfc.config import load_config  # noqa: E402
from windfc.data import build_hourly_dataset  # noqa: E402
from windfc.synthetic import make_synthetic  # noqa: E402


def make_tiny_chronos2(out_dir: Path) -> Path:
    import torch
    from chronos.chronos2 import Chronos2Model
    from chronos.chronos2.config import Chronos2CoreConfig

    torch.manual_seed(0)
    config = Chronos2CoreConfig(
        d_model=32, d_kv=8, d_ff=64, num_layers=2, num_heads=4, architectures=["Chronos2Model"],
        chronos_config={
            "context_length": 2048, "input_patch_size": 16, "input_patch_stride": 16, "output_patch_size": 16,
            "quantiles": [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99],
            "use_reg_token": True, "use_arcsinh": True, "max_output_patches": 64, "time_encoding_scale": 2048,
        },
    )
    Chronos2Model(config).save_pretrained(str(out_dir))
    return out_dir


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory) -> Path:
    pytest.importorskip("chronos")
    return make_tiny_chronos2(tmp_path_factory.mktemp("tiny_chronos2"))


@pytest.fixture(scope="session")
def tmp_cfg(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("windfc")
    raw = tmp / "raw"
    make_synthetic(raw, seed=1)
    return load_config(
        ROOT / "configs" / "smoke.yaml",
        [
            f"paths.scada_csv={raw / 'T1.csv'}",
            f"paths.weather_csv={raw / 'weather_merra2_2018.csv'}",
            f"paths.processed={tmp / 'processed' / 'hourly_clean.parquet'}",
            f"paths.results_dir={tmp / 'results'}",
            f"paths.checkpoints_dir={tmp / 'checkpoints'}",
            "device=cpu",
        ],
    )


@pytest.fixture(scope="session")
def hourly(tmp_cfg):
    df, _ = build_hourly_dataset(tmp_cfg)
    return df


@pytest.fixture()
def tiny_cfg(tmp_cfg, tiny_model_dir, tmp_path):
    """The smoke config pointed at the tiny offline model, with its own results/checkpoints folders."""
    cfg = load_config(ROOT / "configs" / "smoke.yaml", ["device=cpu"])
    cfg["paths"] = {**tmp_cfg["paths"], "results_dir": str(tmp_path / "results"),
                    "checkpoints_dir": str(tmp_path / "checkpoints")}
    cfg["chronos"] = {**cfg["chronos"], "model_id": str(tiny_model_dir), "context_length": 96, "batch_size": 512}
    cfg["finetune"] = {**cfg["finetune"], "context_lengths": [96], "learning_rates": [1e-3, 1e-2], "num_steps": 6,
                       "eval_every": 3, "batch_size": 44, "val_max_origins": 12}
    return cfg
