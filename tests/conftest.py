"""Shared fixtures: a synthetic raw dataset prepared through the real pipeline (once per session)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from windfc.config import load_config  # noqa: E402
from windfc.data import build_hourly_dataset  # noqa: E402
from windfc.synthetic import make_synthetic  # noqa: E402


@pytest.fixture(scope="session")
def tmp_cfg(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("windfc")
    raw = tmp / "raw"
    make_synthetic(raw, seed=1)
    cfg = load_config(
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
    return cfg


@pytest.fixture(scope="session")
def hourly(tmp_cfg):
    df, report = build_hourly_dataset(tmp_cfg)
    return df
