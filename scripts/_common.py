"""Shared bootstrap for the scripts: make `src/` importable without `pip install -e .`."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from windfc.config import base_arg_parser, load_config  # noqa: E402

__all__ = ["ROOT", "base_arg_parser", "load_config"]
