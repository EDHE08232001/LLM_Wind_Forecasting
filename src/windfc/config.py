"""YAML configuration loading with dotted command-line overrides.

The config is kept as a nested dict wrapped in :class:`Config`, which allows
both ``cfg["train"]["epochs"]`` and ``cfg.train.epochs`` access.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Config(dict):
    """A dict with attribute access (recursively)."""

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError as exc:  # pragma: no cover - mirrors AttributeError semantics
            raise AttributeError(key) from exc
        return Config(value) if isinstance(value, dict) and not isinstance(value, Config) else value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def to_dict(self) -> dict:
        return copy.deepcopy(dict(self))


def _parse_value(text: str) -> Any:
    """Parse an override value using YAML rules (numbers, bools, lists, null)."""
    return yaml.safe_load(text)


def apply_overrides(cfg: dict, overrides: list[str] | None) -> dict:
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key.sub=value, got {item!r}")
        key, raw = item.split("=", 1)
        node = cfg
        parts = key.strip().split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = _parse_value(raw)
    return cfg


def resolve_path(path: str | Path) -> Path:
    """Resolve a (possibly relative) path against the project root."""
    p = Path(path).expanduser()
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_config(path: str | Path = "configs/default.yaml", overrides: list[str] | None = None) -> Config:
    path = resolve_path(path)
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    base = raw.pop("base", None)
    if base:  # simple inheritance: `base: default.yaml`
        parent = load_config(path.parent / base).to_dict()
        raw = _deep_merge(parent, raw)
    cfg = apply_overrides(raw, overrides)
    cfg["_config_path"] = str(path)
    return Config(cfg)


def _deep_merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def base_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default="configs/default.yaml", help="YAML config file")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a config value, e.g. --set train.epochs=5 (repeatable)",
    )
    return parser
