"""Small helpers: git hash, results paths, CSV appends, logging."""

from __future__ import annotations

import datetime as _dt
import json
import subprocess
from pathlib import Path

import pandas as pd

from .config import PROJECT_ROOT, Config, resolve_path


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
        sha = out.stdout.strip() or "nogit"
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "nogit"


def now() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def results_dir(cfg: Config) -> Path:
    d = resolve_path(cfg["paths"]["results_dir"])
    (d / "predictions").mkdir(parents=True, exist_ok=True)
    (d / "figures").mkdir(parents=True, exist_ok=True)
    (d / "tables").mkdir(parents=True, exist_ok=True)
    return d


def checkpoints_dir(cfg: Config) -> Path:
    d = resolve_path(cfg["paths"]["checkpoints_dir"])
    d.mkdir(parents=True, exist_ok=True)
    return d


def append_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path)
        new = pd.concat([old, new], ignore_index=True, sort=False)
    new.to_csv(path, index=False)


def save_run_config(cfg: Config, out_dir: Path, tag: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"config_{tag}.json"
    payload = dict(cfg.items())
    payload["_git_commit"] = git_commit()
    payload["_saved_at"] = now()
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


def banner(text: str) -> None:
    line = "=" * max(60, len(text) + 4)
    print(f"\n{line}\n  {text}\n{line}", flush=True)
