"""Device selection (CUDA > MPS > CPU), seeding and accelerator memory cleanup."""

from __future__ import annotations

import gc
import os
import random

import numpy as np

# Let unsupported MPS ops fall back to CPU instead of crashing (must be set before torch import).
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch  # noqa: E402


def mps_available() -> bool:
    return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())


def get_device(preference: str = "auto") -> torch.device:
    """Return a torch device. ``preference`` is one of auto | cuda | mps | cpu."""
    preference = (preference or "auto").lower()
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if mps_available():
            return torch.device("mps")
        return torch.device("cpu")
    if preference.startswith("cuda") and not torch.cuda.is_available():
        print(f"[device] {preference} requested but CUDA is not available -> cpu")
        return torch.device("cpu")
    if preference == "mps" and not mps_available():
        print("[device] mps requested but not available -> cpu")
        return torch.device("cpu")
    return torch.device(preference)


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def free_device_memory(verbose: bool = False) -> None:
    """Release cached allocator memory on CUDA / MPS for *this* process."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:  # pragma: no cover
            pass
    if mps_available():
        try:
            torch.mps.empty_cache()
        except Exception:  # pragma: no cover
            pass
    if verbose:
        print("[device] " + memory_summary())


def memory_summary() -> str:
    parts = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(i)
            parts.append(
                f"cuda:{i} {torch.cuda.get_device_name(i)} "
                f"allocated={torch.cuda.memory_allocated(i) / 2**20:.0f}MiB "
                f"reserved={torch.cuda.memory_reserved(i) / 2**20:.0f}MiB "
                f"free={free / 2**20:.0f}/{total / 2**20:.0f}MiB"
            )
    if mps_available():
        try:
            parts.append(
                f"mps allocated={torch.mps.current_allocated_memory() / 2**20:.0f}MiB "
                f"driver={torch.mps.driver_allocated_memory() / 2**20:.0f}MiB"
            )
        except Exception:  # pragma: no cover
            parts.append("mps (memory stats unavailable)")
    return "; ".join(parts) if parts else "cpu only"


def describe_environment() -> str:
    lines = [
        f"torch {torch.__version__}",
        f"cuda available: {torch.cuda.is_available()}"
        + (f" ({torch.version.cuda}, {torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""),
        f"mps available: {mps_available()}",
        f"selected device: {get_device('auto')}",
    ]
    return "\n".join(lines)
