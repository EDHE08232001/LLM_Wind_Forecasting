"""Print library versions and run a tiny forward/backward pass on the selected device (CUDA / MPS / CPU)."""

import importlib
import platform
import sys

from _common import ROOT  # noqa: F401  (sets up sys.path)

from windfc.chronos import qlora_unavailable_reason
from windfc.device import describe_environment, free_device_memory, get_device, memory_summary


def main() -> None:
    print(f"python {sys.version.split()[0]} on {platform.system()} {platform.machine()}")
    for mod in ["numpy", "pandas", "sklearn", "torch", "transformers", "peft", "chronos", "accelerate",
                "bitsandbytes", "matplotlib"]:
        try:
            m = importlib.import_module(mod)
            print(f"  {mod:<13} {getattr(m, '__version__', 'ok')}")
        except Exception as exc:  # noqa: BLE001
            print(f"  {mod:<13} not available ({type(exc).__name__})")
    print(describe_environment())
    import torch

    dev = get_device("auto")
    layer = torch.nn.TransformerEncoderLayer(d_model=16, nhead=2, dim_feedforward=32, batch_first=True).to(dev)
    layer(torch.randn(4, 24, 16, device=dev)).sum().backward()
    print(f"forward/backward on {dev}: OK  ({memory_summary()})")
    free_device_memory()
    reason = qlora_unavailable_reason(dev)
    print("QLoRA (4-bit): " + ("available" if reason is None else f"not available - {reason}. LoRA will be used."))


if __name__ == "__main__":
    main()
