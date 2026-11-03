"""Print library versions and run a tiny forward/backward pass on the selected device (CUDA / MPS / CPU)."""

import importlib
import platform
import sys

from _common import ROOT  # noqa: F401  (sets up sys.path)

from windfc.device import describe_environment, free_device_memory, get_device, memory_summary


def main() -> None:
    print(f"python {sys.version.split()[0]} on {platform.system()} {platform.machine()}")
    for mod in ["numpy", "pandas", "sklearn", "xgboost", "torch", "transformers", "peft", "chronos",
                "accelerate", "bitsandbytes", "matplotlib"]:
        try:
            m = importlib.import_module(mod)
            print(f"  {mod:<13} {getattr(m, '__version__', 'ok')}")
        except Exception as exc:  # noqa: BLE001
            print(f"  {mod:<13} not available ({type(exc).__name__})")
    print(describe_environment())
    import torch

    dev = get_device("auto")
    model = torch.nn.LSTM(8, 16, batch_first=True).to(dev)
    x = torch.randn(4, 24, 8, device=dev)
    out, _ = model(x)
    out.sum().backward()
    print(f"forward/backward on {dev}: OK  ({memory_summary()})")
    free_device_memory()


if __name__ == "__main__":
    main()
