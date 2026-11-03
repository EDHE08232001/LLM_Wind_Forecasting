"""Release accelerator memory and find stale processes that still hold the GPU.

GPU memory belongs to the *process* that allocated it; it is returned when that process
exits. So "cleaning GPU memory" means (1) emptying the allocator cache of the current
process (done automatically between stages by the pipeline) and (2) stopping leftover
Python processes (crashed runs, orphaned DataLoader workers, forgotten REPLs).

    python scripts/gpu_cleanup.py            # report: memory + candidate processes
    python scripts/gpu_cleanup.py --kill     # also terminate this project's stale python processes
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import signal
import subprocess
import sys

from _common import ROOT


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return ""


def cuda_processes() -> list[tuple[int, str, str]]:
    if not shutil.which("nvidia-smi"):
        return []
    out = _run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"])
    procs = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3 and parts[0].isdigit():
            procs.append((int(parts[0]), parts[1], parts[2]))
    return procs


def project_python_processes() -> list[tuple[int, str]]:
    """Python processes whose command line mentions this repository (excluding ourselves)."""
    me = os.getpid()
    root = str(ROOT)
    procs = []
    if platform.system() == "Windows":
        out = _run(["powershell", "-NoProfile", "-Command",
                    "Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | "
                    "ForEach-Object { \"$($_.ProcessId)`t$($_.CommandLine)\" }"])
    else:
        out = _run(["ps", "-eo", "pid=,args="])
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        pid_s, _, cmd = line.replace("\t", " ").partition(" ")
        if not pid_s.isdigit() or int(pid_s) == me:
            continue
        if "python" in cmd.lower() and (root in cmd or "windfc" in cmd or "scripts/" in cmd.replace("\\", "/")) \
                and "gpu_cleanup" not in cmd:
            procs.append((int(pid_s), cmd[:160]))
    return procs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kill", action="store_true", help="terminate this project's stale python processes")
    parser.add_argument("--yes", action="store_true", help="do not ask for confirmation with --kill")
    args = parser.parse_args()

    try:
        from windfc.device import free_device_memory, memory_summary

        free_device_memory()
        print(f"[gpu] this process: {memory_summary()}")
    except Exception as exc:  # torch missing etc.
        print(f"[gpu] torch not importable ({exc}); skipping allocator cleanup")

    cuda = cuda_processes()
    if cuda:
        print("[gpu] CUDA compute processes (nvidia-smi):")
        for pid, name, mem in cuda:
            print(f"   pid={pid:<8} {mem:>10}  {name}")
    elif shutil.which("nvidia-smi"):
        print("[gpu] no CUDA compute processes are holding GPU memory")
    if platform.system() == "Darwin":
        print("[gpu] macOS/MPS: unified memory is freed when the owning python process exits.")

    stale = project_python_processes()
    if not stale:
        print("[gpu] no stale project python processes found")
        return
    print("[gpu] project python processes still running:")
    for pid, cmd in stale:
        print(f"   pid={pid:<8} {cmd}")
    if not args.kill:
        print("   (re-run with --kill to terminate them)")
        return
    if not args.yes:
        ans = input("Terminate these processes? [y/N] ").strip().lower()
        if ans not in ("y", "yes"):
            print("aborted")
            return
    for pid, _ in stale:
        try:
            if platform.system() == "Windows":
                subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True)
            else:
                os.kill(pid, signal.SIGTERM)
            print(f"   terminated {pid}")
        except Exception as exc:
            print(f"   could not terminate {pid}: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
