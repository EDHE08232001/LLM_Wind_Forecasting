#!/bin/zsh
# =============================================================================
# Environment setup for macOS (zsh). Creates ./venvMac with PyTorch (MPS on
# Apple Silicon), Chronos-2, PEFT (LoRA) and the project itself.
#
#   ./setup_env.zsh                  install everything
#   ./setup_env.zsh --download       ... and download the dataset
#   ./setup_env.zsh --recreate       delete the virtual environment and rebuild it
#   ./setup_env.zsh --python python3.13     use a specific interpreter
#   ./setup_env.zsh --cpu            Linux only: force the CPU-only PyTorch wheel
#   ./setup_env.zsh --cuda cu134     Linux only: force a CUDA wheel index
#
# Afterwards:  source venvMac/bin/activate
# (Also runs on Linux with `bash setup_env.zsh`; the environment is then ./venvLinux.)
# =============================================================================
set -euo pipefail
SCRIPT="$0"
cd "$(dirname "$SCRIPT")"

PYTHON_BIN=""
FORCE_CPU=0
CUDA_TAG="${TORCH_CUDA:-}"
DOWNLOAD=0
RECREATE=0

usage() { sed -n '2,15p' "$SCRIPT"; exit "${1:-0}"; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --cpu) FORCE_CPU=1 ;;
    --cuda) [[ $# -ge 2 ]] || usage 1; CUDA_TAG="$2"; shift ;;
    --download) DOWNLOAD=1 ;;
    --recreate) RECREATE=1 ;;
    --python) [[ $# -ge 2 ]] || usage 1; PYTHON_BIN="$2"; shift ;;
    -h|--help) usage 0 ;;
    *) echo "Unknown option: $1"; usage 1 ;;
  esac
  shift
done

OS="$(uname -s)"
ARCH="$(uname -m)"
case "$OS" in
  Darwin) VENV="venvMac" ;;
  Linux)  VENV="venvLinux" ;;
  *) echo "Unsupported OS: $OS. On Windows use setup_env.ps1 or setup_env.bat."; exit 1 ;;
esac

# ---------------------------------------------------------------- Python 3.10 - 3.14
version_ok() { "$1" -c 'import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] < (3,15) else 1)' 2>/dev/null; }
if [[ -z "$PYTHON_BIN" ]]; then
  for cand in python3.14 python3.13 python3.12 python3.11 python3.10 python3 python; do
    if command -v "$cand" >/dev/null 2>&1 && version_ok "$cand"; then PYTHON_BIN="$cand"; break; fi
  done
fi
if [[ -z "$PYTHON_BIN" ]] || ! version_ok "$PYTHON_BIN"; then
  echo "ERROR: Python 3.10 to 3.14 is required."
  if [[ "$OS" == "Darwin" ]]; then echo "  Install it with:  brew install python@3.13"; fi
  exit 1
fi
echo ">>> Using $("$PYTHON_BIN" --version) ($(command -v "$PYTHON_BIN")) on $OS/$ARCH"

# ---------------------------------------------------------------- virtual environment
if [[ $RECREATE -eq 1 && -d "$VENV" ]]; then
  echo ">>> Removing existing $VENV"
  rm -rf "$VENV"
fi
if [[ ! -x "$VENV/bin/python" ]]; then
  echo ">>> Creating virtual environment ./$VENV"
  "$PYTHON_BIN" -m venv "$VENV"
fi
PY="$VENV/bin/python"
"$PY" -m pip install --upgrade pip setuptools wheel

# ---------------------------------------------------------------- PyTorch
pick_cuda_tag() {
  # Map the newest CUDA version the NVIDIA driver supports (nvidia-smi header) to a PyTorch wheel index.
  local v n
  v="$(nvidia-smi 2>/dev/null | grep -o 'CUDA Version: [0-9]*\.[0-9]*' | awk '{print $3}' | head -1)"
  if [[ -z "$v" ]]; then echo ""; return 0; fi
  n=$(( ${v%%.*} * 10 + ${v##*.} ))
  if   (( n >= 134 )); then echo cu134
  elif (( n >= 132 )); then echo cu132
  elif (( n >= 130 )); then echo cu130
  elif (( n >= 129 )); then echo cu129
  elif (( n >= 128 )); then echo cu128
  elif (( n >= 126 )); then echo cu126
  elif (( n >= 118 )); then echo cu118
  else echo ""; fi
}

WITH_QLORA=0
if [[ "$OS" == "Darwin" ]]; then
  echo ">>> Installing PyTorch for macOS"
  if [[ "$ARCH" == "arm64" ]]; then echo "    Apple Silicon: the MPS (Metal) backend will be used"
  else echo "    Intel Mac: no MPS, the CPU will be used"; fi
  "$PY" -m pip install torch
else
  if [[ $FORCE_CPU -eq 0 && -z "$CUDA_TAG" ]] && command -v nvidia-smi >/dev/null 2>&1; then
    CUDA_TAG="$(pick_cuda_tag)"
  fi
  if [[ $FORCE_CPU -eq 0 && -n "$CUDA_TAG" ]]; then
    echo ">>> NVIDIA GPU -> PyTorch wheel index $CUDA_TAG (+ bitsandbytes for QLoRA)"
    "$PY" -m pip install torch --index-url "https://download.pytorch.org/whl/$CUDA_TAG"
    WITH_QLORA=1
  else
    echo ">>> Installing CPU-only PyTorch"
    "$PY" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
  fi
fi

# ---------------------------------------------------------------- project
echo ">>> Installing the project (editable)"
"$PY" -m pip install -e ".[dev]"
if [[ $WITH_QLORA -eq 1 ]]; then
  echo ">>> Installing bitsandbytes for QLoRA"
  "$PY" -m pip install -e ".[qlora]" || echo "WARNING: bitsandbytes could not be installed - QLoRA will fall back to LoRA."
fi

echo ">>> Environment check"
"$PY" scripts/check_env.py

if [[ $DOWNLOAD -eq 1 ]]; then
  echo ">>> Downloading the dataset"
  "$PY" scripts/00_download_data.py || echo "WARNING: download failed - see README, section 'Dataset', for the manual steps."
fi

cat <<EOF2

Done. Next steps:

    ./run_pipeline.zsh                       # the whole experiment (download, zero-shot, LoRA, report)
    ./run_pipeline.zsh --config configs/smoke.yaml   # quick check on fake data first

Or activate the environment and run the scripts yourself:

    source $VENV/bin/activate
    python scripts/run_all.py
EOF2
