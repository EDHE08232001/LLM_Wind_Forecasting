#!/usr/bin/env bash
# =============================================================================
# Environment setup for macOS (Apple Silicon / MPS) and Linux (NVIDIA CUDA or CPU).
#   macOS -> ./venvMac      Linux -> ./venvLinux
#
# Usage (works from zsh or bash):
#   ./setup_env.sh                  # auto-detect device, install everything
#   ./setup_env.sh --download       # ... and download the dataset
#   ./setup_env.sh --cpu            # force the CPU-only PyTorch wheel
#   ./setup_env.sh --cuda cu134     # force a specific CUDA wheel (Linux)
#   ./setup_env.sh --no-chronos     # skip Chronos-2 / transformers / peft
#   ./setup_env.sh --qlora          # also install bitsandbytes (NVIDIA only)
#   ./setup_env.sh --python python3.12 --recreate
# Afterwards:  source venvMac/bin/activate   (or venvLinux)
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

PYTHON_BIN=""
FORCE_CPU=0
CUDA_TAG="${TORCH_CUDA:-}"
WITH_CHRONOS=1
WITH_QLORA=0
DOWNLOAD=0
RECREATE=0

usage() { sed -n '2,16p' "$0"; exit 0; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --cpu) FORCE_CPU=1 ;;
    --cuda) CUDA_TAG="$2"; shift ;;
    --no-chronos) WITH_CHRONOS=0 ;;
    --qlora) WITH_QLORA=1 ;;
    --download) DOWNLOAD=1 ;;
    --recreate) RECREATE=1 ;;
    --python) PYTHON_BIN="$2"; shift ;;
    -h|--help) usage ;;
    *) echo "Unknown option: $1"; usage ;;
  esac
  shift
done

OS="$(uname -s)"
ARCH="$(uname -m)"
case "$OS" in
  Darwin) VENV="venvMac" ;;
  Linux)  VENV="venvLinux" ;;
  MINGW*|MSYS*|CYGWIN*) VENV="venvWin" ;;
  *) echo "Unsupported OS: $OS"; exit 1 ;;
esac

# ---------------------------------------------------------------- python >= 3.10
version_ok() { "$1" -c 'import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] < (3,14) else 1)' 2>/dev/null; }
if [[ -z "$PYTHON_BIN" ]]; then
  for cand in python3.12 python3.11 python3.13 python3.10 python3 python; do
    if command -v "$cand" >/dev/null 2>&1 && version_ok "$cand"; then PYTHON_BIN="$cand"; break; fi
  done
fi
if [[ -z "$PYTHON_BIN" ]] || ! version_ok "$PYTHON_BIN"; then
  echo "ERROR: need Python 3.10-3.13 (3.11/3.12 recommended)."
  [[ "$OS" == "Darwin" ]] && echo "  Install with:  brew install python@3.12"
  exit 1
fi
echo ">>> Using $($PYTHON_BIN --version) ($(command -v "$PYTHON_BIN")) on $OS/$ARCH"

# ---------------------------------------------------------------- venv
if [[ $RECREATE -eq 1 && -d "$VENV" ]]; then
  echo ">>> Removing existing $VENV"; rm -rf "$VENV"
fi
if [[ ! -d "$VENV" ]]; then
  echo ">>> Creating virtual environment ./$VENV"
  "$PYTHON_BIN" -m venv "$VENV"
fi
# shellcheck disable=SC1090
source "$VENV/bin/activate"
python -m pip install --upgrade pip setuptools wheel

# ---------------------------------------------------------------- torch
pick_cuda_tag() {
  # Map the driver's max supported CUDA version (nvidia-smi header) to a PyTorch wheel index.
  local v major minor
  v="$(nvidia-smi 2>/dev/null | grep -o 'CUDA Version: [0-9]*\.[0-9]*' | awk '{print $3}' | head -1)"
  [[ -z "$v" ]] && { echo ""; return; }
  major="${v%%.*}"; minor="${v##*.}"
  local n=$((major * 10 + minor))
  if   (( n >= 134 )); then echo cu134
  elif (( n >= 132 )); then echo cu132
  elif (( n >= 130 )); then echo cu130
  elif (( n >= 129 )); then echo cu129
  elif (( n >= 128 )); then echo cu128
  elif (( n >= 126 )); then echo cu126
  elif (( n >= 118 )); then echo cu118
  else echo ""; fi
}

if [[ "$OS" == "Darwin" ]]; then
  echo ">>> Installing PyTorch for macOS (MPS backend on Apple Silicon)"
  [[ "$ARCH" != "arm64" ]] && echo "    (Intel Mac: no MPS, CPU will be used)"
  pip install torch
elif [[ $FORCE_CPU -eq 1 ]]; then
  echo ">>> Installing CPU-only PyTorch"
  pip install torch --index-url https://download.pytorch.org/whl/cpu
else
  if [[ -z "$CUDA_TAG" ]] && command -v nvidia-smi >/dev/null 2>&1; then CUDA_TAG="$(pick_cuda_tag)"; fi
  if [[ -n "$CUDA_TAG" ]]; then
    echo ">>> NVIDIA GPU detected -> PyTorch wheel index $CUDA_TAG"
    pip install torch --index-url "https://download.pytorch.org/whl/$CUDA_TAG"
  else
    echo ">>> No NVIDIA GPU detected -> CPU-only PyTorch"
    pip install torch --index-url https://download.pytorch.org/whl/cpu
  fi
fi

# ---------------------------------------------------------------- project
EXTRAS="dev"
[[ $WITH_CHRONOS -eq 1 ]] && EXTRAS="chronos,$EXTRAS"
if [[ $WITH_QLORA -eq 1 ]]; then
  if [[ "$OS" == "Darwin" ]]; then echo "    (QLoRA needs NVIDIA; skipping bitsandbytes on macOS)"; else EXTRAS="$EXTRAS,qlora"; fi
fi
echo ">>> Installing project (editable) with extras [$EXTRAS]"
pip install -e ".[$EXTRAS]"

echo ">>> Environment check"
python scripts/check_env.py

if [[ $DOWNLOAD -eq 1 ]]; then
  echo ">>> Downloading dataset"
  python scripts/00_download_data.py || echo "!!! Download failed - see README 'Dataset' for manual steps."
fi

cat <<EOF

Done. Activate the environment with:

    source $VENV/bin/activate

Then e.g.:
    python scripts/00_download_data.py      # dataset (if not done with --download)
    python scripts/run_all.py               # full pipeline
    python scripts/demo.py --synthetic      # quick demo without downloads
EOF
