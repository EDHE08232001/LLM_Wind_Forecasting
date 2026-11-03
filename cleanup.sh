#!/usr/bin/env bash
# =============================================================================
# Project cleanup (macOS / Linux). Safe by default: only caches and OS junk.
#
#   ./cleanup.sh                 caches (__pycache__, .pytest_cache, .ruff_cache, *.egg-info, build/),
#                                .DS_Store / ._* files, ~/.matplotlib & ~/.cache/matplotlib,
#                                smoke/demo outputs, synthetic data; then frees GPU (CUDA/MPS) memory
#                                and lists stale project python processes holding the GPU
#   ./cleanup.sh --gpu-kill      also terminate stale project python processes (asks first)
#   ./cleanup.sh --outputs       also delete results/predictions, results/figures, results/tables
#   ./cleanup.sh --checkpoints   also delete checkpoints/ (trained nets, LoRA adapters)
#   ./cleanup.sh --data          also delete data/interim and data/processed (raw downloads kept)
#   ./cleanup.sh --hf-cache      also delete the cached Chronos-2 weights (~0.5 GB, re-downloaded on use)
#   ./cleanup.sh --pip-cache     also purge pip's download cache
#   ./cleanup.sh --venv          also delete venvMac / venvLinux
#   ./cleanup.sh --all           everything above except --venv and raw data
#   ./cleanup.sh --dry-run       show what would be removed
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")"

GPU_KILL=0; OUTPUTS=0; CKPT=0; DATA=0; HF=0; PIPC=0; VENVS=0; DRY=0
for a in "$@"; do
  case "$a" in
    --gpu-kill) GPU_KILL=1 ;;
    --outputs) OUTPUTS=1 ;;
    --checkpoints) CKPT=1 ;;
    --data) DATA=1 ;;
    --hf-cache) HF=1 ;;
    --pip-cache) PIPC=1 ;;
    --venv) VENVS=1 ;;
    --all) OUTPUTS=1; CKPT=1; DATA=1; HF=1; PIPC=1 ;;
    --dry-run) DRY=1 ;;
    -h|--help) sed -n '2,19p' "$0"; exit 0 ;;
    *) echo "Unknown option: $a"; exit 1 ;;
  esac
done

rmv() {  # remove paths, report each
  for p in "$@"; do
    [[ -e "$p" || -L "$p" ]] || continue
    if [[ $DRY -eq 1 ]]; then echo "  would remove $p"; else rm -rf -- "$p" && echo "  removed $p"; fi
  done
}
clear_dir_keep_gitkeep() {
  local d="$1"; [[ -d "$d" ]] || return 0
  find "$d" -mindepth 1 ! -name '.gitkeep' -print0 2>/dev/null | while IFS= read -r -d '' f; do rmv "$f"; done
}

echo ">>> Python / tool caches"
find . -path ./venvMac -prune -o -path ./venvLinux -prune -o -path ./.git -prune -o \
  \( -type d \( -name __pycache__ -o -name .pytest_cache -o -name .ruff_cache -o -name .mypy_cache \
     -o -name '*.egg-info' -o -name .ipynb_checkpoints \) -print \) 2>/dev/null | while read -r p; do rmv "$p"; done
rmv build dist .coverage htmlcov

echo ">>> OS junk (.DS_Store, ._*, Thumbs.db)"
find . -path ./.git -prune -o \( -name .DS_Store -o -name '._*' -o -name Thumbs.db -o -name desktop.ini \) -print \
  2>/dev/null | while read -r p; do rmv "$p"; done

echo ">>> matplotlib caches"
rmv "$HOME/.matplotlib" "$HOME/.cache/matplotlib"
[[ -n "${MPLCONFIGDIR:-}" ]] && rmv "$MPLCONFIGDIR"

echo ">>> smoke / demo outputs and synthetic data"
rmv results/smoke results/demo data/synthetic checkpoints/smoke chronos-2-finetuned

if [[ $OUTPUTS -eq 1 ]]; then
  echo ">>> results outputs (metrics.csv kept)"
  for d in results/predictions results/figures results/tables results/runs; do clear_dir_keep_gitkeep "$d"; done
fi
if [[ $CKPT -eq 1 ]]; then echo ">>> checkpoints"; rmv checkpoints; fi
if [[ $DATA -eq 1 ]]; then
  echo ">>> derived data (raw downloads kept)"
  clear_dir_keep_gitkeep data/interim; clear_dir_keep_gitkeep data/processed
fi
if [[ $HF -eq 1 ]]; then
  echo ">>> Hugging Face cache for Chronos-2"
  HF_HUB="${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}"
  rmv "$HF_HUB/models--amazon--chronos-2"
fi
if [[ $PIPC -eq 1 ]]; then
  echo ">>> pip cache"
  if [[ $DRY -eq 1 ]]; then echo "  would run: pip cache purge"; else
    for py in venvMac/bin/python venvLinux/bin/python python3; do
      if command -v "$py" >/dev/null 2>&1 || [[ -x "$py" ]]; then "$py" -m pip cache purge 2>/dev/null && break; fi
    done
  fi
fi

echo ">>> GPU memory (CUDA / MPS)"
PY=""
for cand in venvMac/bin/python venvLinux/bin/python; do [[ -x "$cand" ]] && PY="$cand" && break; done
[[ -z "$PY" ]] && PY="$(command -v python3 || true)"
if [[ -n "$PY" && $DRY -eq 0 ]]; then
  if [[ $GPU_KILL -eq 1 ]]; then "$PY" scripts/gpu_cleanup.py --kill; else "$PY" scripts/gpu_cleanup.py; fi
else
  echo "  (skipped)"
fi

if [[ $VENVS -eq 1 ]]; then echo ">>> virtual environments"; rmv venvMac venvLinux; fi
echo "Cleanup finished."
