#!/bin/zsh
# =============================================================================
# Project cleanup for macOS (zsh). With no options it only removes things that
# are safe to lose and are rebuilt automatically.
#
#   ./cleanup.zsh                 Python and tool caches (__pycache__, .pytest_cache, .ruff_cache,
#                                 *.egg-info, build/), .DS_Store and ._* files, the matplotlib
#                                 cache, smoke-test outputs and synthetic data. Then releases
#                                 GPU memory and lists project Python processes still running.
#   ./cleanup.zsh --results       also results/: metrics.csv, tables, figures, predictions
#   ./cleanup.zsh --checkpoints   also checkpoints/ (the trained LoRA / QLoRA adapters)
#   ./cleanup.zsh --data          also data/processed (rebuilt by scripts/01_prepare_data.py)
#   ./cleanup.zsh --raw-data      also data/raw and the kagglehub download cache
#   ./cleanup.zsh --hf-cache      also the cached Chronos-2 weights (~0.5 GB, re-downloaded on use)
#   ./cleanup.zsh --pip-cache     also pip's download cache
#   ./cleanup.zsh --venv          also the virtual environment (venvMac / venvLinux)
#   ./cleanup.zsh --gpu-kill      also stop leftover project Python processes (asks first)
#   ./cleanup.zsh --all           everything above except --venv and --gpu-kill
#   ./cleanup.zsh --dry-run       only show what would be removed
# =============================================================================
set -uo pipefail
SCRIPT="$0"
cd "$(dirname "$SCRIPT")"

GPU_KILL=0; RESULTS=0; CKPT=0; DATA=0; RAW=0; HF=0; PIPC=0; VENVS=0; DRY=0
for a in "$@"; do
  case "$a" in
    --results) RESULTS=1 ;;
    --checkpoints) CKPT=1 ;;
    --data) DATA=1 ;;
    --raw-data) RAW=1 ;;
    --hf-cache) HF=1 ;;
    --pip-cache) PIPC=1 ;;
    --venv) VENVS=1 ;;
    --gpu-kill) GPU_KILL=1 ;;
    --all) RESULTS=1; CKPT=1; DATA=1; RAW=1; HF=1; PIPC=1 ;;
    --dry-run) DRY=1 ;;
    -h|--help) sed -n '2,20p' "$SCRIPT"; exit 0 ;;
    *) echo "Unknown option: $a (try --help)"; exit 1 ;;
  esac
done

rmv() {  # remove each path that exists, and say so
  local p
  for p in "$@"; do
    if [[ -e "$p" || -L "$p" ]]; then
      if [[ $DRY -eq 1 ]]; then echo "  would remove $p"; else rm -rf -- "$p" && echo "  removed $p"; fi
    fi
  done
}
clear_dir() {  # empty a tracked folder but keep its .gitkeep
  local d="$1" f
  [[ -d "$d" ]] || return 0
  find "$d" -mindepth 1 -maxdepth 1 ! -name '.gitkeep' -print 2>/dev/null | while IFS= read -r f; do rmv "$f"; done
}

echo ">>> Python and tool caches"
find . \( -path ./venvMac -o -path ./venvLinux -o -path ./venvWin -o -path ./.git \) -prune -o \
  -type d \( -name __pycache__ -o -name .pytest_cache -o -name .ruff_cache -o -name .mypy_cache \
  -o -name '*.egg-info' -o -name .ipynb_checkpoints \) -print -prune 2>/dev/null |
  while IFS= read -r p; do rmv "$p"; done
rmv build dist .coverage htmlcov

echo ">>> OS junk (.DS_Store, ._*, Thumbs.db)"
find . \( -path ./venvMac -o -path ./venvLinux -o -path ./venvWin -o -path ./.git \) -prune -o \
  -type f \( -name .DS_Store -o -name '._*' -o -name Thumbs.db -o -name desktop.ini \) -print 2>/dev/null |
  while IFS= read -r p; do rmv "$p"; done

echo ">>> matplotlib cache"
rmv "$HOME/.matplotlib" "$HOME/.cache/matplotlib"

echo ">>> Smoke-test outputs and synthetic data"
rmv results/smoke data/synthetic checkpoints/smoke chronos-2-finetuned

if [[ $RESULTS -eq 1 ]]; then
  echo ">>> Results (metrics.csv, tables, figures, predictions)"
  rmv results/metrics.csv results/runs
  for d in results/predictions results/figures results/tables; do clear_dir "$d"; done
fi
if [[ $CKPT -eq 1 ]]; then echo ">>> Checkpoints (LoRA / QLoRA adapters)"; rmv checkpoints; fi
if [[ $DATA -eq 1 ]]; then echo ">>> Processed data"; clear_dir data/processed; fi
if [[ $RAW -eq 1 ]]; then
  echo ">>> Raw downloads"
  clear_dir data/raw
  rmv "${KAGGLEHUB_CACHE:-$HOME/.cache/kagglehub}/datasets/berkerisen/wind-turbine-scada-dataset"
fi
if [[ $HF -eq 1 ]]; then
  echo ">>> Cached Chronos-2 weights"
  rmv "${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}/models--amazon--chronos-2"
fi

PY=""
for cand in venvMac/bin/python venvLinux/bin/python; do
  if [[ -x "$cand" ]]; then PY="$cand"; break; fi
done
if [[ -z "$PY" ]]; then PY="$(command -v python3 || true)"; fi

if [[ $PIPC -eq 1 ]]; then
  echo ">>> pip cache"
  if [[ -z "$PY" ]]; then echo "  (no python found, skipped)"
  elif [[ $DRY -eq 1 ]]; then echo "  would run: $PY -m pip cache purge"
  else "$PY" -m pip cache purge || true; fi
fi

echo ">>> GPU memory"
if [[ -z "$PY" ]]; then echo "  (no python found, skipped)"
elif [[ $DRY -eq 1 ]]; then echo "  would run: $PY scripts/gpu_cleanup.py"
elif [[ $GPU_KILL -eq 1 ]]; then "$PY" scripts/gpu_cleanup.py --kill || true
else "$PY" scripts/gpu_cleanup.py || true; fi

if [[ $VENVS -eq 1 ]]; then echo ">>> Virtual environments"; rmv venvMac venvLinux; fi
echo "Cleanup finished."
