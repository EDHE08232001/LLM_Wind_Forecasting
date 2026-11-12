#!/bin/zsh
# Run the whole pipeline with the project's virtual environment. Arguments go to scripts/run_all.py.
#   ./run_pipeline.zsh                                  everything (download, zero-shot, LoRA, report)
#   ./run_pipeline.zsh --config configs/smoke.yaml      quick end-to-end check on synthetic data
#   ./run_pipeline.zsh --skip download                  the data is already in data/raw
#   ./run_pipeline.zsh --only report                    rebuild the tables and figures
#   ./run_pipeline.zsh --method qlora                   QLoRA instead of LoRA (NVIDIA GPU only)
set -euo pipefail
cd "$(dirname "$0")"
PY=""
for v in venvMac venvLinux; do
  if [[ -x "$v/bin/python" ]]; then PY="$v/bin/python"; break; fi
done
if [[ -z "$PY" ]]; then echo "No virtual environment found - run ./setup_env.zsh first"; exit 1; fi
exec "$PY" scripts/run_all.py "$@"
