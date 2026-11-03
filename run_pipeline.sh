#!/usr/bin/env bash
# Activate the project venv and run the full pipeline; arguments go to scripts/run_all.py.
#   ./run_pipeline.sh                                  # everything
#   ./run_pipeline.sh --config configs/smoke.yaml      # synthetic smoke test
#   ./run_pipeline.sh --skip finetune
set -euo pipefail
cd "$(dirname "$0")"
for v in venvMac venvLinux; do
  if [[ -f "$v/bin/activate" ]]; then source "$v/bin/activate"; break; fi
done
command -v python >/dev/null || { echo "No venv found - run ./setup_env.sh first"; exit 1; }
python scripts/run_all.py "$@"
