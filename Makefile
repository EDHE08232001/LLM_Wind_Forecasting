# Convenience targets (macOS / Linux). Activate the venv first: source venvMac/bin/activate
PY ?= python
CONFIG ?= configs/default.yaml

.PHONY: help setup data prepare baselines paper chronos finetune finetune-qlora report all smoke demo test test-slow lint clean

help:
	@echo "make setup | data | prepare | baselines | paper | chronos | finetune | finetune-qlora | report | all"
	@echo "make smoke (synthetic end-to-end) | demo | test | test-slow | lint | clean"

setup:
	./setup_env.sh

data:
	$(PY) scripts/00_download_data.py --config $(CONFIG)

prepare:
	$(PY) scripts/01_prepare_data.py --config $(CONFIG)

baselines:
	$(PY) scripts/02_run_baselines.py --config $(CONFIG)

paper:
	$(PY) scripts/03_replicate_paper.py --config $(CONFIG)

chronos:
	$(PY) scripts/04_run_chronos.py --config $(CONFIG)

finetune:
	$(PY) scripts/05_finetune_chronos.py --config $(CONFIG) --method lora

finetune-qlora:
	$(PY) scripts/05_finetune_chronos.py --config $(CONFIG) --method qlora

report:
	$(PY) scripts/06_make_report.py --config $(CONFIG)

all:
	$(PY) scripts/run_all.py --config $(CONFIG)

smoke:
	$(PY) scripts/run_all.py --config configs/smoke.yaml

demo:
	$(PY) scripts/demo.py --config $(CONFIG)

test:
	$(PY) -m pytest -m "not slow"

test-slow:
	$(PY) -m pytest -m slow

lint:
	$(PY) -m ruff check .

clean:
	./cleanup.sh
