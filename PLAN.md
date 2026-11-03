# Wind Power Forecasting: Python Implementation Plan

**Project:** Forecasting Renewable Energy with Machine Learning (Group 17)
**Foundation:** Karaman (2023), *Prediction of Wind Power with Machine Learning Models*, Appl. Sci. 13, 11455
**Pretrained model:** Chronos-2 (`amazon/chronos-2` on Hugging Face)

Everything is plain Python: scripts, a CLI, and a `src/` package. There is no Jupyter or Colab. Scripts
write all figures and tables to disk.

> **Status:** the initial implementation covers every phase below. Each phase lists its implementation
> as *(impl: ...)*. See the README for how to run it.

## 1. Task definition

The paper does same-hour regression (nowcasting) on a random 75/25 split. Our project adds a
chronological split and Chronos-2, which is a forecaster, so every model must be evaluated on the same
task.

* **Target:** active power (kW)
* **Horizons:** 1 h (headline) and 24 h (secondary)
* **Chronological split (2018):** train Jan–Aug, validation Sep–Oct, test Nov–Dec
* **Two evaluation protocols:**
  * **A. Paper protocol:** random 75/25 split, seed 42, same-hour inputs. Used only to replicate the
    paper's numbers.
  * **B. Our protocol:** chronological split, lagged inputs only. Same-hour wind speed and theoretical
    power are not used as inputs for forecasting.
* **Reporting:** all metrics in kW (MAE, RMSE), plus MSE and R². Also report min-max scaled metrics so
  we can compare against paper Table 5.

## 2. Repository layout

See the README, section 9. The layout matches this plan, except that the repo root is used directly
instead of a `wind-forecast/` subfolder. These were added:

* `download.py`, `synthetic.py`, `device.py` and `pipeline.py` in `src/windfc/`
* `demo.py`, `check_env.py`, `gpu_cleanup.py` and `run_all.py` in `scripts/`
* the `setup_env.*`, `cleanup.*` and `run_pipeline.*` platform scripts

## 3. Environment and tooling

* Python 3.14 (3.10+ is fine). Core libraries: numpy, pandas, pyarrow, scikit-learn, xgboost, pyyaml,
  matplotlib.
* PyTorch replaces the paper's TensorFlow, with the same architectures.
* Chronos-2: chronos-forecasting (v2+), transformers, peft (for LoRA), accelerate. bitsandbytes is
  optional, for QLoRA.
* Dev tools: pytest, ruff.
* The device is chosen in code (`cuda > mps > cpu`) and never hard-coded *(impl: `windfc/device.py`)*.
* Environment setup: `setup_env.sh` (creates `venvMac` / `venvLinux`) and `setup_env.ps1` /
  `setup_env.bat` (create `venvWin`).
* Reproducibility: a single `set_seed(seed)` helper. Every metrics row records the config name and the
  git commit.
* Git: `data/` and `results/predictions/` are ignored. `metrics.csv` and the figures are committed so
  teammates can see progress.

## 4. Phases and deliverables

### Phase 1: Data pipeline

*(impl: `data.py`, `scripts/00_download_data.py`, `scripts/01_prepare_data.py`)*

1. Get the Esenköy SCADA csv from Kaggle (10-min, 2018) and MERRA-2 hourly weather from NASA POWER or
   renewables.ninja into `data/raw/`. Both are scripted, and the manual steps are in the README.
2. Parse timestamps, sort, and check for duplicates and gaps.
3. Impute missing values with kNN (k = 5), as the paper does. Fit on the training period only to
   prevent leakage.
4. Resample to hourly (target: 8,760 rows).
5. Cleaning, following the paper:
   * wind speed below 3.5 or above 25.5 m/s
   * zero or negative power above cut-in
   * IQR outliers on active power

   Rows are *flagged*; protocol A drops them. Log how many rows each rule removes.
6. Merge SCADA and weather on timestamp.
7. Write `data/processed/hourly_clean.parquet`.

**Acceptance:** about 8,760 rows, no NaNs, and unit tests on the cleaning rules (`tests/test_data.py`).

### Phase 2: Evaluation harness

*(impl: `splits.py`, `metrics.py`, `evaluate.py`)*

* A chronological split and a random split, seed 42. `ForecastTask` builds supervised samples
  (windows and tabular features). Training targets never cross split boundaries, and every model gets
  the same test targets.
* Metrics: MAE, MSE, RMSE and R² in kW and on the min-max scale; binned errors (bins 0, 5, 10, 13 and
  26 m/s); persistence-relative skill; paired block-bootstrap CI.
* A common model interface, `Forecaster.fit(task)` / `predict(task, split)`. Results are appended to
  `results/metrics.csv`.

### Phase 3: Baselines

*(impl: `models/baselines.py`, `scripts/02_run_baselines.py`)*

* Persistence, and a 24 h seasonal naive. At h = 24 the seasonal naive is identical to persistence.
* Linear regression on lagged power and wind (1, 2, 3, 6, 12 and 24 h before the target at h = 1,
  shifted for longer horizons), the weather at the origin, and calendar features.
* XGBoost on the same features plus rolling mean and std (6 and 24 h), with hour-of-day and
  day-of-year sin/cos. A small grid is searched on validation with early stopping. Direct strategy:
  one model per horizon.

### Phase 4: Paper replication

*(impl: `models/paper_nets.py`, `train.py`, `scripts/03_replicate_paper.py`)*

1. ANN, CNN, RNN and LSTM with the paper's layer sizes. A test checks the parameter counts against the
   paper's tables. ANN and CNN match exactly; RNN and LSTM differ only by PyTorch's second bias vector.
2. Training as in the paper: Adam with lr 0.001, MSE loss, 100 epochs, batch 32, min-max scaling
   fitted on train.
3. Protocol A: random 75/25 split, seed 42.
4. Protocol B: chronological split, lagged inputs, h = 1 and h = 24, early stopping on validation,
   5 seeds (mean ± std).

### Phase 5: Chronos-2

*(impl: `models/chronos.py`, `scripts/04_run_chronos.py`, `scripts/05_finetune_chronos.py`)*

1. Load `amazon/chronos-2` with `Chronos2Pipeline.from_pretrained`. The code targets
   chronos-forecasting 2.3.
2. Zero-shot, univariate.
3. Zero-shot with past covariates.
4. Zero-shot with known-future covariates. These use observed weather, which is optimistic, and the
   variant is flagged as such.
5. Fine-tuning with LoRA (default), QLoRA (4-bit, CUDA only) or full fine-tuning. The grid covers
   context length (168 / 336 / 512 h) × learning rate, and selection is on rolling-origin validation
   RMSE for Sep–Oct. The zero-shot model is included in the selection, so the result is reported
   honestly if fine-tuning does not help.
6. Rolling-origin evaluation over Nov–Dec, with the same targets and origins as every other model.
   Predictions and the 10–90 % intervals are saved.

### Phase 6: Analysis and report

*(impl: `report.py`, `scripts/06_make_report.py`)*

* The main results table (CSV, Markdown and LaTeX)
* Protocol A vs B
* Comparison with paper Table 5
* Error by wind-speed bin, focusing on speeds above 13 m/s
* The power-curve figure
* A sample forecast week
* Skill vs persistence
* Bootstrap CIs
* `slides_results.md`, which fills the slide's TBD cells

## 5. Risks and open decisions

1. **Nowcast vs forecast:** protocol B drops same-hour inputs. The report must explain why the numbers
   drop.
2. **Slide wording:** "replicate the paper's random split for comparability, then re-evaluate
   chronologically."
3. **Q-LoRA vs LoRA:** Chronos-2 has about 120 M parameters, so 4-bit is not needed for memory. Both
   are implemented; keep the slides consistent with what we actually run.
4. **"Attention-based model" row:** implemented as fine-tuned Chronos-2 (`slides_results.md`).
5. **Units:** the paper's MAE of 0.0209 is on the min-max scale, about 75 kW. We always unscale before
   comparing.
6. **Data limits:** one turbine, one year, two test months, so confidence intervals matter.
7. **Chronos-2 API drift:** pin `chronos-forecasting` once results are final.

## 6. Definition of done

* `python scripts/run_all.py` (or `make all`) reproduces every number and figure from the raw data.
* `results/metrics.csv` contains every model × protocol × horizon.
* The tests pass, and the README documents the data download and run commands.
* The slides' TBD cells are filled from `metrics.csv`, not typed by hand.
