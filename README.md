# Wind Power Forecasting with Machine Learning (ELG5255, Group 17)

This repo replicates and extends **Karaman (2023), *Prediction of Wind Power with Machine Learning
Models*, Applied Sciences 13, 11455** on the Esenköy (Nordex N117/3600) SCADA + MERRA-2 data:

| Family | Models | Where |
|---|---|---|
| Baselines | persistence, 24 h seasonal naive, linear regression, XGBoost | `src/windfc/models/baselines.py` |
| Paper networks (PyTorch) | ANN, CNN, RNN, LSTM, with the paper's layer sizes | `src/windfc/models/paper_nets.py` |
| Foundation model | **Chronos-2** (`amazon/chronos-2`): zero-shot (univariate / past covariates / known-future covariates) and fine-tuned with **LoRA** or **QLoRA** | `src/windfc/models/chronos.py` |

Everything runs on **Apple MPS**, **NVIDIA CUDA**, or **CPU**. The device is picked automatically
(`cuda > mps > cpu`) or set with `--set device=cpu`.

Everything is plain Python: scripts, a small CLI, and the `src/windfc` package. There are no notebooks.
Scripts write all tables and figures to disk.

---

## 1. Quick start

### macOS (zsh, Apple Silicon) / Linux

```bash
./setup_env.sh --download
```

```bash
source venvMac/bin/activate        # Linux: source venvLinux/bin/activate
```

```bash
python scripts/run_all.py
```

### Windows (PowerShell)

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_env.ps1 -Download
.\venvWin\Scripts\Activate.ps1
python scripts\run_all.py
```

### Windows (cmd.exe)

```bat
setup_env.bat --download
venvWin\Scripts\activate.bat
python scripts\run_all.py
```

### Try it without downloading anything

```bash
python scripts/demo.py --synthetic
```

```bash
python scripts/run_all.py --config configs/smoke.yaml
```

`--synthetic` and the smoke config use **generated fake data** that has the same file formats as the
real data. They check that the code runs end to end (about 10 minutes on a laptop CPU). Do not report
these numbers.

---

## 2. Environment setup scripts

| Script | Platform | Virtual env |
|---|---|---|
| `setup_env.sh` | macOS (zsh/bash) and Linux | `venvMac/` on macOS, `venvLinux/` on Linux |
| `setup_env.ps1` | Windows PowerShell | `venvWin/` |
| `setup_env.bat` | Windows cmd.exe | `venvWin/` |

The setup scripts:

1. Find Python 3.10 to 3.13. 3.11 or 3.12 is recommended; on macOS, install it with `brew install python@3.12`.
2. Create the venv. Pass `--recreate` / `-Recreate` to rebuild it.
3. Install the right **PyTorch** build:
   * macOS: the standard wheel, which includes the **MPS** backend on Apple Silicon.
   * NVIDIA: the script reads the driver's CUDA version from `nvidia-smi` and picks the matching wheel
     index (`cu134`, `cu132`, `cu130`, `cu129`, `cu128`, `cu126` or `cu118`). The wheels bundle the CUDA
     runtime, so you only need an NVIDIA driver, not the standalone CUDA toolkit. RTX 50-series (Blackwell) GPUs need
     `cu128` or newer, so update the driver if the script picks an older index. To force an index, pass
     `--cuda cu134` / `-Cuda cu134` or set the `TORCH_CUDA` environment variable. The manual
     equivalent on Windows is
     `pip install torch --index-url https://download.pytorch.org/whl/cu134`.
   * No GPU, or `--cpu` / `-Cpu`: the CPU wheel.
4. Run `pip install -e ".[chronos,dev]"`. Pass `--no-chronos` to skip Chronos-2. Pass `--qlora` to also
   install `bitsandbytes`, which QLoRA needs (NVIDIA only).
5. Run `scripts/check_env.py`, which prints library versions and does a forward and backward pass on
   the selected device.
6. If you passed `--download` / `-Download`, download the dataset (section 3).

Manual alternative: install torch for your platform first (see `requirements.txt`), then run
`pip install -e ".[chronos,dev]"`.

---

## 3. Dataset

The `data/` folder is git-ignored. You need two raw files in `data/raw/`:

| File | Content | Source |
|---|---|---|
| `T1.csv` | 10-min SCADA for 2018: active power, wind speed, theoretical power, wind direction (50,530 rows) | Kaggle [`berkerisen/wind-turbine-scada-dataset`](https://www.kaggle.com/datasets/berkerisen/wind-turbine-scada-dataset) |
| `weather_merra2_2018.csv` | Hourly MERRA-2 weather at 40.58545 N, 28.99035 E: temperature, precipitation, air density, ground and top-of-atmosphere irradiance, cloud cover | [NASA POWER](https://power.larc.nasa.gov) hourly API (MERRA-2 based, no key needed) |

### Automatic download

```bash
python scripts/00_download_data.py
```

* The SCADA file comes from Kaggle's official `kagglehub` client. Public datasets need no login.
* The weather file comes from the NASA POWER API. Air density is computed from surface pressure,
  temperature and specific humidity. Cloud amount is converted from % to a 0-1 fraction, the same
  units as MERRA-2 `cldtot`.

### Manual download (if the automatic one fails)

1. **SCADA:** open the Kaggle page above, click **Download**, unzip it, and copy `T1.csv` to
   `data/raw/T1.csv`. Kaggle may ask you to sign in. You can also run the `curl` command shown under
   *Download → cURL* on the Kaggle page.
2. **Weather**, either option:
   * Open this URL in a browser and save the result as `data/raw/weather_merra2_2018.csv`:
     `https://power.larc.nasa.gov/api/temporal/hourly/point?parameters=T2M,PRECTOTCORR,PS,QV2M,ALLSKY_SFC_SW_DWN,TOA_SW_DWN,CLOUD_AMT,WS50M,WD50M&community=RE&latitude=40.58545&longitude=28.99035&start=20180101&end=20181231&format=CSV&time-standard=UTC`
   * Or use [renewables.ninja](https://www.renewables.ninja), the source the paper used. Sign in, go to
     *Weather*, set the point (40.58545, 28.99035), the year 2018, dataset MERRA-2, and the variables
     t2m, prectotland, rhoa, swgdn, swtdn and cldtot. Download the CSV and save it as
     `data/raw/weather_merra2_2018.csv`. The loader detects this format automatically. The anonymous
     ninja API no longer serves 2018, so you have to export from the website.

### Data pipeline (`scripts/01_prepare_data.py`)

Output: `data/processed/hourly_clean.parquet` plus `prepare_report.json`.

1. Reindex the SCADA data onto a full 10-min grid. 2,030 rows are missing, the same number the paper
   reports.
2. **kNN imputation** with k = 5, as in the paper. Missing rows have no SCADA values, so neighbours are
   found by weather and time-of-day similarity. The MERRA-2 50 m wind is included, which makes the
   matching informative. The imputer is fitted on **the training months only**, so the test period
   cannot leak into it. Theoretical power for imputed rows is re-derived from the power curve.
3. Keep the on-the-hour sample (`data.resample: first`). The paper "filtered only the hourly data".
   Set `mean` to use hourly averages instead. The result has 8,760 rows.
4. Merge with weather on the UTC timestamp. The SCADA timezone is not documented; set
   `site.scada_utc_offset_hours` to shift it.
5. **Cleaning flags**, applied in the paper's order:
   * wind speed below 3.5 or above 25.5 m/s
   * active power ≤ 0 while the wind is in range
   * IQR outliers (outside Q1 − 1.5·IQR to Q3 + 1.5·IQR)

   Rows are **flagged, not dropped**, so the forecasting series stays regular. Protocol A drops flagged
   rows like the paper does. Protocol B scores all targets and also reports a `clean` subset.

   On the real hourly data the IQR rule removes 0 rows. The paper reports 68, which probably came from
   applying the rule to a different stage of the data. `prepare_report.json` logs every count.

---

## 4. Experimental protocols

| | Protocol A (paper) | Protocol B (ours) |
|---|---|---|
| Task | same-hour regression ("nowcast") | direct forecasting at **h = 1 h** (headline) and **h = 24 h** |
| Split | random 75/25, seed 42, cleaned rows only | chronological: train Jan–Aug, validation Sep–Oct, test Nov–Dec |
| Inputs | same-hour wind speed, direction, theoretical power and weather | only values known at the origin `T − h` (lags, rolling statistics, weather at the origin, calendar features of the target) |
| Nets | 100 epochs, batch 32, Adam 1e-3, MSE (paper) | same settings, early stopping on validation, 5 seeds (mean ± std) |
| Scaling | min-max fitted on the training rows | min-max fitted on the training months |

The protocol A numbers are much higher. The model sees the same-hour wind speed and theoretical power
(the power curve), so it never has to forecast anything. Protocol B is the real forecasting task.

**Paired evaluation.** In protocol B every model, Chronos-2 included, is scored on exactly the same test
targets (every hour of Nov–Dec). The forecast origin for each target is `T − h`. Inputs may come from
earlier months: that is only the past, so it is not a leak.

**Metrics** (`src/windfc/metrics.py`):

* MAE, MSE, RMSE and R² in **kW**
* The same metrics on the **min-max scale**, comparable with paper Table 5. The paper's MAE of 0.0209
  is about 75 kW.
* Persistence-relative skill: `1 − RMSE / RMSE_persistence`
* Errors by wind-speed bin (`[0,5,10,13,26]` m/s, with a focus on speeds above 13 m/s)
* A paired moving-block bootstrap CI of the RMSE difference against persistence

---

## 5. Running the pipeline

Every script takes `--config <yaml>` and any number of `--set key.sub=value` overrides.

```bash
python scripts/00_download_data.py                       # data/raw/
python scripts/01_prepare_data.py                        # data/processed/hourly_clean.parquet
python scripts/02_run_baselines.py                       # persistence / seasonal naive / linear / XGBoost
python scripts/03_replicate_paper.py                     # ANN / CNN / RNN / LSTM, protocols A and B
python scripts/04_run_chronos.py                         # Chronos-2 zero-shot (3 variants)
python scripts/05_finetune_chronos.py                    # Chronos-2 LoRA fine-tuning (grid on Sep-Oct)
python scripts/06_make_report.py                         # tables + figures
```

```bash
python scripts/run_all.py                                # all of the above
python scripts/run_all.py --skip chronos finetune        # without Chronos-2
python scripts/run_all.py --only baselines report
```

You can also use `make all`, `make baselines`, `make paper`, `make chronos`, `make finetune`,
`make report`, `make smoke` and `make test`. On Windows without `make`, use `run_pipeline.ps1` or
`run_pipeline.bat`: they activate `venvWin` and forward arguments to `run_all.py`. `run_pipeline.sh`
does the same on macOS and Linux.

Useful options:

```bash
python scripts/03_replicate_paper.py --protocols A --models lstm            # paper LSTM only
python scripts/03_replicate_paper.py --protocols B --seeds 0 1 2            # fewer seeds
python scripts/04_run_chronos.py --variants univariate past_covariates
python scripts/05_finetune_chronos.py --no-grid --set finetune.num_steps=300
python scripts/run_all.py --set device=cpu --set train.epochs=20
```

### Chronos-2 fine-tuning: LoRA / QLoRA

`scripts/05_finetune_chronos.py` uses its own training loop, so the same code runs on CUDA, MPS and CPU.

* **LoRA** is the default: `--method lora`. Adapters (rank 8, alpha 16) are attached to the time and
  group attention projections (q, k, v, o) and the output patch head. That is about 1.2 M trainable
  parameters out of 120 M.
* **QLoRA**: `--method qlora`. The frozen base weights are loaded in 4-bit NF4 with bitsandbytes and
  LoRA is trained on top. This needs an **NVIDIA GPU** and `pip install bitsandbytes` (or
  `./setup_env.sh --qlora`). On MPS or CPU the script falls back to LoRA and says so.
* **Full fine-tuning**: `--method full`.

The grid covers context length (168, 336, 512 h) × learning rate (1e-4, 5e-4). Each run is scored
every `eval_every` steps by rolling-origin RMSE in kW on the Sep–Oct validation period, and the best
step is restored. Step 0, the zero-shot model, is included in that comparison: if fine-tuning does not
help, you get the zero-shot weights back and the log says so. The best adapter is then evaluated on
Nov–Dec at every horizon.

Adapters are saved to `checkpoints/chronos2/<method>/ctx<C>_lr<LR>/`. Reload one in the demo with
`--chronos-adapter auto`.

The `future_covariates` variant feeds the observed weather of the forecast window as if it were a
forecast. This is **optimistic**, a stand-in for NWP. The variant is flagged in `metrics.csv` (the
`notes` column) and should be reported as an upper bound.

### Demo / inference

```bash
python scripts/demo.py                                   # persistence + XGBoost + LSTM (checkpoint or quick train)
python scripts/demo.py --horizon 24 --chronos            # + Chronos-2 zero-shot
python scripts/demo.py --chronos-adapter auto            # + best fine-tuned adapter
```

The demo writes `results/demo/demo_week_h<H>.png` and a CSV of predictions. It does not change
`metrics.csv`.

---

## 6. Outputs

| Path | Content | In git? |
|---|---|---|
| `results/metrics.csv` | one row per model × protocol × horizon × split × subset × seed, with git commit and config name | yes |
| `results/tables/` | `main_results.{csv,md,tex}`, `protocol_a_vs_b`, `paper_comparison`, `binned_errors`, `bootstrap_vs_persistence`, `finetune_grid_*`, `slides_results.md` | yes |
| `results/figures/` | error by wind bin, power curve (paper Fig. 9 style), sample-week forecasts, skill vs persistence | yes |
| `results/predictions/` | parquet per run (predictions, 10–90 % intervals for Chronos-2) | no |
| `checkpoints/` | paper nets (`.pt`), Chronos-2 adapters | no |

The report keeps the latest row per (model, protocol, horizon, split, subset, seed) and averages over
seeds. Fill the slide's TBD cells from `results/tables/slides_results.md`.

---

## 7. Cleanup

| Script | Platform |
|---|---|
| `cleanup.sh` | macOS / Linux |
| `cleanup.ps1` | Windows PowerShell |
| `cleanup.bat` | Windows cmd.exe |

By default the cleanup scripts only remove safe things:

* `__pycache__`, `.pytest_cache`, `.ruff_cache` and `*.egg-info`
* `.DS_Store`, `._*` and `Thumbs.db`
* the **matplotlib cache** (`~/.matplotlib`, `~/.cache/matplotlib`, `%USERPROFILE%\.matplotlib`)
* smoke and demo outputs, and the synthetic data

They then **free GPU memory**. They empty the CUDA / MPS allocator cache and list leftover project
Python processes that still hold the GPU. GPU memory belongs to a process and is only returned when the
process exits, so to actually free it, run the script with `--gpu-kill` / `-GpuKill` to terminate those
processes (it asks first).

Optional flags:

| Flag | Also deletes |
|---|---|
| `--outputs` | predictions, figures and tables (`metrics.csv` is kept) |
| `--checkpoints` | trained nets and adapters |
| `--data` | derived data (raw downloads are kept) |
| `--hf-cache` | the cached Chronos-2 weights |
| `--pip-cache` | pip's cache |
| `--venv` | the virtual environment |
| `--all` | everything above except `--venv` |

`--dry-run` shows what would be removed without deleting anything. `python scripts/gpu_cleanup.py` can
also be run on its own.

The pipeline also frees accelerator memory between models and stages: `windfc.device.free_device_memory`
calls `torch.cuda.empty_cache()` and `torch.mps.empty_cache()`.

---

## 8. Tests

```bash
pytest -m "not slow"          # fast suite on synthetic data (~1 min, CPU)
```

```bash
pytest -m slow                # Chronos-2 zero-shot + LoRA step (downloads the model once)
```

```bash
ruff check .
```

The tests cover:

* the cleaning rules
* the raw-format loaders (NASA POWER and renewables.ninja)
* split boundaries and leakage (scalers fitted on train, windows ending at the origin, persistence alignment)
* hand-computed metrics
* **parameter counts against the paper's tables**
* end to end through the harness and report
* Chronos-2 input construction (no future target leaks)

CI runs the fast suite (`.github/workflows/ci.yml`).

### Parameter counts

These match the paper's Keras models when there are 10 inputs:

| Net | Paper (Keras) | Here (PyTorch) | Note |
|---|---|---|---|
| ANN | 2,817 | 2,817 | exact |
| CNN | 10,497 | 10,497 | exact |
| RNN | 5,313 | 5,345 | PyTorch adds a second bias vector `b_hh` (+32) |
| LSTM | 31,233 | 31,489 | PyTorch adds `b_hh` (+4·64) |

The paper's tables imply 10 input features, but its text lists 9. The config uses the 9 documented
ones (`features.paper_inputs`); add a column there to try a 10th.

---

## 9. Repository layout

```
configs/            default.yaml (full experiment), smoke.yaml (synthetic, fast)
src/windfc/         config, device, download, data, synthetic, features, splits, metrics,
                    evaluate (harness), train, pipeline (stages), report
  models/           baselines.py, paper_nets.py, chronos.py
scripts/            00..06 pipeline steps, run_all.py, demo.py, check_env.py, gpu_cleanup.py
tests/              pytest suite
results/            metrics.csv, tables/, figures/ (predictions/ ignored)
setup_env.*, cleanup.*, run_pipeline.*   platform scripts
```

## 10. Notes and caveats

* There is one turbine, one year, and two test months, so look at the bootstrap CIs before claiming a
  difference.
* Chronos-2 is about 120 M parameters. 4-bit quantisation is not needed for memory; QLoRA is offered
  for the project's Q-LoRA experiment, so word the slides according to what you actually run.
* The Chronos-2 API is recent. This code was written against `chronos-forecasting` 2.3. If a future
  release breaks it, pin `chronos-forecasting==2.3.*`.

Reference: Karaman, Ö.A. *Prediction of Wind Power with Machine Learning Models.* Appl. Sci. 2023, 13,
11455. https://doi.org/10.3390/app132011455
