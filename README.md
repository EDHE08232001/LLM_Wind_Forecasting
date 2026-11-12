# Wind Power Forecasting with Chronos-2 + LoRA / QLoRA (ELG5255, Group 17)

This project fine-tunes **Chronos-2** ([`amazon/chronos-2`](https://huggingface.co/amazon/chronos-2), a
120M-parameter pretrained time-series transformer) with **LoRA** or **QLoRA** to predict the active power
of a wind turbine, and compares the result with the paper that inspired it:

> Karaman, Ö.A. *Prediction of Wind Power with Machine Learning Models.* Appl. Sci. 2023, 13, 11455.
> https://doi.org/10.3390/app132011455

The paper's ANN, CNN, RNN and LSTM are **not re-implemented**. Their published test results (Table 5) are
quoted in `src/windfc/paper.py` and placed next to ours in the report.

What the code does, in order:

1. Downloads the same data as the paper: the Esenköy SCADA file (Kaggle) and MERRA-2 weather (NASA POWER).
2. Prepares it the way the paper did (kNN imputation, hourly samples, cleaning rules).
3. Scores Chronos-2 **zero-shot** on a chronological split (train Jan–Aug, validation Sep–Oct, test Nov–Dec).
4. **Fine-tunes** Chronos-2 with LoRA or QLoRA and scores it on the same test months.
5. Writes tables and figures, including the comparison with the paper and the filled-in slide table.

It runs on Apple Silicon (MPS), NVIDIA GPUs (CUDA) and CPU. The device is picked automatically.

---

## 1. Quick start

You need **Python 3.10–3.14** and **git**. The setup script creates a virtual environment inside the repo
and installs everything, including the right PyTorch build for your machine.

### macOS (zsh)

```bash
./setup_env.zsh --download
```

```bash
./run_pipeline.zsh
```

If Python is missing: `brew install python@3.13`.

### Windows 11 (PowerShell)

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_env.ps1 -Download
```

```powershell
powershell -ExecutionPolicy Bypass -File .\run_pipeline.ps1
```

### Windows 11 (cmd.exe)

```bat
setup_env.bat --download
```

```bat
run_pipeline.bat
```

If Python is missing on Windows: `winget install Python.Python.3.13`, then open a new terminal.

### Check that everything works first (recommended)

The smoke configuration runs the whole pipeline on generated **fake** data with 20 training steps. It still
downloads the real Chronos-2 weights (about 0.5 GB, once). Its numbers mean nothing; it only proves the
setup works.

```bash
./run_pipeline.zsh --config configs/smoke.yaml
```

On Windows: `.\run_pipeline.ps1 --config configs/smoke.yaml` or `run_pipeline.bat --config configs/smoke.yaml`.

### Where the results land

| Path | Content |
|---|---|
| `results/tables/slides_results.md` | the slide's results table with the "attention-based model" row filled in |
| `results/tables/paper_comparison.md` | the paper's Table 5 next to every Chronos-2 variant, in the paper's units |
| `results/tables/main_results.md` | all models, both horizons, in kW |
| `results/tables/finetune_summary.md` | which LoRA / QLoRA run was selected and how much it gained on validation |
| `results/figures/` | paper comparison, error by wind speed, power curve, a sample week, training curves |
| `results/metrics.csv` | every score ever logged (the report uses the latest row per model) |
| `checkpoints/chronos2/<method>/<variant>/` | the trained adapters (a few MB each) |

---

## 2. Setup scripts

| Script | Platform | Virtual environment |
|---|---|---|
| `setup_env.zsh` | macOS (zsh) | `venvMac/` |
| `setup_env.ps1` | Windows 11, PowerShell | `venvWin/` |
| `setup_env.bat` | Windows 11, cmd.exe | `venvWin/` |

Each script:

1. Finds Python 3.10–3.14, newest first.
2. Creates the virtual environment.
3. Installs **PyTorch** for the machine:
   * macOS: the standard wheel, which uses the **MPS** backend on Apple Silicon.
   * Windows with an NVIDIA GPU: reads the driver's CUDA version from `nvidia-smi` and picks the matching
     wheel index (`cu134`, `cu132`, `cu130`, `cu129`, `cu128`, `cu126` or `cu118`). The wheels include the
     CUDA runtime, so only the NVIDIA driver is needed. RTX 50-series cards need `cu128` or newer.
   * No NVIDIA GPU: the CPU wheel.
4. Installs the project and its dependencies (`chronos-forecasting`, `transformers`, `peft`, ...).
5. On NVIDIA machines, installs **bitsandbytes**, which QLoRA needs. If that fails, setup continues and
   QLoRA falls back to LoRA.
6. Runs `scripts/check_env.py`, which prints the library versions, the selected device and whether QLoRA
   is available.

Options:

| zsh / cmd | PowerShell | Effect |
|---|---|---|
| `--download` | `-Download` | also download the dataset |
| `--recreate` | `-Recreate` | delete the virtual environment and rebuild it |
| `--cpu` | `-Cpu` | force the CPU-only PyTorch wheel (not needed on macOS) |
| `--cuda cu134` | `-Cuda cu134` | force a CUDA wheel index (Windows / Linux) |
| `--python python3.13` | `-Python "py -3.13"` | use a specific interpreter (zsh and PowerShell only) |

Linux is not a target platform, but `bash setup_env.zsh` works there and creates `venvLinux/`.

Manual setup, if you prefer: create and activate a virtual environment, install PyTorch for your platform
from [pytorch.org](https://pytorch.org/get-started/locally/), then run `pip install -e ".[dev]"` (add
`,qlora` inside the brackets on an NVIDIA machine).

---

## 3. Dataset

`data/` is git-ignored. Two raw files are needed in `data/raw/`:

| File | Content | Source |
|---|---|---|
| `T1.csv` | 10-minute SCADA for 2018: active power, wind speed, theoretical power, wind direction (50,530 rows) | Kaggle [`berkerisen/wind-turbine-scada-dataset`](https://www.kaggle.com/datasets/berkerisen/wind-turbine-scada-dataset) |
| `weather_merra2_2018.csv` | hourly MERRA-2 weather at 40.58545 N, 28.99035 E: temperature, precipitation, air density, ground and top-of-atmosphere irradiance, cloud cover | [NASA POWER](https://power.larc.nasa.gov) hourly API (no key needed) |

### Automatic download

`--download` in the setup script does this, and so does the first stage of `run_pipeline`. To run it alone
(with the virtual environment active):

```bash
python scripts/00_download_data.py
```

The SCADA file comes through Kaggle's `kagglehub` client, which needs no login for public datasets.

### Manual download, if the automatic one fails

1. **SCADA:** open the Kaggle page above, click **Download**, unzip, and copy `T1.csv` to `data/raw/T1.csv`.
2. **Weather:** open this URL in a browser and save the result as `data/raw/weather_merra2_2018.csv`:

   ```
   https://power.larc.nasa.gov/api/temporal/hourly/point?parameters=T2M,PRECTOTCORR,PS,QV2M,ALLSKY_SFC_SW_DWN,TOA_SW_DWN,CLOUD_AMT,WS50M,WD50M&community=RE&latitude=40.58545&longitude=28.99035&start=20180101&end=20181231&format=CSV&time-standard=UTC
   ```

   A [renewables.ninja](https://www.renewables.ninja) weather export (the paper's source) for the same
   point and year also works; the loader recognises both formats.

Then run the pipeline with `--skip download`.

### How the data is prepared (`scripts/01_prepare_data.py`)

Output: `data/processed/hourly_clean.parquet` and `prepare_report.json` (row counts for every step).

1. The SCADA data is put on a full 10-minute grid. 2,030 rows are missing, the number the paper reports.
2. Missing rows are filled by **kNN imputation with k = 5**, as in the paper. Neighbours are found by
   weather and time of day, and only among the training months, so the test months cannot leak in.
3. The on-the-hour sample of each hour is kept, giving 8,760 rows (`data.resample: mean` averages instead).
4. Weather is merged on the timestamp.
5. The paper's **cleaning rules** are applied as flags: wind speed outside 3.5–25.5 m/s, zero or negative
   power while the wind is in range, and IQR outliers. Rows are flagged, not deleted, because Chronos-2
   needs an unbroken hourly series. Every score is reported twice: on `all` hours and on the `clean`
   hours the paper would have kept.

---

## 4. What is evaluated, and how it compares with the paper

### Split

Chronological: **train Jan–Aug, validation Sep–Oct, test Nov–Dec 2018**. The paper used a random 75/25
split, where test hours sit between training hours. Ours is harder and matches how a model would be used.

### Three ways of feeding Chronos-2

| Variant | What Chronos-2 sees | Kind of task |
|---|---|---|
| `univariate` | past active power | forecast |
| `past_covariates` | past power, wind speed and direction, theoretical power, weather | forecast |
| `paper_inputs` | all of the above **plus wind speed, direction, theoretical power and weather at the target hours** | estimation, like the paper |

The paper's networks estimate the power of an hour from **that same hour's** wind speed, theoretical power
and weather. `paper_inputs` gives Chronos-2 the same information (as "known future covariates"), so it is
the variant to put next to the paper's Table 5. The other two are real forecasts, where nothing about the
target hour is known, and they answer a harder question than the paper asked.

Each variant is scored 1 hour and 24 hours ahead with a rolling origin: for every test hour `T`, the model
is given the history up to `T − h` and its `h`-th predicted step is compared with the truth.

### Two references that need no training

* `persistence`: the power at the forecast origin.
* `power_curve`: the manufacturer's theoretical power at the target hour's wind speed. It uses the paper's
  information with no learning at all, so it shows how much of a high R² comes from the inputs alone.

### Metrics

MAE, MSE, RMSE and R² in **kW**, and the same errors on the paper's **min-max scale** (kW error divided by
the training range of active power, about 3,600 kW). The paper's best MAE of 0.0209 is roughly 75 kW.
Also reported: skill against persistence, the coverage of the 10–90 % prediction interval, errors by
wind-speed bin (the paper found its models weakest above 13 m/s), and block-bootstrap confidence intervals
for fine-tuned vs zero-shot.

`r2` is the usual 1 − SS_res/SS_tot. `r2_pearson` is the squared correlation, which is how the paper's
Eq. (2) defines R². It ignores bias, so it is never lower.

### Reading the comparison fairly

The slide table uses `paper_inputs`, 1 hour ahead, `clean` hours (`report.headline` in the config). Three
differences from the paper remain and should be stated with the result:

* The split is chronological, not random.
* Chronos-2 also sees the recent power history, which the paper's networks did not. The 24-hour rows in
  `paper_comparison.md` reduce that advantage, since day-old power says little about the current hour.
* One turbine, one year, two test months: check `tables/bootstrap.csv` before claiming a difference.

---

## 5. Running the pieces yourself

Activate the environment first: `source venvMac/bin/activate` (macOS), `.\venvWin\Scripts\Activate.ps1`
(PowerShell) or `venvWin\Scripts\activate.bat` (cmd). On Windows, write the paths with backslashes.

```bash
python scripts/00_download_data.py      # data/raw/
python scripts/01_prepare_data.py       # data/processed/hourly_clean.parquet
python scripts/02_zero_shot.py          # references + Chronos-2 zero-shot
python scripts/03_finetune.py           # LoRA fine-tuning + test scores
python scripts/04_make_report.py        # tables and figures
```

`python scripts/run_all.py` runs all five. `run_pipeline.*` is the same thing without needing to activate
the environment; both take the same arguments:

```bash
python scripts/run_all.py --skip download            # data already in data/raw
python scripts/run_all.py --only finetune report     # just these stages
python scripts/run_all.py --method qlora             # QLoRA instead of LoRA
python scripts/run_all.py --no-grid                  # one training run per variant (much faster)
```

Every script takes `--config <yaml>` and any number of `--set key.sub=value` overrides, for example
`--set device=cpu` or `--set finetune.num_steps=300`. All settings live in `configs/default.yaml`.

### Fine-tuning options

```bash
python scripts/03_finetune.py                              # LoRA, full grid, both variants
python scripts/03_finetune.py --method qlora               # QLoRA
python scripts/03_finetune.py --variants paper_inputs      # only the variant compared with the paper
python scripts/03_finetune.py --no-grid --set finetune.num_steps=300
python scripts/03_finetune.py --eval-only                  # re-score the saved adapters, no training
```

How it works (`src/windfc/chronos.py`):

* **LoRA**: the 120M base weights are frozen. Rank-8 adapters (alpha 16, dropout 0.05) are added to the
  attention projections (`q`, `k`, `v`, `o` of the time and group attention) and to the output head, the
  same modules the Chronos-2 library uses for LoRA. Only the adapters are trained and saved.
* **QLoRA**: the same adapters, but the frozen base weights are loaded in **4-bit NF4** with bitsandbytes.
  This needs an **NVIDIA GPU**; bitsandbytes has no MPS or CPU kernels. On a Mac, `--method qlora` prints
  why and runs LoRA instead, and the results are labelled `lora`.
* **Grid**: context length (168 and 512 hours) × learning rate (1e-5 and 1e-4), 1,000 steps each, for
  each of the two variants in `finetune.variants`. That is 8 training runs by default.
* **Model selection**: every 100 steps the model is scored by rolling-origin RMSE in kW on the validation
  months, and the best step is kept. Step 0, the untouched zero-shot model, is part of that comparison,
  so fine-tuning cannot make the selected model worse on validation. If it did not help, the log says so.
* The best run per variant is then scored on Nov–Dec at both horizons.

Runtime depends heavily on hardware and has not been measured on your machines. An NVIDIA GPU will be
much faster than an Apple Silicon laptop; on the Mac, start with `--no-grid` or
`--variants paper_inputs`, and lower `finetune.num_steps` if needed.

---

## 6. Cleanup

| Script | Platform |
|---|---|
| `cleanup.zsh` | macOS |
| `cleanup.ps1` | Windows 11, PowerShell |
| `cleanup.bat` | Windows 11, cmd.exe |

With no options the scripts remove only what is safe to lose:

* `__pycache__`, `.pytest_cache`, `.ruff_cache`, `*.egg-info`, `build/`
* `.DS_Store`, `._*`, `Thumbs.db`, `desktop.ini`
* the matplotlib cache
* smoke-test outputs and the synthetic data

They then release GPU memory held by the cleanup process itself and list any project Python processes
that are still running. GPU memory belongs to a process and comes back only when that process ends, so a
crashed or forgotten run is what actually keeps it busy; `--gpu-kill` stops those (it asks first).

Options (PowerShell uses `-Results`, `-Checkpoints`, `-Data`, `-RawData`, `-HfCache`, `-PipCache`, `-Venv`,
`-GpuKill`, `-All`, `-DryRun`):

| Option | Also removes |
|---|---|
| `--results` | `results/`: metrics.csv, tables, figures, predictions |
| `--checkpoints` | `checkpoints/`: the trained LoRA / QLoRA adapters |
| `--data` | `data/processed` (rebuilt by `01_prepare_data.py`) |
| `--raw-data` | `data/raw` and the kagglehub download cache |
| `--hf-cache` | the cached Chronos-2 weights (about 0.5 GB, downloaded again on next use) |
| `--pip-cache` | pip's download cache |
| `--venv` | the virtual environment |
| `--gpu-kill` | leftover project Python processes |
| `--all` | everything above except `--venv` and `--gpu-kill` |
| `--dry-run` | nothing: only lists what would be removed |

Examples:

```bash
./cleanup.zsh --dry-run --all
```

```bash
./cleanup.zsh --results --checkpoints
```

To remove every trace of the project from the machine: `--all --venv`, then delete the folder.

---

## 7. Tests

With the environment active:

```bash
pytest -m "not slow"
```

```bash
pytest -m slow
```

```bash
ruff check .
```

The fast suite needs no downloads. It uses synthetic data and a tiny, randomly initialised model with the
real Chronos-2 architecture, and covers the data cleaning rules, the split, the metrics, that the target is
never leaked into the model inputs, that LoRA trains only the adapter, saving and reloading adapters, the
QLoRA fallback, and the whole pipeline through to the report. The slow test downloads the real Chronos-2
and runs zero-shot plus two LoRA steps. CI runs the fast suite.

---

## 8. Repository layout

```
configs/         default.yaml (the experiment), smoke.yaml (fast check on fake data)
scripts/         00..04 pipeline steps, run_all.py, check_env.py, gpu_cleanup.py
src/windfc/      chronos.py   Chronos-2 inputs, rolling forecasts, LoRA / QLoRA fine-tuning
                 pipeline.py  the stages          report.py   tables and figures
                 data.py      loading, imputation, cleaning    download.py, synthetic.py
                 splits.py    chronological split and targets  metrics.py, evaluate.py
                 paper.py     the paper's Table 5
tests/           pytest suite
results/         metrics.csv, tables/, figures/ (predictions/ is git-ignored)
setup_env.*  run_pipeline.*  cleanup.*      one of each for zsh, PowerShell and cmd
```

## 9. Notes

* Chronos-2 fits in memory without quantisation (120M parameters), so QLoRA here is about studying the
  technique, not about making the model fit. Word the slides according to what you ran: if the run was on
  a Mac, it was LoRA.
* The code is written against `chronos-forecasting` 2.3. If a later release breaks it, pin
  `chronos-forecasting==2.3.*` in `pyproject.toml`.
* Chronos-2 handles horizons up to 1,024 steps and contexts up to 8,192, far beyond what is used here.
