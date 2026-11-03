"""Dataset download helpers.

* SCADA: Kaggle dataset ``berkerisen/wind-turbine-scada-dataset`` (file ``T1.csv``),
  fetched with the official ``kagglehub`` client; if that fails, the README
  explains the manual download.
* Weather: MERRA-2 meteorology for the turbine location from NASA POWER
  (https://power.larc.nasa.gov, no API key). The paper used the renewables.ninja
  MERRA-2 export; that API no longer serves 2018 to anonymous users, but a CSV
  exported from the website can be dropped into ``data/raw`` instead
  (``data.load_weather`` understands both formats).
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import requests

KAGGLE_DATASET = "berkerisen/wind-turbine-scada-dataset"
NASA_POWER_URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"
# Temperature, precipitation, pressure + humidity (-> air density), ground and
# top-of-atmosphere irradiance, cloud amount, and 50 m wind (used only for imputation).
NASA_POWER_PARAMS = [
    "T2M",
    "PRECTOTCORR",
    "PS",
    "QV2M",
    "ALLSKY_SFC_SW_DWN",
    "TOA_SW_DWN",
    "CLOUD_AMT",
    "WS50M",
    "WD50M",
]


def _get(url: str, retries: int = 4, **kwargs) -> requests.Response:
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=kwargs.pop("timeout", 120), **kwargs)
            resp.raise_for_status()
            return resp
        except Exception as exc:  # network hiccup -> exponential backoff
            last_exc = exc
            wait = 2 ** (attempt + 1)
            print(f"[download] {url} failed ({exc}); retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"Download failed after {retries} attempts: {url}") from last_exc


def download_scada(out_csv: Path, force: bool = False) -> Path:
    """Fetch ``T1.csv`` with Kaggle's official ``kagglehub`` client (anonymous for public datasets).

    If ``kagglehub`` is unavailable or the download fails, follow the manual steps
    in the README (browser download or the ``curl`` command shown on the Kaggle page).
    """
    out_csv = Path(out_csv)
    if out_csv.exists() and not force:
        print(f"[download] SCADA already present: {out_csv}")
        return out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError("kagglehub is not installed (pip install kagglehub)") from exc
    print(f"[download] Kaggle {KAGGLE_DATASET} via kagglehub ...")
    folder = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    csvs = sorted(folder.rglob("*.csv"))
    if not csvs:
        raise RuntimeError(f"No CSV found in {folder}")
    shutil.copyfile(csvs[0], out_csv)
    print(f"[download] wrote {out_csv} ({out_csv.stat().st_size / 1e6:.1f} MB)")
    return out_csv


def download_weather_nasa_power(
    out_csv: Path,
    latitude: float,
    longitude: float,
    year: int = 2018,
    force: bool = False,
) -> Path:
    out_csv = Path(out_csv)
    if out_csv.exists() and not force:
        print(f"[download] weather already present: {out_csv}")
        return out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    params = {
        "parameters": ",".join(NASA_POWER_PARAMS),
        "community": "RE",
        "latitude": latitude,
        "longitude": longitude,
        "start": f"{year}0101",
        "end": f"{year}1231",
        "format": "CSV",
        "time-standard": "UTC",
    }
    print(f"[download] NASA POWER (MERRA-2) hourly weather for ({latitude}, {longitude}), {year} ...")
    resp = _get(NASA_POWER_URL, params=params, timeout=300)
    text = resp.text
    if "-END HEADER-" not in text:
        raise RuntimeError(f"Unexpected NASA POWER response: {text[:500]}")
    out_csv.write_text(text, encoding="utf-8")
    print(f"[download] wrote {out_csv} ({out_csv.stat().st_size / 1e6:.1f} MB)")
    return out_csv
