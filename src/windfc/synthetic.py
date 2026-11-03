"""Synthetic SCADA + weather files with the same formats as the real downloads.

Used by the tests, the smoke config and ``scripts/demo.py --synthetic`` so the
whole pipeline can be exercised without downloading anything. The numbers are
physically plausible but NOT real data; never report results from them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RATED_KW = 3600.0


def power_curve(ws: np.ndarray, cut_in: float = 3.0, rated_ws: float = 13.0, cut_out: float = 25.0) -> np.ndarray:
    ws = np.asarray(ws, dtype=np.float64)
    frac = np.clip((ws - cut_in) / (rated_ws - cut_in), 0.0, 1.0) ** 3
    p = RATED_KW * frac
    p[(ws < cut_in) | (ws > cut_out)] = 0.0
    return p


def make_synthetic(out_dir: str | Path, year: int = 2018, seed: int = 0, missing_frac: float = 0.04) -> tuple[Path, Path]:
    """Write ``T1.csv`` (Kaggle format) and ``weather_merra2_<year>.csv`` (NASA POWER format)."""
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- hourly weather
    hours = pd.date_range(f"{year}-01-01", f"{year}-12-31 23:00", freq="1h")
    n_h = len(hours)
    doy = hours.day_of_year.to_numpy()
    hod = hours.hour.to_numpy()
    temp = 14 - 9 * np.cos(2 * np.pi * (doy - 15) / 365) + 4 * np.sin(2 * np.pi * (hod - 9) / 24)
    temp += np.cumsum(rng.normal(0, 0.15, n_h)) * 0.1 + rng.normal(0, 0.5, n_h)
    ps = 100.5 + 0.8 * np.sin(2 * np.pi * doy / 9.0) + rng.normal(0, 0.1, n_h)  # kPa
    qv = np.clip(6 + 4 * np.sin(2 * np.pi * (doy - 100) / 365) + rng.normal(0, 0.5, n_h), 1, None)  # g/kg
    cloud = np.clip(50 + 30 * np.sin(2 * np.pi * doy / 23.0) + rng.normal(0, 15, n_h), 0, 100)
    decl = 23.44 * np.sin(2 * np.pi * (284 + doy) / 365)
    lat = np.deg2rad(40.6)
    hour_angle = np.deg2rad(15 * (hod + 2 - 12))  # ~UTC+2 solar offset
    cos_z = np.sin(lat) * np.sin(np.deg2rad(decl)) + np.cos(lat) * np.cos(np.deg2rad(decl)) * np.cos(hour_angle)
    toa = np.clip(1361 * cos_z, 0, None)
    ground = toa * (0.75 - 0.5 * cloud / 100) * np.clip(rng.normal(1, 0.05, n_h), 0.7, 1.2)
    precip = np.where(rng.random(n_h) < 0.04 * (cloud / 60), rng.exponential(0.8, n_h), 0.0)

    # wind: AR(1) latent + diurnal + seasonal, Weibull-ish marginal
    z = np.zeros(n_h)
    for i in range(1, n_h):
        z[i] = 0.97 * z[i - 1] + rng.normal(0, 0.25)
    ws_h = np.clip(7.5 + 1.5 * np.cos(2 * np.pi * doy / 365) + 1.0 * np.sin(2 * np.pi * (hod - 14) / 24) + 2.6 * z,
                   0.2, 27)
    wd_h = np.mod(40 + 180 * (z > 0.5) + 30 * np.sin(2 * np.pi * doy / 50) + rng.normal(0, 25, n_h), 360)

    weather = pd.DataFrame({
        "YEAR": hours.year, "MO": hours.month, "DY": hours.day, "HR": hours.hour,
        "T2M": temp.round(2), "PRECTOTCORR": precip.round(2), "PS": ps.round(2), "QV2M": qv.round(2),
        "ALLSKY_SFC_SW_DWN": ground.round(2), "TOA_SW_DWN": toa.round(2), "CLOUD_AMT": cloud.round(2),
        "WS50M": (ws_h * 1.1).round(2), "WD50M": wd_h.round(2),
    })
    header = (
        "-BEGIN HEADER-\nSYNTHETIC data in NASA/POWER format (windfc.synthetic) - NOT REAL\n"
        "The value for missing source data that cannot be computed is: -999\n-END HEADER-\n"
    )
    weather_path = out_dir / f"weather_merra2_{year}.csv"
    weather_path.write_text(header + weather.to_csv(index=False))

    # ---------------- 10-min SCADA
    ten = pd.date_range(f"{year}-01-01", f"{year}-12-31 23:50", freq="10min")
    n = len(ten)
    ws = np.interp(np.arange(n) / 6.0, np.arange(n_h), ws_h) + rng.normal(0, 0.35, n)
    ws = np.clip(ws, 0, 30)
    wd = np.mod(np.interp(np.arange(n) / 6.0, np.arange(n_h), wd_h) + rng.normal(0, 5, n), 360)
    theo = power_curve(ws)
    density = (ps * 1000 / (287.05 * (temp + 273.15)))
    dens10 = np.interp(np.arange(n) / 6.0, np.arange(n_h), density)
    active = theo * (dens10 / 1.225) * rng.normal(0.97, 0.04, n)
    active = np.clip(active, -5, RATED_KW)
    # curtailment / downtime episodes with zero output
    for _ in range(25):
        s = rng.integers(0, n - 200)
        active[s:s + rng.integers(6, 150)] = 0.0
    keep = rng.random(n) > missing_frac
    scada = pd.DataFrame({
        "Date/Time": ten.strftime("%d %m %Y %H:%M"),
        "LV ActivePower (kW)": active,
        "Wind Speed (m/s)": ws,
        "Theoretical_Power_Curve (KWh)": theo,
        "Wind Direction (°)": wd,
    })[keep]
    scada_path = out_dir / "T1.csv"
    scada.to_csv(scada_path, index=False, encoding="utf-8-sig")
    return scada_path, weather_path
