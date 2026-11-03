"""Load, impute, resample, merge and clean the SCADA + MERRA-2 data.

Pipeline (``build_hourly_dataset``), following Karaman (2023) Sec. 2.1:

1. Load the 10-min SCADA file and reindex it onto a complete 10-min grid
   (the Kaggle file has 50,530 of 52,560 rows -> 2,030 missing, as in the paper).
2. Load hourly weather (NASA POWER MERRA-2 or renewables.ninja export).
3. Impute missing SCADA rows with kNN (k=5). Distances use the weather
   variables + time-of-day (the SCADA values themselves are absent for
   missing rows). The imputer is fit on the training period only.
4. Keep hourly samples ("first" = the on-the-hour record, as the paper
   "filtered only the hourly data") or hourly means.
5. Merge with weather on the UTC timestamp.
6. Compute cleaning flags (cut-in / cut-out, non-positive power, IQR outliers).
   Rows are *flagged* rather than dropped so the series stays regular for
   forecasting; protocol A drops flagged rows like the paper does.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

from .config import Config, resolve_path

SCADA_COLUMNS = {
    "LV ActivePower (kW)": "active_power",
    "Wind Speed (m/s)": "wind_speed",
    "Theoretical_Power_Curve (KWh)": "theoretical_power",
    "Wind Direction (°)": "wind_direction",
}
SCADA_VARS = ["active_power", "wind_speed", "theoretical_power", "wind_direction"]
WEATHER_VARS = [
    "temperature",
    "precipitation",
    "air_density",
    "ground_irradiance",
    "toa_irradiance",
    "cloud_cover",
]
FLAG_COLUMNS = ["flag_low_wind", "flag_high_wind", "flag_nonpositive_power", "flag_iqr_outlier"]

R_DRY = 287.05  # J / (kg K)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_scada(path: str | Path, utc_offset_hours: float = 0.0) -> pd.DataFrame:
    """Load the Kaggle SCADA csv -> DataFrame indexed by (UTC) timestamp."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    rename = {}
    for col in df.columns:
        for raw, new in SCADA_COLUMNS.items():
            if col.lower().replace(" ", "") == raw.lower().replace(" ", ""):
                rename[col] = new
        if col.lower().startswith("date"):
            rename[col] = "timestamp"
        if col.lower().startswith("wind direction"):
            rename[col] = "wind_direction"
    df = df.rename(columns=rename)
    missing = {"timestamp", *SCADA_VARS} - set(df.columns)
    if missing:
        raise ValueError(f"SCADA file {path} is missing columns {missing}; got {list(df.columns)}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%d %m %Y %H:%M")
    if utc_offset_hours:
        df["timestamp"] = df["timestamp"] - pd.Timedelta(hours=float(utc_offset_hours))
    df = df.sort_values("timestamp").drop_duplicates("timestamp", keep="first")
    return df.set_index("timestamp")[SCADA_VARS].astype("float64")


def _read_nasa_power(text: str) -> pd.DataFrame:
    body = text.split("-END HEADER-", 1)[1].lstrip()
    df = pd.read_csv(io.StringIO(body))
    df = df.replace(-999.0, np.nan)
    ts = pd.to_datetime(
        dict(year=df["YEAR"], month=df["MO"], day=df["DY"], hour=df["HR"]),
    )
    t_k = df["T2M"] + 273.15
    q = df["QV2M"] / 1000.0  # g/kg -> kg/kg
    rho = df["PS"] * 1000.0 / (R_DRY * t_k * (1.0 + 0.608 * q))  # kPa -> Pa
    out = pd.DataFrame(
        {
            "temperature": df["T2M"].to_numpy(),
            "precipitation": df["PRECTOTCORR"].to_numpy(),
            "air_density": rho.to_numpy(),
            "ground_irradiance": df["ALLSKY_SFC_SW_DWN"].to_numpy(),
            "toa_irradiance": df["TOA_SW_DWN"].to_numpy(),
            "cloud_cover": (df["CLOUD_AMT"] / 100.0).to_numpy(),  # % -> fraction (like MERRA-2 cldtot)
        },
        index=pd.DatetimeIndex(ts, name="timestamp"),
    )
    if "WS50M" in df:
        out["merra_ws50m"] = df["WS50M"].to_numpy()
    if "WD50M" in df:
        out["merra_wd50m"] = df["WD50M"].to_numpy()
    return out


def _read_ninja(text: str) -> pd.DataFrame:
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    df = pd.read_csv(io.StringIO("\n".join(lines)))
    cols = {c.lower(): c for c in df.columns}
    tcol = cols.get("time") or cols.get("utc_time") or df.columns[0]
    ts = pd.to_datetime(df[tcol], utc=True).dt.tz_convert(None)

    def pick(*names: str) -> pd.Series:
        for n in names:
            if n in cols:
                return pd.to_numeric(df[cols[n]], errors="coerce")
        return pd.Series(np.nan, index=df.index)

    out = pd.DataFrame(
        {
            "temperature": pick("t2m", "temperature").to_numpy(),
            "precipitation": pick("prectotland", "precipitation", "prectot").to_numpy(),
            "air_density": pick("rhoa", "air_density").to_numpy(),
            "ground_irradiance": pick("swgdn", "ground_irradiance", "irradiance_surface").to_numpy(),
            "toa_irradiance": pick("swtdn", "toa_irradiance", "irradiance_toa").to_numpy(),
            "cloud_cover": pick("cldtot", "cloud_cover").to_numpy(),
        },
        index=pd.DatetimeIndex(ts, name="timestamp"),
    )
    return out


def load_weather(path: str | Path) -> pd.DataFrame:
    """Load hourly weather (NASA POWER CSV, renewables.ninja CSV or a tidy CSV)."""
    text = Path(path).read_text(encoding="utf-8-sig")
    if "-END HEADER-" in text:
        df = _read_nasa_power(text)
    else:
        df = _read_ninja(text)
    df = df[~df.index.duplicated(keep="first")].sort_index()
    df = df.interpolate(method="time", limit_direction="both")
    if df[WEATHER_VARS].isna().all().any():
        empty = [c for c in WEATHER_VARS if df[c].isna().all()]
        raise ValueError(f"Weather file {path} has no values for {empty}")
    return df


# ---------------------------------------------------------------------------
# Imputation / resampling / cleaning
# ---------------------------------------------------------------------------
def add_direction_components(df: pd.DataFrame, col: str = "wind_direction", prefix: str = "wd") -> pd.DataFrame:
    rad = np.deg2rad(df[col])
    df[f"{prefix}_sin"] = np.sin(rad)
    df[f"{prefix}_cos"] = np.cos(rad)
    return df


def direction_from_components(s: pd.Series | np.ndarray, c: pd.Series | np.ndarray) -> np.ndarray:
    return np.mod(np.rad2deg(np.arctan2(s, c)), 360.0)


def knn_impute_scada(
    scada: pd.DataFrame,
    weather_10min: pd.DataFrame,
    k: int = 5,
    fit_mask: np.ndarray | None = None,
) -> tuple[pd.DataFrame, int]:
    """Impute missing SCADA rows with kNN on weather + time-of-day features.

    Returns the imputed frame and the number of imputed rows.
    """
    df = scada.copy()
    df = add_direction_components(df)
    target_cols = ["active_power", "wind_speed", "wd_sin", "wd_cos"]
    missing_rows = df[target_cols].isna().any(axis=1).to_numpy()
    n_missing = int(missing_rows.sum())
    if n_missing == 0:
        return scada.copy(), 0

    aux = weather_10min.reindex(df.index).copy()
    if "merra_wd50m" in aux:
        aux = add_direction_components(aux, "merra_wd50m", "mwd")
        aux = aux.drop(columns=["merra_wd50m"])
    hours = df.index.hour + df.index.minute / 60.0
    aux["hod_sin"] = np.sin(2 * np.pi * hours / 24.0)
    aux["hod_cos"] = np.cos(2 * np.pi * hours / 24.0)
    aux = aux.interpolate(limit_direction="both").fillna(0.0)

    fit_mask = np.ones(len(df), dtype=bool) if fit_mask is None else np.asarray(fit_mask, dtype=bool)
    matrix = pd.concat([df[target_cols], aux], axis=1)
    scaler = StandardScaler().fit(matrix[fit_mask])
    scaled = scaler.transform(matrix)
    imputer = KNNImputer(n_neighbors=k, weights="uniform")
    imputer.fit(scaled[fit_mask])
    filled = scaled.copy()
    filled[missing_rows] = imputer.transform(scaled[missing_rows])
    filled = scaler.inverse_transform(filled)
    out = pd.DataFrame(filled[:, : len(target_cols)], index=df.index, columns=target_cols)
    # Keep observed values exactly; only replace the gaps.
    for col in target_cols:
        out[col] = df[col].where(df[col].notna(), out[col])

    result = pd.DataFrame(index=df.index)
    result["active_power"] = out["active_power"]
    result["wind_speed"] = out["wind_speed"].clip(lower=0.0)
    result["wind_direction"] = direction_from_components(out["wd_sin"], out["wd_cos"])
    # Theoretical power is a deterministic power curve of wind speed: re-derive it.
    known = df[["wind_speed", "theoretical_power"]].dropna().sort_values("wind_speed")
    curve_ws = known["wind_speed"].to_numpy()
    curve_p = known["theoretical_power"].to_numpy()
    theo = df["theoretical_power"].copy()
    gaps = theo.isna().to_numpy()
    if len(curve_ws):
        theo[gaps] = np.interp(result["wind_speed"].to_numpy()[gaps], curve_ws, curve_p)
    result["theoretical_power"] = theo
    result["imputed"] = missing_rows
    return result, n_missing


def to_hourly(df10: pd.DataFrame, how: str = "first") -> pd.DataFrame:
    if how == "first":
        hourly = df10[df10.index.minute == 0].copy()
        hourly.index = hourly.index.floor("h")
        return hourly
    if how == "mean":
        tmp = add_direction_components(df10.copy())
        agg = tmp.resample("1h").mean(numeric_only=True)
        agg["wind_direction"] = direction_from_components(agg["wd_sin"], agg["wd_cos"])
        if "imputed" in df10:
            agg["imputed"] = df10["imputed"].resample("1h").max().astype(bool)
        return agg.drop(columns=["wd_sin", "wd_cos"])
    raise ValueError(f"Unknown resample mode {how!r}")


def compute_clean_flags(
    df: pd.DataFrame,
    cut_in: float = 3.5,
    cut_out: float = 25.5,
    iqr_factor: float = 1.5,
) -> tuple[pd.DataFrame, dict]:
    """Add the paper's cleaning flags (applied sequentially, like the paper)."""
    df = df.copy()
    counts: dict[str, int] = {"rows": int(len(df))}
    df["flag_low_wind"] = df["wind_speed"] < cut_in
    df["flag_high_wind"] = df["wind_speed"] > cut_out
    remaining = ~(df["flag_low_wind"] | df["flag_high_wind"])
    counts["removed_wind_out_of_range"] = int((~remaining).sum())

    df["flag_nonpositive_power"] = remaining & (df["active_power"] <= 0)
    counts["removed_nonpositive_power"] = int(df["flag_nonpositive_power"].sum())
    remaining &= ~df["flag_nonpositive_power"]

    ap = df.loc[remaining, "active_power"]
    q1, q3 = ap.quantile(0.25), ap.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - iqr_factor * iqr, q3 + iqr_factor * iqr
    df["flag_iqr_outlier"] = remaining & ((df["active_power"] < lo) | (df["active_power"] > hi))
    counts["removed_iqr_outliers"] = int(df["flag_iqr_outlier"].sum())
    counts["iqr_bounds_kw"] = [float(lo), float(hi)]

    df["is_clean"] = ~df[FLAG_COLUMNS].any(axis=1)
    counts["clean_rows"] = int(df["is_clean"].sum())
    return df, counts


def build_hourly_dataset(cfg: Config, scada_csv: str | Path | None = None, weather_csv: str | Path | None = None,
                         save: bool = True) -> tuple[pd.DataFrame, dict]:
    paths = cfg["paths"]
    dcfg = cfg["data"]
    year = int(dcfg["year"])
    scada_csv = resolve_path(scada_csv or paths["scada_csv"])
    weather_csv = resolve_path(weather_csv or paths["weather_csv"])
    for p in (scada_csv, weather_csv):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found. Run `python scripts/00_download_data.py` or see README 'Dataset'."
            )

    report: dict = {}
    scada = load_scada(scada_csv, cfg["site"].get("scada_utc_offset_hours", 0))
    report["scada_rows_raw"] = int(len(scada))
    grid = pd.date_range(f"{year}-01-01 00:00", f"{year}-12-31 23:50", freq=dcfg.get("freq_raw", "10min"))
    scada = scada.reindex(grid)
    scada.index.name = "timestamp"
    report["scada_rows_grid"] = int(len(grid))
    report["scada_rows_missing"] = int(scada["active_power"].isna().sum())

    weather = load_weather(weather_csv)
    hourly_grid = pd.date_range(f"{year}-01-01 00:00", f"{year}-12-31 23:00", freq="1h")
    weather = weather.reindex(weather.index.union(hourly_grid)).interpolate(method="time", limit_direction="both")
    weather = weather.reindex(hourly_grid)
    weather.index.name = "timestamp"
    weather_10 = weather.reindex(grid, method="ffill")

    imp = dcfg.get("impute", {})
    fit_on = imp.get("fit_on", "train")
    if fit_on == "train":
        # Neighbours may only come from the training months (Jan-Aug).
        train_end = pd.Timestamp(cfg["protocol_b"]["train_end"])
        fit_mask = np.asarray(grid <= train_end + pd.Timedelta(minutes=59))
    else:
        fit_mask = np.ones(len(grid), dtype=bool)
    scada_imp, n_imp = knn_impute_scada(scada, weather_10, k=int(imp.get("k", 5)), fit_mask=fit_mask)
    report["imputed_rows_10min"] = n_imp
    report["impute_fit_on"] = fit_on

    hourly = to_hourly(scada_imp, dcfg.get("resample", "first"))
    hourly = hourly.reindex(hourly_grid)
    hourly.index.name = "timestamp"
    merged = hourly.join(weather, how="left")
    merged = add_direction_components(merged)
    if "merra_wd50m" in merged:
        merged = merged.drop(columns=["merra_wd50m"])
    merged["imputed"] = merged["imputed"].fillna(True).astype(bool)

    ccfg = dcfg.get("clean", {})
    merged, counts = compute_clean_flags(
        merged,
        cut_in=float(ccfg.get("cut_in_ms", 3.5)),
        cut_out=float(ccfg.get("cut_out_ms", 25.5)),
        iqr_factor=float(ccfg.get("iqr_factor", 1.5)),
    )
    report.update(counts)
    report["hourly_rows"] = int(len(merged))
    report["nan_cells"] = int(merged.drop(columns=["merra_ws50m"], errors="ignore").isna().sum().sum())

    if report["nan_cells"]:
        merged = merged.interpolate(limit_direction="both")
    if save:
        out = resolve_path(paths["processed"])
        out.parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(out)
        (out.parent / "prepare_report.json").write_text(json.dumps(report, indent=2))
        report["output"] = str(out)
    return merged, report


def load_processed(cfg: Config) -> pd.DataFrame:
    path = resolve_path(cfg["paths"]["processed"])
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python scripts/01_prepare_data.py` first.")
    df = pd.read_parquet(path)
    df.index = pd.DatetimeIndex(df.index, name="timestamp")
    return df
