"""Feature engineering: calendar features, origin-relative lags, rolling stats, windows.

Protocol B convention: a sample is identified by its *target* time ``T`` and
horizon ``h``. The forecast origin is ``t = T - h``; every input must be known
at ``t``. Lag features are named ``{col}_o{k}`` = value ``k`` hours *before the
origin* (``o0`` = last observed value at the origin).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    idx = pd.DatetimeIndex(df.index)
    hour = idx.hour.to_numpy()
    doy = idx.day_of_year.to_numpy()
    df["hod_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hod_cos"] = np.cos(2 * np.pi * hour / 24.0)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return df


def lag_offsets(lags: list[int]) -> list[int]:
    """Convert target-relative lags at h=1 (1, 2, 3, 6, ...) to origin offsets (0, 1, 2, 5, ...)."""
    return sorted({max(int(lag) - 1, 0) for lag in lags})


def required_history(lags: list[int], rolling_windows: list[int], lookback: int) -> int:
    """Number of hours of history (including the origin) needed by any feature."""
    need = [lookback]
    need += [o + 1 for o in lag_offsets(lags)]
    need += [int(w) for w in rolling_windows]
    return max(need)


def tabular_features(
    df: pd.DataFrame,
    horizon: int,
    lags: list[int],
    lag_columns: list[str],
    rolling_windows: list[int],
    current_columns: list[str],
) -> pd.DataFrame:
    """Build the tabular design matrix indexed by target time (direct strategy)."""
    h = int(horizon)
    feats: dict[str, pd.Series] = {}
    for col in lag_columns:
        for off in lag_offsets(lags):
            feats[f"{col}_o{off}"] = df[col].shift(h + off)
    for w in rolling_windows:
        roll = df["active_power"].rolling(int(w), min_periods=int(w))
        feats[f"active_power_mean{w}"] = roll.mean().shift(h)
        feats[f"active_power_std{w}"] = roll.std().shift(h)
    for col in current_columns:
        if col in lag_columns:
            continue  # already included as offset 0
        feats[f"{col}_o0"] = df[col].shift(h)
    cal = add_calendar(df[[]])  # calendar of the *target* time (known in advance)
    for col in ["hod_sin", "hod_cos", "doy_sin", "doy_cos"]:
        feats[col] = cal[col]
    return pd.DataFrame(feats, index=df.index)


def sliding_windows(matrix: np.ndarray, end_positions: np.ndarray, lookback: int) -> np.ndarray:
    """Return windows ``matrix[e-lookback+1 : e+1]`` for each end position ``e``.

    ``matrix`` has shape (T, C); output shape is (N, lookback, C).
    """
    end_positions = np.asarray(end_positions, dtype=int)
    if len(end_positions) and (end_positions.min() < lookback - 1 or end_positions.max() >= len(matrix)):
        raise IndexError("window end positions out of range")
    offsets = np.arange(-lookback + 1, 1)
    idx = end_positions[:, None] + offsets[None, :]
    return matrix[idx]
