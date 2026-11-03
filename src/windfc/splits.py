"""Train/val/test splits and the shared :class:`ForecastTask` used by every model.

Protocol A (paper): random 75/25 split of the *cleaned* hourly rows (seed 42),
same-hour regression of active power from same-hour inputs.

Protocol B (ours): chronological split (train Jan-Aug, val Sep-Oct, test Nov-Dec),
direct ``h``-hour-ahead forecasting from inputs known at the origin ``T - h``.
Samples are indexed by target time; a target belongs to the split its timestamp
falls in. Inputs may come from earlier periods (that is just the past, not a
leak), so every model is scored on exactly the same test targets.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import Config
from .features import required_history, sliding_windows, tabular_features

SPLITS = ("train", "val", "test")


def chronological_labels(index: pd.DatetimeIndex, train_end: str, val_end: str) -> np.ndarray:
    idx = pd.DatetimeIndex(index)
    labels = np.full(len(idx), "test", dtype=object)
    labels[idx <= pd.Timestamp(val_end)] = "val"
    labels[idx <= pd.Timestamp(train_end)] = "train"
    return labels


def chronological_split(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pb = cfg["protocol_b"]
    labels = chronological_labels(df.index, pb["train_end"], pb["val_end"])
    return df[labels == "train"], df[labels == "val"], df[labels == "test"]


def random_split(df: pd.DataFrame, test_size: float = 0.25, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_idx, test_idx = train_test_split(np.arange(len(df)), test_size=test_size, random_state=seed)
    return df.iloc[np.sort(train_idx)], df.iloc[np.sort(test_idx)]


@dataclass
class MinMax:
    """Column-wise min-max scaler (fit on training rows only)."""

    lo: np.ndarray = field(default_factory=lambda: np.zeros(1))
    hi: np.ndarray = field(default_factory=lambda: np.ones(1))

    def fit(self, x: np.ndarray) -> MinMax:
        x = np.asarray(x, dtype=np.float64)
        self.lo = np.nanmin(x, axis=0)
        self.hi = np.nanmax(x, axis=0)
        return self

    @property
    def span(self) -> np.ndarray:
        span = self.hi - self.lo
        return np.where(span == 0, 1.0, span)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=np.float64) - self.lo) / self.span

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=np.float64) * self.span + self.lo


class ForecastTask:
    """Everything a model needs for one (protocol, horizon) experiment."""

    def __init__(self, df: pd.DataFrame, cfg: Config, protocol: str = "B", horizon: int = 1):
        self.cfg = cfg
        self.protocol = protocol.upper()
        self.horizon = int(horizon) if self.protocol == "B" else 0
        self.df = df.sort_index()
        fcfg = cfg["features"]
        self.lookback = int(fcfg["lookback"])
        self.lags = list(fcfg["lags"])
        self.rolling = list(fcfg["rolling_windows"])
        self.target = "active_power"
        self._positions: dict[str, np.ndarray] = {}

        if self.protocol == "A":
            self.input_columns = list(fcfg["paper_inputs"])
            clean_pos = np.flatnonzero(self.df["is_clean"].to_numpy())
            pa = cfg["protocol_a"]
            tr, te = train_test_split(clean_pos, test_size=float(pa["test_size"]), random_state=int(pa["seed"]))
            self._positions = {"train": np.sort(tr), "val": np.array([], dtype=int), "test": np.sort(te)}
            self.window_columns = self.input_columns
        elif self.protocol == "B":
            pb = cfg["protocol_b"]
            labels = chronological_labels(self.df.index, pb["train_end"], pb["val_end"])
            self.labels = labels
            need = required_history(self.lags, self.rolling, self.lookback)
            first_valid = self.horizon + need - 1
            pos = np.arange(len(self.df))
            ok = pos >= first_valid
            for s in SPLITS:
                self._positions[s] = pos[ok & (labels == s)]
            self.window_columns = list(fcfg["window_inputs"])
            self.input_columns = self.window_columns
        else:
            raise ValueError(f"Unknown protocol {protocol!r}")

        train_rows = self._train_rows()
        self.x_scaler = MinMax().fit(self.df[self.window_columns].to_numpy()[train_rows])
        self.y_scaler = MinMax().fit(self.df[[self.target]].to_numpy()[train_rows])
        self._tab_cache: pd.DataFrame | None = None

    # ------------------------------------------------------------------ basics
    def _train_rows(self) -> np.ndarray:
        if self.protocol == "A":
            return self._positions["train"]
        return np.flatnonzero(self.labels == "train")

    @property
    def name(self) -> str:
        return "A" if self.protocol == "A" else f"B_h{self.horizon}"

    @property
    def has_val(self) -> bool:
        return len(self._positions["val"]) > 0

    @property
    def target_range(self) -> float:
        return float(self.y_scaler.span[0])

    def positions(self, split: str) -> np.ndarray:
        return self._positions[split]

    def target_index(self, split: str) -> pd.DatetimeIndex:
        return self.df.index[self._positions[split]]

    def y(self, split: str) -> np.ndarray:
        return self.df[self.target].to_numpy()[self._positions[split]]

    def column(self, split: str, col: str) -> np.ndarray:
        return self.df[col].to_numpy()[self._positions[split]]

    def wind_speed(self, split: str) -> np.ndarray:
        return self.column(split, "wind_speed")

    def is_clean(self, split: str) -> np.ndarray:
        return self.column(split, "is_clean").astype(bool)

    def persistence(self, split: str) -> np.ndarray:
        """Value at the forecast origin (protocol B only)."""
        if self.protocol != "B":
            raise ValueError("persistence is only defined for protocol B")
        return self.df[self.target].to_numpy()[self._positions[split] - self.horizon]

    def seasonal_naive(self, split: str, period: int = 24) -> np.ndarray:
        lag = period * int(np.ceil(self.horizon / period))
        return self.df[self.target].to_numpy()[self._positions[split] - lag]

    def scale_y(self, y: np.ndarray) -> np.ndarray:
        return self.y_scaler.transform(np.asarray(y).reshape(-1, 1)).ravel()

    def unscale_y(self, y: np.ndarray) -> np.ndarray:
        return self.y_scaler.inverse(np.asarray(y).reshape(-1, 1)).ravel()

    # ---------------------------------------------------------------- tabular
    def tabular(self, split: str) -> tuple[pd.DataFrame, np.ndarray]:
        """Design matrix for linear / tree models (kW units, unscaled)."""
        pos = self._positions[split]
        if self.protocol == "A":
            x = self.df[self.input_columns].iloc[pos]
            return x.reset_index(drop=True), self.y(split)
        if self._tab_cache is None:
            f = self.cfg["features"]
            self._tab_cache = tabular_features(
                self.df,
                self.horizon,
                lags=f["lags"],
                lag_columns=f["lag_columns"],
                rolling_windows=f["rolling_windows"],
                current_columns=f["current_columns"],
            )
        x = self._tab_cache.iloc[pos]
        if x.isna().any().any():
            x = x.fillna(0.0)
        return x.reset_index(drop=True), self.y(split)

    # ---------------------------------------------------------------- windows
    def windows(self, split: str) -> tuple[np.ndarray, np.ndarray]:
        """Scaled inputs for the neural nets: (N, L, C) float32, and scaled targets (N,)."""
        pos = self._positions[split]
        mat = self.x_scaler.transform(self.df[self.window_columns].to_numpy()).astype(np.float32)
        if self.protocol == "A":
            # The paper reshapes the feature vector into a 1-D "sequence" (F steps, 1 channel).
            x = mat[pos][:, :, None]
        else:
            x = sliding_windows(mat, pos - self.horizon, self.lookback)
        y = self.scale_y(self.y(split)).astype(np.float32)
        return x, y

    def describe(self) -> str:
        sizes = {s: len(p) for s, p in self._positions.items()}
        return f"task {self.name}: {sizes} inputs={len(self.window_columns)} lookback={self.lookback}"
