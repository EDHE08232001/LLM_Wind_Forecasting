"""Chronological split and the :class:`ForecastTask` every evaluation goes through.

The year is split by time: train Jan-Aug, validation Sep-Oct, test Nov-Dec
(``split`` in the config). A sample is identified by its *target* hour ``T``
and belongs to the split ``T`` falls in. For horizon ``h`` the forecast is
issued at the origin ``T - h``. Model inputs may reach back into earlier
months (that is just the past, not a leak), so every test hour is scored.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config

SPLITS = ("train", "val", "test")


def chronological_labels(index: pd.DatetimeIndex, train_end: str, val_end: str) -> np.ndarray:
    idx = pd.DatetimeIndex(index)
    labels = np.full(len(idx), "test", dtype=object)
    labels[idx <= pd.Timestamp(val_end)] = "val"
    labels[idx <= pd.Timestamp(train_end)] = "train"
    return labels


def split_labels(df: pd.DataFrame, cfg: Config) -> np.ndarray:
    return chronological_labels(df.index, cfg["split"]["train_end"], cfg["split"]["val_end"])


class ForecastTask:
    """Targets, references and bookkeeping for one forecast horizon."""

    target = "active_power"

    def __init__(self, df: pd.DataFrame, cfg: Config, horizon: int = 1, min_history: int = 24):
        self.cfg = cfg
        self.horizon = int(horizon)
        self.df = df.sort_index()
        self.labels = split_labels(self.df, cfg)
        pos = np.arange(len(self.df))
        ok = pos >= self.horizon + int(min_history)  # the first targets have too little history
        self._positions = {s: pos[ok & (self.labels == s)] for s in SPLITS}
        y_train = self.df[self.target].to_numpy()[self.labels == "train"]
        self.y_min, self.y_max = float(np.nanmin(y_train)), float(np.nanmax(y_train))

    @property
    def name(self) -> str:
        return f"h{self.horizon}"

    @property
    def target_range(self) -> float:
        """Min-max range of the training target: divides kW errors into the paper's scaled units."""
        return (self.y_max - self.y_min) or 1.0

    def positions(self, split: str) -> np.ndarray:
        return self._positions[split]

    def target_index(self, split: str) -> pd.DatetimeIndex:
        return self.df.index[self._positions[split]]

    def column(self, split: str, col: str) -> np.ndarray:
        return self.df[col].to_numpy()[self._positions[split]]

    def y(self, split: str) -> np.ndarray:
        return self.column(split, self.target)

    def wind_speed(self, split: str) -> np.ndarray:
        return self.column(split, "wind_speed")

    def is_clean(self, split: str) -> np.ndarray:
        """Targets that pass the paper's cleaning rules (the rows the paper kept)."""
        return self.column(split, "is_clean").astype(bool)

    def persistence(self, split: str) -> np.ndarray:
        """Reference forecast: the power observed at the origin ``T - h``."""
        return self.df[self.target].to_numpy()[self._positions[split] - self.horizon]

    def power_curve(self, split: str) -> np.ndarray:
        """Reference estimate: the manufacturer's theoretical power at the target hour's wind speed."""
        return self.column(split, "theoretical_power")

    def describe(self) -> str:
        sizes = {s: len(p) for s, p in self._positions.items()}
        return f"task {self.name}: targets={sizes} target_range={self.target_range:.0f} kW"
