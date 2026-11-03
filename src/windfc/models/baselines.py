"""Simple baselines: persistence, seasonal naive, linear regression, XGBoost."""

from __future__ import annotations

import itertools

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..evaluate import Forecaster
from ..metrics import rmse
from ..splits import ForecastTask


class Persistence(Forecaster):
    """y_hat(t + h) = y(t)."""

    name = "persistence"
    family = "baseline"

    def predict(self, task: ForecastTask, split: str) -> np.ndarray:
        return task.persistence(split)


class SeasonalNaive(Forecaster):
    """y_hat(T) = y(T - 24 * ceil(h / 24)). Identical to persistence when h = 24."""

    name = "seasonal_naive"
    family = "baseline"

    def __init__(self, period: int = 24):
        self.period = period

    def predict(self, task: ForecastTask, split: str) -> np.ndarray:
        return task.seasonal_naive(split, self.period)


class LinearModel(Forecaster):
    name = "linear"
    family = "baseline"

    def fit(self, task: ForecastTask) -> LinearModel:
        x, y = task.tabular("train")
        self.columns = list(x.columns)
        self.model = make_pipeline(StandardScaler(), LinearRegression()).fit(x.to_numpy(), y)
        return self

    def predict(self, task: ForecastTask, split: str) -> np.ndarray:
        x, _ = task.tabular(split)
        return self.model.predict(x[self.columns].to_numpy())


class XGBoostModel(Forecaster):
    """Gradient boosting (direct strategy: one model per horizon).

    With a validation split, a small grid is searched with early stopping on
    validation RMSE; otherwise ``default`` parameters are used (protocol A).
    """

    name = "xgboost"
    family = "baseline"

    def __init__(self, cfg: dict, device: str = "cpu", seed: int = 42):
        self.cfg = cfg
        self.device = "cuda" if str(device).startswith("cuda") else "cpu"
        self.seed = seed
        self.best_params: dict = {}

    def _make(self, **params):
        import xgboost as xgb

        return xgb.XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            device=self.device,
            random_state=self.seed,
            n_jobs=-1,
            **params,
        )

    def fit(self, task: ForecastTask) -> XGBoostModel:
        x_tr, y_tr = task.tabular("train")
        self.columns = list(x_tr.columns)
        if not task.has_val:
            params = dict(self.cfg["default"])
            self.model = self._make(**params).fit(x_tr.to_numpy(), y_tr)
            self.best_params = params
            return self
        x_va, y_va = task.tabular("val")
        grid = self.cfg["grid"]
        keys = list(grid)
        best = (np.inf, None, None)
        for values in itertools.product(*(grid[k] for k in keys)):
            params = dict(zip(keys, values))
            model = self._make(
                n_estimators=int(self.cfg["n_estimators"]),
                early_stopping_rounds=int(self.cfg["early_stopping_rounds"]),
                **params,
            )
            model.fit(x_tr.to_numpy(), y_tr, eval_set=[(x_va.to_numpy(), y_va)], verbose=False)
            score = rmse(y_va, model.predict(x_va.to_numpy()))
            if score < best[0]:
                best = (score, params, model)
        self.best_params = {**best[1], "best_iteration": int(getattr(best[2], "best_iteration", -1) or -1)}
        self.model = best[2]
        self.notes = f"params={self.best_params}"
        print(f"  [xgboost] best val RMSE={best[0]:.1f} kW with {self.best_params}")
        return self

    def predict(self, task: ForecastTask, split: str) -> np.ndarray:
        x, _ = task.tabular(split)
        return self.model.predict(x[self.columns].to_numpy())


def build_baseline(name: str, cfg: dict, device: str = "cpu", seed: int = 42) -> Forecaster:
    name = name.lower()
    if name == "persistence":
        return Persistence()
    if name == "seasonal_naive":
        return SeasonalNaive()
    if name == "linear":
        return LinearModel()
    if name == "xgboost":
        return XGBoostModel(cfg["baselines"]["xgboost"], device=device, seed=seed)
    raise ValueError(f"Unknown baseline {name!r}")
