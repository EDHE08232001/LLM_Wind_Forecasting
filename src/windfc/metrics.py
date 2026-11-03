"""Error metrics in kW and on the min-max scale, binned errors and paired bootstrap."""

from __future__ import annotations

import numpy as np
import pandas as pd


def mae(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(yhat))))


def mse(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2))


def rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.sqrt(mse(y, yhat)))


def r2(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    ss_res = np.sum((y - np.asarray(yhat)) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def regression_metrics(y: np.ndarray, yhat: np.ndarray, scale_range: float | None = None) -> dict[str, float]:
    """MAE / MSE / RMSE / R^2 in kW, plus min-max-scaled versions (paper Table 5 units).

    Scaled metrics divide errors by the min-max range of the training target:
    MAE_s = MAE / range, MSE_s = MSE / range^2, RMSE_s = RMSE / range.
    """
    y = np.asarray(y, dtype=np.float64)
    yhat = np.asarray(yhat, dtype=np.float64)
    out = {
        "n": int(len(y)),
        "mae_kw": mae(y, yhat),
        "mse_kw": mse(y, yhat),
        "rmse_kw": rmse(y, yhat),
        "r2": r2(y, yhat),
    }
    if scale_range:
        out["mae_scaled"] = out["mae_kw"] / scale_range
        out["mse_scaled"] = out["mse_kw"] / scale_range**2
        out["rmse_scaled"] = out["rmse_kw"] / scale_range
    return out


def skill_score(rmse_model: float, rmse_reference: float) -> float:
    """Persistence-relative skill: 1 - RMSE_model / RMSE_reference."""
    return float(1.0 - rmse_model / rmse_reference) if rmse_reference > 0 else float("nan")


def binned_errors(
    y: np.ndarray, yhat: np.ndarray, wind_speed: np.ndarray, bins: list[float] = (0, 5, 10, 13, 26)
) -> pd.DataFrame:
    y, yhat, ws = (np.asarray(a, dtype=np.float64) for a in (y, yhat, wind_speed))
    cats = pd.cut(ws, bins=list(bins), right=False, include_lowest=True)
    df = pd.DataFrame({"bin": cats, "err": yhat - y})
    rows = []
    for b, g in df.groupby("bin", observed=False):
        e = g["err"].to_numpy()
        rows.append(
            {
                "wind_bin": str(b),
                "n": int(len(e)),
                "mae_kw": float(np.mean(np.abs(e))) if len(e) else np.nan,
                "rmse_kw": float(np.sqrt(np.mean(e**2))) if len(e) else np.nan,
                "bias_kw": float(np.mean(e)) if len(e) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def bootstrap_rmse_diff(
    y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, n_boot: int = 2000, seed: int = 0, block: int = 24
) -> dict[str, float]:
    """Paired (moving-block) bootstrap of RMSE(a) - RMSE(b). Negative = a is better.

    Blocks of ``block`` consecutive hours keep the autocorrelation of errors.
    """
    y, a, b = (np.asarray(v, dtype=np.float64) for v in (y, pred_a, pred_b))
    n = len(y)
    rng = np.random.default_rng(seed)
    ea, eb = (a - y) ** 2, (b - y) ** 2
    block = max(1, min(block, n))
    n_blocks = int(np.ceil(n / block))
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
        diffs[i] = np.sqrt(ea[idx].mean()) - np.sqrt(eb[idx].mean())
    point = float(np.sqrt(ea.mean()) - np.sqrt(eb.mean()))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"rmse_diff": point, "ci_low": float(lo), "ci_high": float(hi), "p_a_better": float(np.mean(diffs < 0))}
