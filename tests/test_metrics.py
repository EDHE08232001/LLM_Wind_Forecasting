import numpy as np
import pytest

from windfc.metrics import binned_errors, bootstrap_rmse_diff, r2_pearson, regression_metrics, skill_score


def test_hand_computed():
    y = np.array([0.0, 1.0, 2.0, 3.0])
    p = np.array([0.0, 2.0, 2.0, 1.0])
    m = regression_metrics(y, p, scale_range=10.0)
    assert m["mae_kw"] == pytest.approx(0.75)
    assert m["mse_kw"] == pytest.approx(1.25)
    assert m["rmse_kw"] == pytest.approx(np.sqrt(1.25))
    assert m["r2"] == pytest.approx(1 - 5.0 / 5.0)
    assert m["mae_scaled"] == pytest.approx(0.075)
    assert m["mse_scaled"] == pytest.approx(0.0125)


def test_perfect_and_skill():
    y = np.arange(10.0)
    assert regression_metrics(y, y)["r2"] == 1.0
    assert skill_score(5.0, 10.0) == pytest.approx(0.5)


def test_paper_r2_ignores_bias_but_ours_does_not():
    y = np.arange(10.0)
    biased = 2 * y + 5  # perfectly correlated, badly wrong
    assert r2_pearson(y, biased) == pytest.approx(1.0)
    assert regression_metrics(y, biased)["r2"] < 0


def test_binned():
    y = np.array([0, 0, 100, 100.0])
    p = np.array([10, 0, 100, 50.0])
    ws = np.array([1, 2, 11, 15.0])
    b = binned_errors(y, p, ws, [0, 5, 10, 13, 26])
    assert b["n"].tolist() == [2, 0, 1, 1]
    assert b.loc[0, "mae_kw"] == pytest.approx(5.0)
    assert b.loc[3, "bias_kw"] == pytest.approx(-50.0)


def test_bootstrap_sign():
    rng = np.random.default_rng(0)
    y = rng.normal(size=500)
    good, bad = y + rng.normal(0, 0.1, 500), y + rng.normal(0, 1.0, 500)
    r = bootstrap_rmse_diff(y, good, bad, n_boot=300)
    assert r["rmse_diff"] < 0 and r["ci_high"] < 0 and r["p_a_better"] > 0.99
