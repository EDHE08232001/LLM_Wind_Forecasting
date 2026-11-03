import numpy as np
import pandas as pd

from windfc.features import lag_offsets, sliding_windows, tabular_features
from windfc.splits import ForecastTask, chronological_labels


def test_chronological_labels():
    idx = pd.date_range("2018-08-31 22:00", periods=4, freq="h")
    lab = chronological_labels(idx, "2018-08-31 23:00", "2018-09-01 00:00")
    assert lab.tolist() == ["train", "train", "val", "test"]


def test_protocol_b_no_overlap_and_alignment(tmp_cfg, hourly):
    for h in (1, 24):
        t = ForecastTask(hourly, tmp_cfg, "B", h)
        tr, va, te = (t.target_index(s) for s in ("train", "val", "test"))
        assert tr.max() < va.min() and va.max() < te.min()
        assert te.min() == pd.Timestamp("2018-11-01 00:00")  # every test target is scored
        # persistence = value at the origin T - h
        np.testing.assert_array_equal(t.persistence("test"), hourly["active_power"].shift(h).loc[te].to_numpy())
        # windows end at the origin: last step of the active_power channel == persistence (scaled)
        x, y = t.windows("test")
        ch = t.window_columns.index("active_power")
        last = t.x_scaler.inverse(x[:, -1, :])[:, ch]
        np.testing.assert_allclose(last, t.persistence("test"), rtol=1e-4, atol=1e-2)
        assert x.shape == (len(te), t.lookback, len(t.window_columns)) and x.dtype == np.float32
        np.testing.assert_allclose(t.unscale_y(y), t.y("test"), rtol=1e-4, atol=1e-2)


def test_scalers_fit_on_train_only(tmp_cfg, hourly):
    t = ForecastTask(hourly, tmp_cfg, "B", 1)
    train_max = hourly.loc[t.labels == "train", "active_power"].max()
    assert np.isclose(t.y_scaler.hi[0], train_max)


def test_tabular_features_use_only_past(tmp_cfg, hourly):
    h = 24
    f = tabular_features(hourly, h, [1, 2, 24], ["active_power"], [6], ["temperature"])
    T = hourly.index[1000]
    assert f.loc[T, "active_power_o0"] == hourly["active_power"].iloc[1000 - h]
    assert f.loc[T, "active_power_o23"] == hourly["active_power"].iloc[1000 - h - 23]
    assert np.isclose(f.loc[T, "active_power_mean6"], hourly["active_power"].iloc[1000 - h - 5:1000 - h + 1].mean())
    assert f.loc[T, "temperature_o0"] == hourly["temperature"].iloc[1000 - h]
    assert lag_offsets([1, 2, 3, 6, 12, 24]) == [0, 1, 2, 5, 11, 23]


def test_protocol_a_random_split(tmp_cfg, hourly):
    t = ForecastTask(hourly, tmp_cfg, "A")
    n_tr, n_te = len(t.positions("train")), len(t.positions("test"))
    assert abs(n_te / (n_tr + n_te) - 0.25) < 0.01
    assert not set(t.positions("train")) & set(t.positions("test"))
    assert t.is_clean("train").all() and t.is_clean("test").all()
    x, _ = t.windows("train")
    assert x.shape[1:] == (len(t.input_columns), 1)  # paper reshape: features as a 1-D sequence
    # same seed -> same split
    assert np.array_equal(ForecastTask(hourly, tmp_cfg, "A").positions("test"), t.positions("test"))


def test_sliding_windows():
    m = np.arange(20).reshape(10, 2)
    w = sliding_windows(m, np.array([2, 9]), 3)
    assert w.shape == (2, 3, 2) and w[0, -1, 0] == 4 and w[1, 0, 0] == 14
