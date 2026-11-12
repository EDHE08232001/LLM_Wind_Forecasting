import numpy as np
import pandas as pd

from windfc.splits import ForecastTask, chronological_labels


def test_chronological_labels():
    idx = pd.date_range("2018-08-31 22:00", periods=4, freq="h")
    lab = chronological_labels(idx, "2018-08-31 23:00", "2018-09-01 00:00")
    assert lab.tolist() == ["train", "train", "val", "test"]


def test_splits_do_not_overlap_and_references_align(tmp_cfg, hourly):
    for h in (1, 24):
        t = ForecastTask(hourly, tmp_cfg, h)
        tr, va, te = (t.target_index(s) for s in ("train", "val", "test"))
        assert tr.max() < va.min() and va.max() < te.min()
        assert va.min() == pd.Timestamp("2018-09-01 00:00")
        assert te.min() == pd.Timestamp("2018-11-01 00:00") and len(te) == 61 * 24  # every test hour is scored
        # persistence = the value at the origin T - h; the power curve = same-hour theoretical power
        np.testing.assert_array_equal(t.persistence("test"), hourly["active_power"].shift(h).loc[te].to_numpy())
        np.testing.assert_array_equal(t.power_curve("test"), hourly.loc[te, "theoretical_power"].to_numpy())


def test_scaling_range_comes_from_training_months_only(tmp_cfg, hourly):
    t = ForecastTask(hourly, tmp_cfg, 1)
    train = hourly.loc[t.labels == "train", "active_power"]
    assert np.isclose(t.target_range, train.max() - train.min())
