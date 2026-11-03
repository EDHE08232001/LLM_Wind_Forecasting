import numpy as np
import pandas as pd

from windfc.data import compute_clean_flags, direction_from_components, load_scada, load_weather, to_hourly


def test_load_raw_formats(tmp_cfg):
    scada = load_scada(tmp_cfg["paths"]["scada_csv"])
    assert list(scada.columns) == ["active_power", "wind_speed", "theoretical_power", "wind_direction"]
    assert scada.index.is_monotonic_increasing and scada.index.is_unique
    w = load_weather(tmp_cfg["paths"]["weather_csv"])
    assert {"temperature", "air_density", "cloud_cover"} <= set(w.columns)
    assert 1.0 < w["air_density"].median() < 1.4  # kg/m^3, computed from pressure/temperature/humidity
    assert w["cloud_cover"].max() <= 1.0


def test_ninja_format(tmp_path):
    text = "# renewables.ninja\n# units\ntime,local_time,t2m,prectotland,rhoa,swgdn,swtdn,cldtot\n" + "\n".join(
        f"2018-01-01 {h:02d}:00,x,{h},0.1,1.2,{h * 10},{h * 20},0.5" for h in range(24)
    )
    p = tmp_path / "ninja.csv"
    p.write_text(text)
    w = load_weather(p)
    assert len(w) == 24 and w["temperature"].iloc[5] == 5 and w["air_density"].iloc[0] == 1.2


def test_hourly_dataset(hourly):
    assert len(hourly) == 8760
    assert hourly.drop(columns=["merra_ws50m"], errors="ignore").isna().sum().sum() == 0
    assert hourly.index.freq is None or hourly.index.freqstr in ("h", "H")
    assert hourly["imputed"].any()  # synthetic data has gaps that were kNN-imputed
    assert hourly["wind_direction"].between(0, 360).all()


def test_clean_rules():
    df = pd.DataFrame(
        {
            "wind_speed": [2.0, 5.0, 26.0, 8.0, 8.0, 9.0, 10.0, 11.0],
            "active_power": [10.0, 500.0, 0.0, 0.0, -3.0, 900.0, 1000.0, 1100.0],
        }
    )
    out, counts = compute_clean_flags(df, 3.5, 25.5, 1.5)
    assert counts["removed_wind_out_of_range"] == 2
    assert counts["removed_nonpositive_power"] == 2
    assert out["is_clean"].tolist() == [False, True, False, False, False, True, True, True]


def test_iqr_outlier():
    ap = [100.0] * 20 + [101.0] * 20 + [5000.0]
    df = pd.DataFrame({"wind_speed": [8.0] * len(ap), "active_power": ap})
    out, counts = compute_clean_flags(df)
    assert counts["removed_iqr_outliers"] == 1 and not out["is_clean"].iloc[-1]


def test_to_hourly_first_and_mean():
    idx = pd.date_range("2018-01-01", periods=12, freq="10min")
    df = pd.DataFrame({"active_power": np.arange(12.0), "wind_speed": 5.0, "theoretical_power": 1.0,
                       "wind_direction": [350.0, 10.0] * 6}, index=idx)
    first = to_hourly(df, "first")
    assert first["active_power"].tolist() == [0.0, 6.0]
    mean = to_hourly(df, "mean")
    assert mean["active_power"].tolist() == [2.5, 8.5]
    d = mean["wind_direction"].iloc[0]
    assert min(d, 360 - d) < 1e-6  # circular mean of 350 and 10 is 0


def test_direction_roundtrip():
    deg = np.array([0.0, 45.0, 180.0, 359.0])
    r = np.deg2rad(deg)
    np.testing.assert_allclose(direction_from_components(np.sin(r), np.cos(r)), deg, atol=1e-9)
