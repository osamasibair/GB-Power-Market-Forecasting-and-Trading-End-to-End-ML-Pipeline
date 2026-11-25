import numpy as np
import pandas as pd
import torch

from features import build_features
from forecast_windows import day_windows, history_steps, horizon, windows
from patchtst import PatchTSTForecaster
from plot_patchtst import influence, influence_table, pick_week

# see test_patchtst.py: one thread stops PyTorch freezing on macOS when LightGBM ran earlier in the same pytest session
torch.set_num_threads(1)


def make_features(days=40):
    index = pd.date_range("2025-11-01", periods=days * 48, freq="30min")
    daily = np.sin(2 * np.pi * (np.arange(len(index)) % 48 - 14) / 48)
    raw = pd.DataFrame({"demand": 28000 + 5000 * daily, "temperature_2m": 8.0, "wind_speed_10m": 5.0,
                        "shortwave_radiation": np.clip(daily, 0, None) * 300, "relative_humidity_2m": 75.0,
                        "cloud_cover": 50.0}, index=index)
    return build_features(raw)


def test_influence_has_one_row_per_forecast_half_hour_and_one_column_per_past_value():
    df = make_features()
    model = PatchTSTForecaster(max_epochs=1, d_model=16, hidden=32).fit(df, windows(df.index, stride=16))
    grads = influence(df, model, day_windows(df.index)[-3:])
    assert grads.shape == (horizon, history_steps)
    assert (grads >= 0).all() and grads.sum() > 0


def test_pick_week_keeps_only_days_in_that_week():
    df = make_features()
    days = day_windows(df.index)
    start, week = pick_week(df, days, "2025-12-01")
    local = df.index[week].tz_localize("UTC").tz_convert("Europe/London")
    assert len(week) == 7
    assert (local >= start).all() and (local < start + pd.Timedelta(days=7)).all()


def test_pick_week_defaults_to_the_week_with_the_highest_peak():
    df = make_features()
    df.loc[df.index[(df.index >= "2025-12-10") & (df.index < "2025-12-11")], "demand"] += 9000
    start, _ = pick_week(df, day_windows(df.index), None)
    assert start == pd.Timestamp("2025-12-08", tz="Europe/London")

def test_influence_table_has_one_row_per_forecast_and_past_half_hour():
    grads = np.arange(horizon * history_steps, dtype=float).reshape(horizon, history_steps)
    table = influence_table(grads)
    assert len(table) == horizon * history_steps
    assert table["forecast_time"].iloc[0] == "00:00" and table["forecast_time"].iloc[-1] == "23:30"
    assert table["hours_before_midnight"].max() == 184 and table["hours_before_midnight"].min() == 16.5
    assert table["influence"].iloc[-1] == grads[-1, -1]