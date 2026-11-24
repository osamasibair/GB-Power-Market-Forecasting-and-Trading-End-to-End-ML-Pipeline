"""Forecast windows: which past week of demand each forecast sees, and the weather and calendar inputs for the day it forecasts. doesnt import pytorch, so scripts that also train LightGBM can use it without loading both libraries in one
process. on macos their threading libraries clash and the process freezes or crashes."""
import numpy as np
import pandas as pd

history_steps = 336
gap_steps = 32
horizon = 48
covariates = ["temperature_2m", "shortwave_radiation", "hdd", "is_holiday", "is_christmas"]
weather = ["temperature_2m", "shortwave_radiation", "hdd"]
cycles = {"period": 48, "dayofweek": 7, "dayofyear": 365.25}


def covariate_matrix(df):
    columns = [df[name].to_numpy(dtype=np.float32) for name in covariates]
    for name, length in cycles.items():
        angle = 2 * np.pi * df[name].to_numpy(dtype=np.float32) / length
        columns += [np.sin(angle), np.cos(angle)]
    return np.stack(columns, axis=1).astype(np.float32)


def covariate_columns(use_weather=True, solar=True, christmas=True):
    """Which columns of covariate_matrix to use: all of them, or without tomorrows weather (calendar only), without
    the sunshine, or without the Christmas flag."""
    names = covariates + [f"{kind}_{name}" for name in cycles for kind in ("sin", "cos")]
    dropped = set() if use_weather else set(weather)
    if not solar:
        dropped.add("shortwave_radiation")
    if not christmas:
        dropped.add("is_christmas")
    return np.array([i for i, name in enumerate(names) if name not in dropped])


def windows(index, first=None, last=None, stride=1):
    """Positions where a forecast window starts: a full week of history, then the gap, then 48 half hours in [first, last)."""
    starts = np.arange(gap_steps + history_steps, len(index) - horizon + 1, stride)
    span = index[starts + horizon - 1] - index[starts - gap_steps - history_steps]
    starts = starts[span == pd.Timedelta(minutes=30) * (history_steps + gap_steps + horizon - 1)]
    if first is not None:
        starts = starts[index[starts] >= pd.Timestamp(first)]
    if last is not None:
        starts = starts[index[starts + horizon - 1] < pd.Timestamp(last)]
    return starts


def day_windows(index, first=None, last=None):
    """Windows that start at UK midnight on days with 48 half hours, the forecasts the live job makes (clock change days are skipped)."""
    local = index.tz_localize("UTC").tz_convert("Europe/London")
    dates = pd.Series(local.date)
    full_day = (dates.map(dates.value_counts()) == horizon).to_numpy()
    midnight = (local.hour == 0) & (local.minute == 0) & full_day
    starts = windows(index, first, last)
    return starts[midnight[starts]]


def arrays(df, starts):
    demand = df["demand"].to_numpy(dtype=np.float32)
    ahead = starts[:, None] + np.arange(horizon)
    history = demand[starts[:, None] + np.arange(-gap_steps - history_steps, -gap_steps)]
    return history, covariate_matrix(df)[ahead], demand[ahead]