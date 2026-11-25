"""Forecast windows: which past week of demand each forecast sees, and the weather and calendar inputs for the day it forecasts. doesnt import pytorch, so scripts that also train LightGBM can use it without loading both libraries in one
process. on macos their threading libraries clash and the process freezes or crashes."""
import numpy as np
import pandas as pd

history_steps = 336
lookbacks = [7, 14, 21, 28, 42]  # the history lengths in days every model chooses from on the validation years
longest_history = max(lookbacks) * 48
first_training_day = "2022-04-20"  # the first midnight with the longest history and the gap behind it: every model trains from here
gap_steps = 32
horizon = 48
# tomorrow's weather and calendar, given to every model with no selection: every physically motivated input, so no model
# chooses the inputs and none is favoured, and each model decides for itself how much to use each one. Read as numbers:
covariates = ["temperature_2m", "shortwave_radiation", "hdd", "is_holiday", "is_christmas", "temperature_24h",
              "temperature_72h", "wind_speed_10m", "relative_humidity_2m", "cloud_cover"]
# read by the neural and foundation models as positions on a circle (sin and cos), and by the tree models as numbers
cycles = {"period": 48, "dayofweek": 7, "dayofyear": 365.25}
known_inputs = covariates + list(cycles)  # every model's weather and calendar inputs
weather = ["temperature_2m", "shortwave_radiation", "hdd", "temperature_24h", "temperature_72h", "wind_speed_10m",
           "relative_humidity_2m", "cloud_cover"]
names = covariates + [f"{kind}_{name}" for name in cycles for kind in ("sin", "cos")]


def covariate_matrix(df):
    columns = [df[name].to_numpy(dtype=np.float32) for name in covariates]
    for name, length in cycles.items():
        angle = 2 * np.pi * df[name].to_numpy(dtype=np.float32) / length
        columns += [np.sin(angle), np.cos(angle)]
    return np.stack(columns, axis=1).astype(np.float32)


def covariate_columns(use_weather=True, solar=True, christmas=True):
    """Which columns of covariate_matrix to use: all of them, or without tomorrows weather (calendar only), without
    the sunshine, or without the Christmas flag."""
    dropped = set() if use_weather else set(weather)
    if not solar:
        dropped.add("shortwave_radiation")
    if not christmas:
        dropped.add("is_christmas")
    return np.array([i for i, name in enumerate(names) if name not in dropped])


def windows(index, first=None, last=None, stride=1, history=history_steps):
    """Positions where a forecast window starts: `history` half hours of history, then the gap, then 48 half hours in [first, last).
    Training uses history=longest_history whatever the model's own history, so every history length trains on the same days."""
    starts = np.arange(gap_steps + history, len(index) - horizon + 1, stride)
    span = index[starts + horizon - 1] - index[starts - gap_steps - history]
    starts = starts[span == pd.Timedelta(minutes=30) * (history + gap_steps + horizon - 1)]
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


def arrays(df, starts, history=history_steps):
    if len(starts) and starts.min() < gap_steps + history:
        raise ValueError(f"a window starts before {history // 48} days of history are available")
    demand = df["demand"].to_numpy(dtype=np.float32)
    ahead = starts[:, None] + np.arange(horizon)
    past = demand[starts[:, None] + np.arange(-gap_steps - history, -gap_steps)]
    return past, covariate_matrix(df)[ahead], demand[ahead]