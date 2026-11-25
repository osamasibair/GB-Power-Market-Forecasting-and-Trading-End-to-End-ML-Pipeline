"""Create a synthetic dataset and models so the API tests can run in CI."""

import os
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from db import save_table
from features import build_features
from train import features

if os.environ.get("CI") != "true":
    sys.exit("This overwrites the models and features table with fake ones, so it only runs in CI.")

index = pd.date_range("2026-07-01", "2026-08-31 23:30", freq="30min", name="timestamp")
hours = index.hour + index.minute / 60
noise = np.random.default_rng(0).normal(0, 500, len(index))
raw = pd.DataFrame({
    "demand": 30000 + 5000 * np.sin((hours - 6) / 24 * 2 * np.pi) + noise,
    "temperature_2m": 15 + 5 * np.sin((hours - 9) / 24 * 2 * np.pi),
    "wind_speed_10m": 5.0,
    "shortwave_radiation": np.clip(600 * np.sin((hours - 6) / 12 * np.pi), 0, None),
    "relative_humidity_2m": 75.0,
    "cloud_cover": 50.0,
}, index=index)
df = build_features(raw)
save_table(df, "features")


def fit(**params):
    return lgb.LGBMRegressor(n_estimators=20, verbose=-1, **params).fit(df[features], df["demand"])


Path("models").mkdir(exist_ok=True)
joblib.dump(fit(), "models/model.joblib")
joblib.dump({"models": {q: fit(objective="quantile", alpha=q) for q in (0.1, 0.5, 0.9)}, "adjustment": 0.0}, "models/quantile_models.joblib")
print(f"created synthetic features ({len(df)} rows) and models for the API tests")