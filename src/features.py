"""Build model features from the processed dataset."""

import holidays
import pandas as pd

from db import load_table, save_table

uk_holidays = holidays.UnitedKingdom(subdiv="ENG", years=range(2022, 2028))

def build_features(df):
    df = df.copy()
    local = df.index.tz_localize("UTC").tz_convert("Europe/London")
    df["period"] = local.hour * 2 + (local.minute // 30) + 1
    df["dayofweek"] = local.dayofweek
    df["dayofyear"] = local.dayofyear
    df["is_holiday"] = pd.Series(local.date, index=df.index).isin(uk_holidays).astype(int)
    df["is_christmas"] = (((local.month == 12) & (local.day >= 24)) | ((local.month == 1) & (local.day == 1))).astype(int)
    df["hdd"] = (15.5 - df["temperature_2m"]).clip(lower=0)
    df["demand_lag_recent"] = df["demand"].shift(48).where(local.hour < 8, df["demand"].shift(96))
    df["demand_lag_336"] = df["demand"].shift(336)
    df["demand_roll_96"] = df["demand"].shift(96).rolling(48).mean()
    # recent temperature: buildings warm up and cool down slowly, so demand follows the last day or days of temperature
    # as well as the current one (the "recency effect"). Weather comes from forecasts, known ahead like temperature_2m
    df["temperature_24h"] = df["temperature_2m"].rolling(48).mean()
    df["temperature_72h"] = df["temperature_2m"].rolling(144).mean()
    return df.dropna(subset=df.columns.drop("demand"))

if __name__ == "__main__":
    df = load_table("dataset")
    out = build_features(df)
    pd.set_option("display.max_columns", None)
    print(f"{len(out)} rows, {out.shape[1]} columns")
    print(out.head())
    save_table(out, "features")

