"""Fetching raw demand and weather data from public APIs."""

import os
import time
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

raw = Path("data/raw")
elexon = "https://data.elexon.co.uk/bmrs/api/v1/demand/outturn"
weather = "https://historical-forecast-api.open-meteo.com/v1/forecast"
weather_forecast = "https://api.open-meteo.com/v1/forecast"
market_index = "https://data.elexon.co.uk/bmrs/api/v1/balancing/pricing/market-index"
wind_solar = "https://data.elexon.co.uk/bmrs/api/v1/generation/actual/per-type/wind-and-solar"
retry = Retry(total=5, backoff_factor=10, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET"], raise_on_status=False)
session = requests.Session()
session.mount("https://", HTTPAdapter(max_retries=retry))

def fetch_demand(start, end):
    frames = []
    for period_start in pd.date_range(start, end, freq="7D"):
        period_end = min(period_start + pd.offsets.Day(6), pd.Timestamp(end))
        dates = {
            "settlementDateFrom": period_start.strftime("%Y-%m-%d"),
            "settlementDateTo": period_end.strftime("%Y-%m-%d"),
                }
        response = session.get(elexon, params=dates, timeout=30)
        response.raise_for_status()
        frames.append(pd.DataFrame(response.json()["data"]))
        print(f"demand: {period_start:%Y-%m-%d} to {period_end:%Y-%m-%d}")
        time.sleep(0.5)
    return pd.concat(frames, ignore_index=True)

def fetch_weather(start, end, lat=51.5072, lon=-0.1276):
    query = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": "temperature_2m,wind_speed_10m,shortwave_radiation",
        "timezone": "UTC",
    }
    response = session.get(weather, params=query, timeout=60)
    response.raise_for_status()
    return pd.DataFrame(response.json()["hourly"])

def fetch_weather_forecast(past_days, forecast_days, lat=51.5072, lon=-0.1276):
    query = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,wind_speed_10m,shortwave_radiation",
        "timezone": "UTC",
        "past_days": past_days,
        "forecast_days": forecast_days,
    }
    response = session.get(weather_forecast, params=query, timeout=60)
    response.raise_for_status()
    return pd.DataFrame(response.json()["hourly"])

def fetch_weekly(url, start, end, label, extra_params=None):
    frames = []
    for period_start in pd.date_range(start, end, freq="7D"):
        period_end = min(period_start + pd.offsets.Day(7) - pd.offsets.Minute(30), pd.Timestamp(end))
        query = {
            "from": period_start.strftime("%Y-%m-%dT%H:%MZ"),
            "to": period_end.strftime("%Y-%m-%dT%H:%MZ"),
        }
        if extra_params:
            query.update(extra_params)
        response = session.get(url, params=query, timeout=30)
        response.raise_for_status()
        frames.append(pd.DataFrame(response.json()["data"]))
        print(f"{label}: {period_start:%Y-%m-%d} to {period_end:%Y-%m-%d}")
        time.sleep(0.5)
    return pd.concat(frames, ignore_index=True)

def fetch_prices(start, end):
    return fetch_weekly(market_index, start, end, "prices", {"dataProviders": "APXMIDP"})

def fetch_generation(start, end):
    return fetch_weekly(wind_solar, start, end, "generation")

if __name__ == "__main__":
    start, end = "2023-01-01", os.environ.get("DATA_END", "2026-09-01")
    raw.mkdir(parents=True, exist_ok=True)
    if not (raw / "generation.csv").exists():
        fetch_generation(start, end).to_csv(raw / "generation.csv", index=False)
    if not (raw / "demand.csv").exists(): fetch_demand(start, end).to_csv(raw / "demand.csv", index=False)
    if not (raw / "weather.csv").exists(): fetch_weather(start, end).to_csv(raw / "weather.csv", index=False)
    if not (raw / "prices.csv").exists(): fetch_prices(start, end).to_csv(raw / "prices.csv", index=False)
    print("saved to data/raw")





