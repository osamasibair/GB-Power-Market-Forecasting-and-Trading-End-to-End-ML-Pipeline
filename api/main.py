"""HTTP API for day ahead electricity demand forecasting."""

import sys
from datetime import date
from pathlib import Path

from fastapi import FastAPI, HTTPException

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from db import load_table
from evaluate import metrics
from predict import load_model, load_quantile_models, predict_day
from train import split_date

app = FastAPI(title="Demand Forecast")
model = load_model()
df = load_table("features")
quantile = load_quantile_models()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/predict", responses={404: {"description": "No data for that date"}})
def predict(day: date):
    try:
        result = predict_day(model, df, day, quantile)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error))
    m = metrics(result["actual"], result["forecast"])
    return {
        "date": day,
        "in_training_data": day < date.fromisoformat(split_date),
        "mae": round(m["mae"]),
        "mape": round(m["mape"], 2),
        "interval": "80%",
        "forecast": [
            {"timestamp_utc": ts.isoformat(), "low": round(l), "forecast": round(f), "high": round(h), "actual": round(a)}
            for ts, f, a, l, h in zip(result.index, result["forecast"], result["actual"], result["low"], result["high"])
        ],
    }

@app.get("/forecast/latest", responses={404: {"description": "No live forecasts yet"}})
def latest_forecast():
    try:
        live = load_table("live_forecasts")
    except ValueError:
        raise HTTPException(status_code=404, detail="no live forecasts yet")
    local_day = live.index.tz_localize("UTC").tz_convert("Europe/London").date
    day = local_day.max()
    result = live[local_day == day]
    return {
        "date": day,
        "interval": "80%",
        "forecast": [
            {"timestamp_utc": ts.isoformat(), "low": round(low), "forecast": round(forecast), "high": round(high)}
            for ts, forecast, low, high in zip(result.index, result["forecast"], result["low"], result["high"])
        ],
    }