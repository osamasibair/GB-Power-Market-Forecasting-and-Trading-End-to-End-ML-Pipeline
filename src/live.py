"""Forecast tomorrows demand from the latest data, and score earlier forecasts once the actual demand is known."""
import pandas as pd
from anthropic import APIError

from briefing import write_briefing
from db import load_table, save_table
from features import build_features
from gb_weather import gb_average
from ingest import fetch_demand, fetch_weather_forecast, fetch_weather_points
from monitor import daily_report, live_adjustment
from predict import load_model, load_quantile_models, predict_day

history_days = 21


def latest_demand(start, end):
    raw = fetch_demand(start, end)
    raw["startTime"] = pd.to_datetime(raw["startTime"]).dt.tz_convert("UTC").dt.tz_localize(None)
    demand = raw.drop_duplicates("startTime", keep="last").set_index("startTime")["initialDemandOutturn"].sort_index()
    full = pd.date_range(demand.index.min(), demand.index.max(), freq="30min")
    return demand.reindex(full).interpolate(method="time").rename("demand")


def latest_weather():
    """GB-wide weather for the past weeks and the next days, made the same way as the training data's."""
    raw = fetch_weather_points(fetch_weather_forecast, past_days=history_days, forecast_days=3)
    raw["time"] = pd.to_datetime(raw["time"])
    return gb_average(raw).resample("30min").interpolate(method="time")


if __name__ == "__main__":
    today = pd.Timestamp.now(tz="Europe/London").normalize()
    tomorrow = (today + pd.Timedelta(days=1)).date()
    demand = latest_demand(f"{today - pd.Timedelta(days=history_days):%Y-%m-%d}", f"{today:%Y-%m-%d}")
    df = build_features(latest_weather().join(demand))

    try:
        earlier = load_table("live_forecasts")
        earlier["actual"] = demand.reindex(earlier.index).fillna(earlier["actual"])  
        extra = live_adjustment(earlier)  
    except ValueError:
        earlier, extra = None, 0.0  

    quantile = load_quantile_models()  
    quantile = {**quantile, "adjustment": quantile["adjustment"] + extra}  
    forecast = predict_day(load_model(), df, tomorrow, quantile)  
    if len(forecast) < 46:
        raise SystemExit(f"only {len(forecast)} half-hours of {tomorrow} could be forecast, so some recent data is missing")
    forecast["baseline"] = df.loc[forecast.index, "demand_lag_336"]
    forecast["extra"] = extra  

    if earlier is None:  
        live = forecast 
    else: 
        live = pd.concat([earlier[~earlier.index.isin(forecast.index)], forecast]).sort_index()
    live["actual"] = demand.reindex(live.index).fillna(live["actual"])
    save_table(live, "live_forecasts")
    print(f"forecast {tomorrow}: {forecast['forecast'].min():.0f} to {forecast['forecast'].max():.0f} MW, intervals widened by {extra:.0f} MW from recent live errors") 

    scored = live.dropna(subset=["actual"])
    daily = daily_report(scored, scored["baseline"])
    if len(daily):
        save_table(daily, "monitoring_live")
        print(f"scored {len(daily)} live days: mean MAPE {daily['mape'].mean():.2f}% (baseline {daily['baseline_mape'].mean():.2f}%), mean coverage {daily['coverage'].mean():.1f}%")
    else:
        print("no complete days to score yet")

    try:
        briefing = write_briefing(forecast, daily, extra)
    except APIError as error:
        briefing = None
        print(f"briefing skipped: {error}")
    if briefing is None:
        print("no briefing written")
    else:
        save_table(briefing, "live_briefing")
        print(f"briefing: {briefing['briefing'].iloc[0]}")