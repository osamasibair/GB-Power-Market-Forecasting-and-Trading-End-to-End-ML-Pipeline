import numpy as np
import pandas as pd
import pytest

from gb_weather import gb_average, load_weights, regions
from validate import check_weather


def test_the_saved_weights_cover_every_region_and_each_kind_adds_up_to_one():
    weights = load_weights()
    assert list(weights["region"]) == list(regions["region"])
    assert np.isclose(weights["population_weight"].sum(), 1) and np.isclose(weights["solar_weight"].sum(), 1)
    assert (weights[["population_weight", "solar_weight"]] > 0).all().all()


def points(times, values):
    """Weather at two regions, with each column set from `values` = {region: (temperature, wind, sunshine)}, and
    humidity and cloud cover set from the temperature (80 + temperature, 50 + temperature)."""
    rows = [{"time": t, "region": region, "temperature_2m": temp, "wind_speed_10m": wind, "shortwave_radiation": sun,
             "relative_humidity_2m": 80 + temp, "cloud_cover": 50 + temp}
            for t in times for region, (temp, wind, sun) in values.items()]
    return pd.DataFrame(rows)


two = pd.DataFrame({"region": ["A", "B"], "population_weight": [0.75, 0.25], "solar_weight": [0.2, 0.8]})


def test_temperature_and_wind_follow_population_and_sunshine_follows_solar_capacity():
    times = pd.date_range("2024-06-01", periods=3, freq="h")
    out = gb_average(points(times, {"A": (10.0, 4.0, 100.0), "B": (20.0, 8.0, 600.0)}), two)
    assert list(out.index) == list(times)
    assert np.allclose(out["temperature_2m"], 0.75 * 10 + 0.25 * 20)
    assert np.allclose(out["wind_speed_10m"], 0.75 * 4 + 0.25 * 8)
    assert np.allclose(out["shortwave_radiation"], 0.2 * 100 + 0.8 * 600)
    assert np.allclose(out["relative_humidity_2m"], 0.75 * 90 + 0.25 * 100)
    assert np.allclose(out["cloud_cover"], 0.75 * 60 + 0.25 * 70)


def test_a_missing_region_or_hour_is_an_error_not_a_quietly_different_average():
    times = pd.date_range("2024-06-01", periods=3, freq="h")
    only_a = points(times, {"A": (10.0, 4.0, 100.0)})
    with pytest.raises(ValueError, match="missing"):
        gb_average(only_a, two)
    gappy = points(times, {"A": (10.0, 4.0, 100.0), "B": (20.0, 8.0, 600.0)}).drop(index=1)
    with pytest.raises(ValueError, match="missing weather"):
        gb_average(gappy, two)


def test_validation_flags_a_region_with_a_gap_and_a_region_with_no_weather():
    times = pd.date_range("2024-06-01", periods=4, freq="h")
    values = {region: (10.0, 4.0, 100.0) for region in regions["region"]}
    df = points(times, values)
    assert check_weather(df) == []
    gap = df.drop(index=df[(df["region"] == "Wales") & (df["time"] == times[1])].index)
    assert any(p.startswith("Wales: 1 gaps") for p in check_weather(gap))
    no_scotland = df[df["region"] != "Scotland"]
    assert any("no weather for Scotland" in p for p in check_weather(no_scotland))
    old_download = df.drop(columns=["relative_humidity_2m", "cloud_cover"])
    assert any("no relative_humidity_2m, cloud_cover column" in p for p in check_weather(old_download))

def test_every_weather_request_names_the_met_office_model(monkeypatch):
    import ingest

    seen = []

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"hourly": {"time": []}}

    monkeypatch.setattr(ingest.session, "get", lambda url, params, timeout: seen.append(params) or Response())
    ingest.fetch_weather("2022-03-01", "2022-03-02")
    ingest.fetch_weather_forecast(past_days=1, forecast_days=1)
    assert [p["models"] for p in seen] == ["ukmo_seamless", "ukmo_seamless"]


def test_every_weather_request_asks_for_every_variable(monkeypatch):
    import ingest
    from gb_weather import variables

    seen = []

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"hourly": {"time": []}}

    monkeypatch.setattr(ingest.session, "get", lambda url, params, timeout: seen.append(params) or Response())
    ingest.fetch_weather("2022-03-01", "2022-03-02")
    ingest.fetch_weather_forecast(past_days=1, forecast_days=1)
    assert [p["hourly"].split(",") for p in seen] == [variables, variables]
    assert {"relative_humidity_2m", "cloud_cover", "wind_speed_10m"} <= set(variables)