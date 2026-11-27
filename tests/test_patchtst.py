import numpy as np
import pandas as pd
import pytest
import torch

from evaluate import metrics
from features import build_features
from forecast_windows import (
    arrays,
    covariate_columns,
    covariate_matrix,
    day_windows,
    gap_steps,
    history_steps,
    horizon,
    longest_history,
    windows,
)
from patchtst import PatchTST, PatchTSTForecaster, RevIN, chronos_forecast, same_time

torch.set_num_threads(1)

def make_features(start="2024-01-01", days=120, seed=0):
    rng = np.random.default_rng(seed)
    index = pd.date_range(start, periods=days * 48, freq="30min")
    step = np.arange(len(index))
    temperature = 10 - 6 * np.cos(2 * np.pi * index.dayofyear / 365.25) + rng.normal(0, 2, len(index))
    daily = np.sin(2 * np.pi * (step % 48 - 14) / 48)
    demand = 28000 + 5000 * daily - 250 * temperature - 2500 * (index.dayofweek >= 5) + rng.normal(0, 300, len(index))
    raw = pd.DataFrame({"demand": demand, "temperature_2m": temperature, "wind_speed_10m": 5.0,
                        "shortwave_radiation": np.clip(daily, 0, None) * 300, "relative_humidity_2m": 75.0,
                        "cloud_cover": 50.0}, index=index)
    return build_features(raw)


def test_windows_need_a_full_week_and_the_gap_before_them():
    df = make_features(days=20)
    starts = windows(df.index)
    assert starts.min() == history_steps + gap_steps
    assert starts.max() == len(df) - horizon


def test_windows_skip_gaps_in_the_data():
    df = make_features(days=20)
    gappy = df.drop(df.index[400])
    whole = windows(df.index)
    assert len(windows(gappy.index)) < len(whole)


def test_day_windows_start_at_uk_midnight_and_skip_clock_change_days():
    df = make_features("2024-03-01", days=60)
    starts = day_windows(df.index)
    local = df.index[starts].tz_localize("UTC").tz_convert("Europe/London")
    assert ((local.hour == 0) & (local.minute == 0)).all()
    assert pd.Timestamp("2024-03-31").date() not in set(local.date)
    assert pd.Timestamp("2024-04-10").date() in set(local.date)


def test_history_ends_16_hours_before_the_forecast_day():
    df = make_features(days=20)
    start = day_windows(df.index)[:1]
    history, future, actual = arrays(df, start)
    assert history.shape == (1, history_steps) and future.shape[:2] == (1, horizon)
    assert history[0, -1] == np.float32(df["demand"].iloc[start[0] - gap_steps - 1])
    assert np.allclose(actual[0], df["demand"].iloc[start[0]:start[0] + horizon])


def test_revin_round_trip_returns_the_original_values():
    revin = RevIN()
    x = torch.randn(4, 336) * 3000 + 30000
    normalised, mean, std = revin.normalise(x)
    assert torch.allclose(normalised.mean(dim=1), torch.zeros(4), atol=1e-4)
    restored = revin.denormalise(normalised[:, :48, None], mean, std)[..., 0]
    assert torch.allclose(restored, x[:, :48], rtol=1e-4)


def test_quantiles_never_cross():
    torch.manual_seed(0)
    net = PatchTST(n_covariates=11)
    out = net(torch.randn(8, history_steps) * 3000 + 30000, torch.randn(8, horizon, 11))
    assert out.shape == (8, horizon, 3)
    assert (out[..., 0] <= out[..., 1]).all() and (out[..., 1] <= out[..., 2]).all()

@pytest.mark.parametrize("pooling", ["flatten", "mean", "attention"])
def test_every_pooling_gives_forecasts_whose_quantiles_never_cross(pooling):
    torch.manual_seed(0)
    net = PatchTST(n_covariates=11, pooling=pooling)
    out = net(torch.randn(8, history_steps) * 3000 + 30000, torch.randn(8, horizon, 11))
    assert out.shape == (8, horizon, 3)
    assert (out[..., 0] <= out[..., 1]).all() and (out[..., 1] <= out[..., 2]).all()


def test_attention_pooling_needs_far_fewer_parameters_than_flatten():
    def count(net):
        return sum(p.numel() for p in net.parameters())

    assert count(PatchTST(11, pooling="attention")) < count(PatchTST(11, pooling="flatten")) / 2


def test_each_forecast_half_hour_spreads_its_attention_over_the_27_patches():
    net = PatchTST(11).eval()
    net.keep_attention = True
    net(torch.randn(2, history_steps) * 3000 + 30000, torch.randn(2, horizon, 11))
    assert net.attention.shape == (2, horizon, 27)
    assert torch.allclose(net.attention.sum(dim=-1), torch.ones(2, horizon))


def test_unknown_pooling_is_rejected():
    with pytest.raises(ValueError):
        PatchTST(11, pooling="max")


def test_models_saved_before_pooling_existed_load_with_the_flatten_head(tmp_path):
    df = make_features(days=30)
    test = day_windows(df.index)[-2:]
    model = PatchTSTForecaster(max_epochs=1, d_model=16, hidden=32, pooling="flatten").fit(df, windows(df.index, stride=16))
    model.save(tmp_path / "old.pt")
    saved = torch.load(tmp_path / "old.pt")
    del saved["settings"]["pooling"]
    torch.save(saved, tmp_path / "old.pt")
    loaded, _ = PatchTSTForecaster.load(tmp_path / "old.pt")
    assert loaded.settings["pooling"] == "flatten"
    assert np.allclose(model.predict(df, test), loaded.predict(df, test))


def test_same_time_lags_are_the_demand_2_and_7_days_before_each_forecast_half_hour():
    df = make_features(days=20)
    start = day_windows(df.index)[:1]
    history, _, _ = arrays(df, start)
    lags = same_time(torch.from_numpy(history))
    demand = df["demand"].to_numpy(dtype=np.float32)
    assert lags.shape == (1, horizon, 2)
    assert np.allclose(lags[0, :, 0], demand[start[0] - 96:start[0] - 96 + horizon])
    assert np.allclose(lags[0, :, 1], demand[start[0] - 336:start[0] - 336 + horizon])


def test_lags_add_two_inputs_per_half_hour_and_keep_quantiles_in_order():
    torch.manual_seed(0)
    without, with_lags = PatchTST(11), PatchTST(11, lags=1)
    assert with_lags.covariate[0].in_features == without.covariate[0].in_features + 2
    out = with_lags(torch.randn(4, history_steps) * 3000 + 30000, torch.randn(4, horizon, 11))
    assert out.shape == (4, horizon, 3)
    assert (out[..., 0] <= out[..., 1]).all() and (out[..., 1] <= out[..., 2]).all()


def test_lags_must_be_0_or_1():
    with pytest.raises(ValueError):
        PatchTST(11, lags=2)


def test_without_revin_every_week_uses_the_same_fixed_scale():
    net = PatchTST(11, revin=0)
    net.scale.copy_(torch.tensor([30000.0, 3000.0]))
    history = torch.randn(2, history_steps) * 3000 + 30000
    x, _, _ = net.normalise(history)
    assert torch.allclose(x, (history - 30000) / 3000)
    out = net(history, torch.randn(2, horizon, 11))
    assert (out[..., 0] <= out[..., 1]).all() and (out[..., 1] <= out[..., 2]).all()
    assert "scale" in net.state_dict() and "scale" not in PatchTST(11).state_dict()


def test_without_revin_the_scale_comes_from_the_training_data_and_is_saved(tmp_path):
    df = make_features(days=30)
    train, test = windows(df.index, stride=16), day_windows(df.index)[-2:]
    model = PatchTSTForecaster(max_epochs=1, d_model=16, hidden=32, revin=0).fit(df, train)
    history, _, _ = arrays(df, train)
    assert np.isclose(model.net.scale[0].item(), history.mean(), rtol=1e-4)
    model.save(tmp_path / "fixed.pt")
    loaded, _ = PatchTSTForecaster.load(tmp_path / "fixed.pt")
    assert np.allclose(model.predict(df, test), loaded.predict(df, test))


def test_without_weather_only_the_calendar_inputs_are_used():
    df = make_features(days=30)
    model = PatchTSTForecaster(max_epochs=1, d_model=16, hidden=32, weather=0).fit(df, windows(df.index, stride=16))
    assert len(model.cov_mean) == 5 + 2 * 3 - 3
    assert model.predict(df, day_windows(df.index)[-2:]).shape == (2, horizon, 3)


def test_revin_must_be_0_or_1():
    with pytest.raises(ValueError):
        PatchTST(11, revin=2)


def test_learns_the_daily_shape_after_a_short_training_run():
    df = make_features(days=150)
    train, test = windows(df.index, last="2024-05-01", stride=4), day_windows(df.index, first="2024-05-01")
    model = PatchTSTForecaster(max_epochs=8, d_model=32, hidden=64).fit(df, train)
    predicted = model.predict(df, test)
    actual = df["demand"].to_numpy()[test[:, None] + np.arange(horizon)]
    flat = np.full_like(actual, df["demand"].iloc[:train.max()].mean())
    assert predicted.shape == (len(test), horizon, 3)
    assert metrics(actual.ravel(), predicted[..., 1].ravel())["mape"] < metrics(actual.ravel(), flat.ravel())["mape"] / 2


def test_early_stopping_records_the_best_epoch():
    df = make_features(days=60)
    valid = day_windows(df.index, first="2024-02-20")
    model = PatchTSTForecaster(max_epochs=4, patience=2, d_model=16, hidden=32)
    model.fit(df, windows(df.index, last="2024-02-20", stride=8), valid=valid)
    assert 1 <= model.best_epochs <= len(model.valid_losses) <= 4


def test_same_seed_gives_the_same_forecast():
    df = make_features(days=30)
    train, test = windows(df.index, stride=16), day_windows(df.index)[-2:]
    first = PatchTSTForecaster(max_epochs=2, d_model=16, hidden=32).fit(df, train).predict(df, test)
    second = PatchTSTForecaster(max_epochs=2, d_model=16, hidden=32).fit(df, train).predict(df, test)
    assert np.allclose(first, second)


def test_saved_model_gives_the_same_forecast_after_loading(tmp_path):
    df = make_features(days=30)
    test = day_windows(df.index)[-2:]
    model = PatchTSTForecaster(max_epochs=1, d_model=16, hidden=32).fit(df, windows(df.index, stride=16))
    model.save(tmp_path / "patchtst.pt", adjustment=123.0)
    loaded, saved = PatchTSTForecaster.load(tmp_path / "patchtst.pt")
    assert saved["adjustment"] == 123.0
    assert np.allclose(model.predict(df, test), loaded.predict(df, test))


def test_unknown_settings_are_rejected():
    with pytest.raises(ValueError):
        PatchTSTForecaster(layer=3)


def test_covariate_columns_without_weather_keep_only_the_calendar():
    from forecast_windows import names

    def kept(columns):
        return [names[i] for i in columns]

    assert kept(covariate_columns(True)) == names and len(names) == 10 + 6
    assert kept(covariate_columns(False)) == ["is_holiday", "is_christmas"] + names[-6:]
    assert "shortwave_radiation" not in kept(covariate_columns(True, solar=False))
    assert "is_christmas" not in kept(covariate_columns(True, christmas=False))
    assert len(covariate_columns(True, solar=False, christmas=False)) == 14


def test_covariates_include_sin_and_cos_of_the_calendar():
    df = make_features(days=10)
    assert covariate_matrix(df).shape == (len(df), 10 + 2 * 3)


def test_chronos_forecast_uses_only_history_before_the_cutoff_and_keeps_the_forecast_day():
    class FakePipeline:
        def predict_quantiles(self, inputs, prediction_length, quantile_levels):
            self.last_values = [float(x[-1]) for x in inputs]
            values = torch.arange(prediction_length, dtype=torch.float32)
            return values[None, :, None].repeat(len(inputs), 1, len(quantile_levels)), None

    df = make_features(days=20)
    starts = day_windows(df.index)[:3]
    pipeline = FakePipeline()
    out = chronos_forecast(df, starts, pipeline, context=500)
    assert out.shape == (3, horizon, 3)
    assert out[0, 0, 0] == gap_steps
    assert pipeline.last_values[-1] == pytest.approx(df["demand"].iloc[starts[-1] - gap_steps - 1])

def test_forecast_table_keeps_every_model_next_to_the_actual_demand():
    from train_patchtst import forecast_table
    index = pd.date_range("2025-09-01", periods=4, freq="30min")
    table = forecast_table(index, np.array([1.0, 2, 3, 4]), {"lightgbm": np.zeros(4), "patchtst": np.ones(4)})
    assert list(table.columns) == ["actual", "lightgbm", "patchtst"]
    assert table.index.equals(index) and table["actual"].iloc[-1] == 4


def test_retraining_patchtst_keeps_the_other_models_columns(monkeypatch):
    import train_patchtst

    index = pd.date_range("2025-09-01", periods=4, freq="30min")
    old = pd.DataFrame({"actual": 1.0, "patchtst": 0.0, "ridge": [5.0, 6, 7, 8]}, index=index)
    monkeypatch.setattr(train_patchtst, "load_table", lambda name: old)
    new = train_patchtst.forecast_table(index, np.ones(4), {"patchtst": np.full(4, 2.0)})
    kept = train_patchtst.keep_other_models(new, "transformer_forecasts")
    assert kept["patchtst"].tolist() == [2.0] * 4 and kept["ridge"].tolist() == [5.0, 6, 7, 8]
    other_days = train_patchtst.forecast_table(index + pd.Timedelta("1D"), np.ones(4), {"patchtst": np.full(4, 2.0)})
    assert "ridge" not in train_patchtst.keep_other_models(other_days, "transformer_forecasts")


def test_a_longer_history_reads_more_patches_and_keeps_quantiles_in_order():
    net = PatchTST(11, pooling="attention", lags=1, history_days=14)
    assert net.position.shape[1] == (14 * 48 - 24) // 12 + 1
    out = net(torch.randn(4, 14 * 48) * 3000 + 30000, torch.randn(4, horizon, 11))
    assert out.shape == (4, horizon, 3) and (out[..., 0] <= out[..., 1]).all() and (out[..., 1] <= out[..., 2]).all()


def test_lags_come_from_the_end_of_any_history_length():
    history = torch.arange(2 * 14 * 48, dtype=torch.float32).reshape(2, -1)
    lags = same_time(history)
    end = 14 * 48 + gap_steps
    assert torch.equal(lags[0, :, 0], history[0, end - 96:end - 96 + horizon])
    assert torch.equal(lags[0, :, 1], history[0, end - 336:end - 336 + horizon])


def test_history_days_outside_the_compared_lookbacks_are_rejected():
    with pytest.raises(ValueError, match="history_days"):
        PatchTST(11, history_days=10)


def test_the_forecaster_reads_its_own_history_length():
    df = make_features(days=120)
    train = windows(df.index, stride=16, history=longest_history)
    model = PatchTSTForecaster(max_epochs=1, d_model=16, hidden=32, history_days=21).fit(df, train)
    history, _, _ = model.tensors(df, day_windows(df.index)[-2:])
    assert history.shape == (2, 21 * 48)
    assert model.predict(df, day_windows(df.index)[-2:]).shape == (2, horizon, 3)


def test_windows_for_training_leave_room_for_the_longest_history():
    df = make_features(days=120)
    assert windows(df.index, history=longest_history).min() == longest_history + gap_steps


def test_a_window_without_enough_history_is_refused():
    df = make_features(days=120)
    with pytest.raises(ValueError, match="history"):
        arrays(df, windows(df.index)[:1], history=14 * 48)


def test_models_saved_before_history_days_existed_load_with_a_week(tmp_path):
    df = make_features(days=60)
    model = PatchTSTForecaster(max_epochs=1, d_model=16, hidden=32).fit(df, windows(df.index, stride=16))
    model.settings.pop("history_days")
    model.save(tmp_path / "old.pt")
    loaded, _ = PatchTSTForecaster.load(tmp_path / "old.pt")
    assert loaded.history() == history_steps


def test_without_keep_best_training_runs_every_epoch_and_keeps_the_last():
    df = make_features(days=60)
    valid = day_windows(df.index, first="2024-02-20")
    model = PatchTSTForecaster(epochs=3, patience=1, d_model=16, hidden=32)
    model.fit(df, windows(df.index, last="2024-02-20", stride=8), valid=valid, epochs=3, keep_best=False)
    assert len(model.valid_losses) == 3 and model.best_epochs == 3


def test_the_final_model_averages_its_seeds_and_its_range_never_crosses_the_median(monkeypatch):
    import train_patchtst

    df = make_features("2023-11-01", days=140)
    monkeypatch.setattr(train_patchtst, "first_training_day", "2024-01-10")
    calib, test = day_windows(df.index, "2024-02-20", "2024-03-05"), day_windows(df.index, "2024-03-05", "2024-03-15")

    def actual(days):
        return df["demand"].to_numpy()[days[:, None] + np.arange(48)].ravel()

    settings = {"epochs": 2, "d_model": 16, "hidden": 32, "train_stride": 16}
    _, model, _, losses, forecasts, _, seed_mapes = train_patchtst.patchtst(
        df, calib, test, actual(calib), actual(test), "2024-02-20", "2024-03-05", settings, seeds=(42, 1))
    assert len(seed_mapes) == 2 and model.settings["seed"] == 42 and len(losses) == 2
    assert (forecasts["patchtst_low"] <= forecasts["patchtst"]).all() and (forecasts["patchtst"] <= forecasts["patchtst_high"]).all()


def test_training_runs_where_patchtst_device_says_and_the_model_comes_back_to_the_cpu(monkeypatch):
    monkeypatch.setenv("PATCHTST_DEVICE", "cpu")
    df = make_features(days=40)
    model = PatchTSTForecaster(epochs=1, d_model=16, hidden=32, revin=0).fit(df, windows(df.index, stride=16), epochs=1)
    assert all(p.device.type == "cpu" for p in model.net.parameters()) and model.net.scale.device.type == "cpu"