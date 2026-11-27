import itertools
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import optuna
import pandas as pd
import pytest
from test_patchtst import make_features

from forecast_windows import known_inputs
from tune_boosting import (
    add_lags,
    columns_for,
    compare,
    confirm,
    defaults,
    lag_columns,
    make_model,
    seeds_for,
    storage_for,
    suggest,
    tuned_forecasts,
)
from validation import before, fold_rows

# LightGBM and XGBoost are fitted in their own process that imports only tune_boosting (no PyTorch): on macOS, training
# them in a process that has loaded PyTorch (as the other test files do) can crash, because each brings its own OpenMP


def in_own_process(function, *args):
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as pool:
        return pool.submit(function, *args).result()


def two_years(df):
    starts = [df.index[80 * 48], df.index[110 * 48], df.index[-1]]
    return [(before(df, a), fold_rows(df, a, b)) for a, b in itertools.pairwise(starts)]


@pytest.mark.parametrize("name", ["lightgbm", "xgboost"])
def test_the_first_trial_is_exactly_the_default_settings(name):
    assert suggest(name, optuna.trial.FixedTrial(defaults[name])) == defaults[name]
    model = make_model(name, defaults[name], 0.5)
    assert model.get_params()["n_estimators"] == 500 and model.get_params()["learning_rate"] == 0.05


def test_both_models_are_quantile_models():
    assert make_model("lightgbm", defaults["lightgbm"], 0.9).get_params()["alpha"] == 0.9
    assert make_model("xgboost", defaults["xgboost"], 0.9).get_params()["quantile_alpha"] == 0.9


def test_default_settings_train_once_and_settings_with_sampling_train_five_seeds():
    assert len(seeds_for(defaults["lightgbm"])) == 1 and len(seeds_for(defaults["xgboost"])) == 1
    assert len(seeds_for({**defaults["lightgbm"], "subsample": 0.8})) == 5


def test_longer_lookbacks_add_the_same_time_each_earlier_week():
    assert lag_columns(7) == ["demand_lag_recent", "demand_lag_336", "demand_roll_96"]
    assert lag_columns(21)[3:] == ["demand_lag_14d", "demand_lag_21d"]
    assert lag_columns(42)[-1] == "demand_lag_42d" and len(lag_columns(42)) == 3 + 5
    with pytest.raises(ValueError):
        lag_columns(10)


def test_lags_are_matched_by_time_so_a_gap_gives_a_missing_value_not_the_wrong_day():
    df = make_features(days=80)
    out = add_lags(df)
    t = out.index[-1]
    assert out.loc[t, "demand_lag_14d"] == df.loc[t - pd.Timedelta(days=14), "demand"]
    gappy = add_lags(df.drop(df.index[1000:1010]))
    after_gap = df.index[1005] + pd.Timedelta(days=14)
    assert np.isnan(gappy.loc[after_gap, "demand_lag_14d"])


def test_both_models_get_every_weather_and_calendar_input_plus_their_lags():
    for days in (7, 42):
        assert columns_for(days) == known_inputs + lag_columns(days)
    for name in ("shortwave_radiation", "is_christmas", "temperature_24h", "temperature_72h", "wind_speed_10m",
                 "relative_humidity_2m", "cloud_cover", "period", "dayofweek", "dayofyear", "is_holiday", "hdd"):
        assert name in known_inputs


def test_each_model_and_lookback_has_its_own_study():
    assert storage_for("lightgbm", 7).endswith("optuna_lightgbm_7d_wide2.db")
    assert storage_for("xgboost", 28).endswith("optuna_xgboost_28d_wide2.db")


@pytest.mark.parametrize("name", ["lightgbm", "xgboost"])
def test_options_are_ranked_by_the_pinball_loss_averaged_over_both_years(name):
    df = add_lags(make_features("2023-11-01", days=150))
    small = {**defaults[name], "n_estimators": 30}
    options = {"7 days": columns_for(7), "14 days": columns_for(14)}
    results = in_own_process(compare, name, small, options, two_years(df))
    assert [r["loss"] for r in results] == sorted(r["loss"] for r in results)
    for r in results:
        assert np.isclose(r["loss"], (r["loss_year1"] + r["loss_year2"]) / 2) and r["loss_std"] == 0


@pytest.mark.parametrize("name", ["lightgbm", "xgboost"])
def test_tuned_forecasts_average_the_seeds_and_the_range_contains_the_forecast(name):
    df = add_lags(make_features("2023-11-01", days=150))
    calibration_start, split_date = df.index[90 * 48], df.index[120 * 48]
    calib = df[(df.index >= calibration_start) & (df.index < split_date)]
    test = df[df.index >= split_date]
    small = {**defaults[name], "n_estimators": 50, "subsample": 0.8}
    out = in_own_process(tuned_forecasts, name, df, calib, test, small, columns_for(7),
                         calibration_start, split_date, [42, 1])
    key = f"{name}_tuned"
    for part, frame in (("calibration", calib), ("forecasts", test)):
        point, low, high = (out[part][k] for k in (key, f"{key}_low", f"{key}_high"))
        assert len(point) == len(frame) and (low <= point).all() and (point <= high).all()
    inside = (calib["demand"].to_numpy() >= out["calibration"][f"{key}_low"]) & \
             (calib["demand"].to_numpy() <= out["calibration"][f"{key}_high"])
    assert inside.mean() >= 0.8
    assert len(out["seed_mapes"]) == 2
    assert np.mean(np.abs(out["forecasts"][key] - test["demand"]) / test["demand"]) < 0.06


@pytest.mark.parametrize("name", ["lightgbm", "xgboost"])
def test_the_ranges_reach_past_where_the_first_search_found_its_best_settings(name):
    trial = optuna.create_study().ask()
    suggest(name, trial)
    ranges = {key: (d.low, d.high) for key, d in trial.distributions.items()}
    assert ranges["n_estimators"] == (100, 5000) and ranges["colsample_bytree"] == (0.2, 1.0)
    assert ranges["learning_rate"] == ((0.005, 0.5) if name == "lightgbm" else (0.005, 1.0))
    assert ranges["reg_lambda"] == (0.0, 10.0)
    if name == "lightgbm":
        assert ranges["num_leaves"] == (2, 255) and ranges["min_child_samples"] == (5, 1000)
    else:
        assert ranges["max_depth"] == (1, 12) and ranges["min_child_weight"] == (0.01, 100.0)
    assert all(low <= defaults[name][key] <= high for key, (low, high) in ranges.items())


def test_the_best_trials_are_chosen_by_their_loss_averaged_over_every_seed():
    from types import SimpleNamespace

    base = {**defaults["lightgbm"], "subsample": 0.8}
    lucky = SimpleNamespace(number=1, value=280.0, params={**base, "num_leaves": 5})
    steady = SimpleNamespace(number=2, value=281.0, params={**base, "num_leaves": 7})
    later = {5: 290.0, 7: 282.0}
    seen = []

    def scores(name, settings, columns, years, seed):
        seen.append(seed)
        return [{"loss": later[settings["num_leaves"]]}] * 2

    results = confirm("lightgbm", [lucky, steady], [], [], scores)
    assert [r["trial"].number for r in results] == [2, 1] and seen == [1, 2, 3, 4] * 2
    assert results[0]["mean"] == (281.0 + 4 * 282.0) / 5


def test_settings_without_randomness_are_not_trained_again_for_other_seeds():
    from types import SimpleNamespace

    fixed = SimpleNamespace(number=0, value=300.0, params=defaults["xgboost"])
    results = confirm("xgboost", [fixed], [], [], lambda *a: pytest.fail("trained again"))
    assert results[0]["losses"] == [300.0] * 5 and results[0]["std"] == 0