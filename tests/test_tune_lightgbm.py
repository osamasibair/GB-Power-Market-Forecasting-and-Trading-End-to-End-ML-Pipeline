import itertools
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import optuna
from test_patchtst import make_features

from train import features
from tune_lightgbm import (
    check_features,
    defaults,
    make_model,
    storage_for,
    suggest,
    tuned_forecasts,
    used_features,
)
from validation import before, fold_rows


def test_the_first_trial_is_exactly_the_current_settings():
    settings = suggest(optuna.trial.FixedTrial(defaults))
    assert settings == defaults
    model = make_model(settings)
    assert model.n_estimators == 500 and model.learning_rate == 0.05


def test_the_sunshine_and_christmas_features_can_be_switched_off():
    assert used_features({"solar": 1, "christmas": 1}) == features
    used = used_features({"solar": 1, "christmas": 0})
    assert "is_christmas" not in used and "shortwave_radiation" in used and len(used) == len(features) - 1
    assert len(used_features({"solar": 0, "christmas": 0})) == len(features) - 2


def test_each_choice_of_features_has_its_own_study():
    assert storage_for({"solar": 1, "christmas": 1}).endswith("optuna_lightgbm_2years.db")
    assert storage_for({"solar": 1, "christmas": 0}).endswith("optuna_lightgbm_2years_no_christmas.db")
    assert storage_for({"solar": 0, "christmas": 0}).endswith("optuna_lightgbm_2years_no_solar_no_christmas.db")


def test_the_feature_check_scores_every_combination_by_the_average_of_both_years():
    df = make_features(days=150)
    starts = [df.index[80 * 48], df.index[110 * 48], df.index[-1]]
    years = [(before(df, a), fold_rows(df, a, b)) for a, b in itertools.pairwise(starts)]
    # run in its own process that imports only tune_lightgbm (no PyTorch), for the same macOS reason as below
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as pool:
        results = pool.submit(check_features, {**defaults, "n_estimators": 30}, years, [42, 1]).result()
    assert len(results) == 4
    for r in results:
        assert np.isclose(r["mape"], (r["year1"] + r["year2"]) / 2)


def test_tuned_forecasts_are_calibrated_on_the_calibration_period_and_ranges_contain_the_forecast():
    df = make_features(days=150)
    calibration_start, split_date = df.index[90 * 48], df.index[120 * 48]
    calib = df[(df.index >= calibration_start) & (df.index < split_date)]
    test = df[df.index >= split_date]
    small = {**defaults, "n_estimators": 50}
    # fitted in its own process: on macOS, training LightGBM in a process that has loaded PyTorch (as the other
    # test files do) can crash, because both bring their own OpenMP library
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as pool:
        out = pool.submit(tuned_forecasts, df, calib, test, small, 42, features, calibration_start, split_date).result()
    for part, frame in (("calibration", calib), ("forecasts", test)):
        point, low, high = (out[part][k] for k in ("lightgbm_tuned", "lightgbm_tuned_low", "lightgbm_tuned_high"))
        assert len(point) == len(frame)
        assert (low <= point).all() and (point <= high).all()
    inside = (calib["demand"].to_numpy() >= out["calibration"]["lightgbm_tuned_low"]) & \
             (calib["demand"].to_numpy() <= out["calibration"]["lightgbm_tuned_high"])
    assert inside.mean() >= 0.8
    point = out["forecasts"]["lightgbm_tuned"]
    assert np.mean(np.abs(point - test["demand"]) / test["demand"]) < 0.05