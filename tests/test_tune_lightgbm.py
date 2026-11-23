import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import optuna
from test_patchtst import make_features

from train import features
from tune_lightgbm import defaults, make_model, suggest, tuned_forecasts


def test_the_first_trial_is_exactly_the_current_settings():
    settings = suggest(optuna.trial.FixedTrial(defaults))
    assert settings == defaults
    model = make_model(settings)
    assert model.n_estimators == 500 and model.learning_rate == 0.05


def test_tuned_forecasts_are_calibrated_on_the_calibration_period_and_ranges_contain_the_forecast():
    df = make_features(days=150)
    assert set(features) <= set(df.columns)
    calibration_start, split_date = df.index[90 * 48], df.index[120 * 48]
    calib = df[(df.index >= calibration_start) & (df.index < split_date)]
    test = df[df.index >= split_date]
    small = {**defaults, "n_estimators": 50}
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as pool:
        out = pool.submit(tuned_forecasts, df, calib, test, small, 42, calibration_start, split_date).result()
    for part, frame in (("calibration", calib), ("forecasts", test)):
        point, low, high = (out[part][k] for k in ("lightgbm_tuned", "lightgbm_tuned_low", "lightgbm_tuned_high"))
        assert len(point) == len(frame)
        assert (low <= point).all() and (point <= high).all()
    inside = (calib["demand"].to_numpy() >= out["calibration"]["lightgbm_tuned_low"]) & \
             (calib["demand"].to_numpy() <= out["calibration"]["lightgbm_tuned_high"])
    assert inside.mean() >= 0.8
    point = out["forecasts"]["lightgbm_tuned"]
    assert np.mean(np.abs(point - test["demand"]) / test["demand"]) < 0.05