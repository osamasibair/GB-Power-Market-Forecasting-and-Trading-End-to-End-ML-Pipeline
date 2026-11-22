import numpy as np
from test_patchtst import make_features

from ridge import alphas, choose_alpha, design, ridge_forecasts


def test_the_day_of_the_year_wraps_round_so_new_year_neighbours_are_close():
    df = make_features("2024-12-20", days=16)
    x = design(df)
    end_of_year = x[df["dayofyear"] == 366].iloc[0]  # 31 December 2024, a leap year
    new_year = x[df["dayofyear"] == 1].iloc[0]
    distance = np.hypot(end_of_year["year_sin"] - new_year["year_sin"], end_of_year["year_cos"] - new_year["year_cos"])
    assert distance < 0.05


def test_the_penalty_is_chosen_from_the_grid_using_the_calibration_period_only():
    df = make_features(days=120)
    fit_part, calib = df.iloc[: 80 * 48], df.iloc[80 * 48 : 100 * 48]
    alpha, scores = choose_alpha(fit_part, calib)
    assert alpha in scores and set(scores) == {float(a) for a in alphas}
    assert scores[alpha] == min(scores.values())


def test_forecasts_are_close_and_the_range_is_symmetric_and_calibrated_on_the_calibration_period():
    df = make_features(days=150)
    calibration_start, split_date = df.index[90 * 48], df.index[120 * 48]
    calib = df[(df.index >= calibration_start) & (df.index < split_date)]
    test = df[df.index >= split_date]
    out = ridge_forecasts(df, calib, test, calibration_start, split_date)
    point, low, high = (out["forecasts"][k] for k in ("ridge", "ridge_low", "ridge_high"))
    assert len(point) == len(test)
    assert np.allclose(high - point, point - low)
    errors = np.abs(calib["demand"].to_numpy() - out["calibration"]["ridge"])
    assert np.isclose(high[0] - point[0], np.quantile(errors, 0.8))
    assert np.mean(np.abs(point - test["demand"]) / test["demand"]) < 0.05