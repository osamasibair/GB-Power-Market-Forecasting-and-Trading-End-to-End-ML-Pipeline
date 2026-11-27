import itertools

import numpy as np
import pandas as pd
import pytest
from test_patchtst import make_features

from quantiles import calibration_start
from train import split_date
from validation import before, fold_rows, folds


def test_the_validation_years_run_back_to_back_and_stop_where_the_test_year_starts():
    assert all(end == next_start for (_, end), (next_start, _) in itertools.pairwise(folds))
    assert folds[-1] == (calibration_start, split_date)
    assert folds[0][1] <= calibration_start


def test_a_validation_year_is_only_forecast_from_data_before_it():
    df = make_features("2023-08-01", days=60)
    start, end = "2023-09-01", "2023-09-15"
    year = fold_rows(df, start, end)
    assert year.index.min() >= pd.Timestamp(start) - pd.Timedelta(hours=1)
    assert year.index.max() < pd.Timestamp(end) + pd.Timedelta(days=1)
    assert len(year) % 48 == 0
    assert before(df, start).index.max() < pd.Timestamp(start)


def test_patchtst_is_scored_on_each_validation_year_by_a_model_trained_before_it(monkeypatch):
    import validation

    df = make_features("2023-11-01", days=140)
    monkeypatch.setattr(validation, "folds", [("2024-02-15", "2024-03-01"), ("2024-03-01", "2024-03-15")])
    import forecast_windows
    monkeypatch.setattr(forecast_windows, "first_training_day", "2023-12-14")
    years = validation.patchtst_years(df, {"epochs": 2, "d_model": 16, "hidden": 32, "train_stride": 16})
    assert len(years) == 2
    assert all(y["loss"] > 0 and 0 < y["mape"] < 50 and y["epochs"] == 2 for y in years)


def test_every_history_length_is_trained_on_the_same_days(monkeypatch):
    import patchtst
    import validation

    df = make_features("2023-11-01", days=140)
    monkeypatch.setattr(validation, "folds", [("2024-02-15", "2024-03-01")])
    import forecast_windows
    monkeypatch.setattr(forecast_windows, "first_training_day", "2023-12-14")
    trained = {}

    def fit(self, df, starts, **kwargs):
        trained[self.settings["history_days"]] = starts
        raise StopIteration

    monkeypatch.setattr(patchtst.PatchTSTForecaster, "fit", fit)
    for days in (7, 42):
        try:
            validation.patchtst_years(df, {"history_days": days, "train_stride": 16})
        except StopIteration:
            pass
    assert (trained[7] == trained[42]).all()
    assert trained[7].min() >= 42 * 48


def test_the_pinball_loss_is_in_mw_and_scores_each_quantile_on_its_own_side():
    from validation import pinball_mw

    actual = np.array([[100.0]])
    predicted = np.array([[[90.0, 100.0, 110.0]]])  # 10% forecast 10 MW low, median exact, 90% forecast 10 MW high
    assert pinball_mw(actual, predicted, (0.1, 0.5, 0.9)) == pytest.approx((0.1 * 10 + 0 + 0.1 * 10) / 3)
    assert pinball_mw(actual * 2, predicted * 2, (0.1, 0.5, 0.9)) == pytest.approx(2 * (0.1 * 10 + 0.1 * 10) / 3)


def test_a_best_setting_in_the_outer_tenth_of_its_range_is_at_an_edge():
    import optuna

    from validation import edge_settings

    distributions = {"rate": optuna.distributions.FloatDistribution(0.005, 0.5, log=True),
                     "penalty": optuna.distributions.FloatDistribution(0.0, 10.0),
                     "depth": optuna.distributions.IntDistribution(1, 12),
                     "size": optuna.distributions.CategoricalDistribution([32, 64, 128])}
    # 0.4 is 95% of the way along the log scale, 5.0 is the middle, depth 1 is a natural end, 128 the last choice
    params = {"rate": 0.4, "penalty": 5.0, "depth": 1, "size": 128}
    natural = {("depth", "low"): 1}
    assert edge_settings(params, distributions, natural) == [("rate", 0.4, "high"), ("size", 128, "high")]
    assert edge_settings({**params, "rate": 0.05, "size": 64}, distributions, natural) == []
    # an end that isn't yet at the setting's limit is still flagged
    assert edge_settings({**params, "rate": 0.05, "size": 64}, {**distributions,
                         "depth": optuna.distributions.IntDistribution(3, 12)}, natural) == [("depth", 1, "low")]
    

def test_the_year_being_scored_is_never_used_to_choose_when_to_stop_training(monkeypatch):
    import patchtst
    import validation

    df = make_features("2023-11-01", days=140)
    monkeypatch.setattr(validation, "folds", [("2024-02-15", "2024-03-01")])
    import forecast_windows
    monkeypatch.setattr(forecast_windows, "first_training_day", "2023-12-14")
    calls = []
    real_fit = patchtst.PatchTSTForecaster.fit

    def fit(self, df, starts, **kwargs):
        calls.append(kwargs)
        return real_fit(self, df, starts, **kwargs)

    monkeypatch.setattr(patchtst.PatchTSTForecaster, "fit", fit)
    validation.patchtst_years(df, {"epochs": 2, "d_model": 16, "hidden": 32, "train_stride": 16})
    assert calls == [{"epochs": 2}]  # no validation windows passed in, so nothing is picked from the scored year