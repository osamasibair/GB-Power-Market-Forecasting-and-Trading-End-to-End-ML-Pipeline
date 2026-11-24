import itertools

import pandas as pd
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

    df = make_features("2024-01-01", days=90)
    monkeypatch.setattr(validation, "folds", [("2024-02-15", "2024-03-01"), ("2024-03-01", "2024-03-15")])
    years = validation.patchtst_years(df, {"max_epochs": 2, "d_model": 16, "hidden": 32, "train_stride": 16})
    assert len(years) == 2
    assert all(y["loss"] > 0 and 0 < y["mape"] < 50 and 1 <= y["best_epoch"] <= 2 for y in years)