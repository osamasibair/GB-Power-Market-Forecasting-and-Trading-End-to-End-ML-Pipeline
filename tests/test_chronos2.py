import numpy as np
import pytest
import torch
from test_patchtst import make_features

from chronos2 import (
    add_columns,
    day_inputs,
    export,
    load_export,
    names,
    read_forecasts,
    regular_grid,
    steps_ahead,
    to_rows,
    training_series,
)
from forecast_windows import day_windows, gap_steps, horizon

torch.set_num_threads(1)


def test_the_grid_has_no_gaps_and_missing_demand_stays_missing():
    df = make_features(days=20)
    df = df.drop(df.index[500:510])
    demand, cov, grid = regular_grid(df)
    assert len(grid) == len(df) + 10 and np.isnan(demand).sum() == 10
    assert not np.isnan(cov).any() and cov.shape == (len(grid), len(names))


def test_each_day_sees_demand_up_to_08_00_and_covariates_for_the_next_80_half_hours():
    df = make_features(days=20)
    demand, cov, grid = regular_grid(df)
    start = day_windows(df.index)[-1:]
    position = grid.get_indexer(df.index[start])[0]
    (item,) = day_inputs(demand, cov, [position], context=500, use_covariates=True)
    assert len(item["target"]) == 500 and item["target"][-1] == demand[position - gap_steps - 1]
    assert all(len(values) == steps_ahead for values in item["future_covariates"].values())
    assert np.allclose(item["future_covariates"]["temperature_2m"][-horizon:], cov[position:position + horizon, 0])
    (plain,) = day_inputs(demand, cov, [position], context=500, use_covariates=False)
    assert set(plain) == {"target"}


def test_fine_tuning_only_sees_data_before_the_cutoff():
    df = make_features(days=20)
    demand, cov, _ = regular_grid(df)
    (series,) = training_series(demand, cov, end=600)
    assert len(series["target"]) == 600 and all(len(v) == 600 for v in series["past_covariates"].values())


def test_new_columns_must_match_the_saved_table():
    import pandas as pd

    table = pd.DataFrame({"actual": [1.0, 2.0]})
    assert list(add_columns(table, {"chronos2": np.array([1.5, 2.5])}).columns) == ["actual", "chronos2"]
    with pytest.raises(SystemExit):
        add_columns(table, {"chronos2": np.array([1.0])})


def test_the_exported_file_gives_back_the_same_inputs_and_days(tmp_path):
    df = make_features(days=20)
    df = df.drop(df.index[500:510])
    demand, cov, grid = regular_grid(df)
    days = grid.get_indexer(df.index[day_windows(df.index)])
    calib, test = days[:5], days[5:]
    export(tmp_path / "input.csv.gz", demand, cov, grid, calib, test)
    demand2, cov2, grid2, calib2, test2 = load_export(tmp_path / "input.csv.gz")
    assert np.allclose(demand2, demand, equal_nan=True) and np.allclose(cov2, cov, atol=1e-5)
    assert grid2.equals(grid) and list(calib2) == list(calib) and list(test2) == list(test)


def test_forecasts_made_elsewhere_are_read_back_in_order_and_checked(tmp_path):
    import pandas as pd

    df = make_features(days=20)
    _, _, grid = regular_grid(df)
    days = grid.get_indexer(df.index[day_windows(df.index)])
    calib, test = days[:3], days[3:5]
    made = np.arange(5 * horizon * 3, dtype=float).reshape(5, horizon, 3)
    pd.concat([to_rows(made[:3], grid, calib, "calibration"), to_rows(made[3:], grid, test, "test")]).to_csv(tmp_path / "out.csv.gz")

    def times(positions):
        return grid[(positions[:, None] + np.arange(horizon)).ravel()]

    calib_back, test_back = read_forecasts(tmp_path / "out.csv.gz", times(calib), times(test))
    assert np.allclose(calib_back, made[:3].reshape(-1, 3)) and np.allclose(test_back, made[3:].reshape(-1, 3))
    with pytest.raises(SystemExit):
        read_forecasts(tmp_path / "out.csv.gz", times(calib[:2]), times(test))


def tiny_pipeline():
    chronos = pytest.importorskip("chronos")
    from chronos.chronos2.config import Chronos2CoreConfig

    levels = [0.01, 0.05, *[round(x, 1) for x in np.arange(0.1, 0.95, 0.1)], 0.95, 0.99]
    config = Chronos2CoreConfig(d_model=32, d_kv=8, d_ff=64, num_layers=1, num_heads=4, chronos_config={
        "context_length": 1024, "input_patch_size": 16, "input_patch_stride": 16, "output_patch_size": 16,
        "quantiles": levels, "use_reg_token": True, "use_arcsinh": True, "max_output_patches": 8})
    return chronos.Chronos2Pipeline(chronos.Chronos2Model(config))


def test_forecasts_from_a_tiny_untrained_chronos2_have_the_right_shape_and_order():
    pipeline = tiny_pipeline()
    from chronos2 import forecast

    df = make_features(days=30)
    demand, cov, grid = regular_grid(df)
    positions = grid.get_indexer(df.index[day_windows(df.index)[-2:]])
    out = forecast(pipeline, demand, cov, positions, context=512, use_covariates=True)
    assert out.shape == (2, horizon, 3)
    assert (out[..., 0] <= out[..., 1]).all() and (out[..., 1] <= out[..., 2]).all()


def test_fine_tuning_runs_and_gives_calibration_and_test_forecasts(tmp_path, monkeypatch):
    from argparse import Namespace

    pipeline = tiny_pipeline()
    from chronos2 import fine_tuned_forecasts

    monkeypatch.chdir(tmp_path)
    df = make_features(days=60)
    demand, cov, grid = regular_grid(df)
    days = grid.get_indexer(df.index[day_windows(df.index)])
    settings = Namespace(context=512, steps=2, learning_rate=1e-5, batch_size=8, accumulate=2, seed=42)
    calib, test = fine_tuned_forecasts(pipeline, demand, cov, days[-6:-3], days[-3:], settings)
    assert calib.shape == test.shape == (3, horizon, 3)


def test_fine_tuning_with_the_same_seed_repeats_and_another_seed_differs(tmp_path, monkeypatch):
    from chronos2 import fine_tune, forecast

    monkeypatch.chdir(tmp_path)
    df = make_features(days=60)
    demand, cov, grid = regular_grid(df)
    days = grid.get_indexer(df.index[day_windows(df.index)])

    def run(seed):
        torch.manual_seed(0)  # the same starting model every time, so only the fine tuning seed differs
        tuned = fine_tune(tiny_pipeline(), demand, cov, days[-3] - gap_steps, 512, 3, 1e-3, 8, 1, seed, tmp_path / str(seed))
        return forecast(tuned, demand, cov, days[-3:], 512, True)

    first = run(1)
    assert np.allclose(first, run(1))
    assert not np.allclose(first, run(2))


def test_the_gpu_machine_reads_the_input_names_from_the_export(tmp_path, monkeypatch):
    import chronos2

    df = make_features(days=20)
    demand, cov, grid = regular_grid(df)
    days = grid.get_indexer(df.index[day_windows(df.index)])
    export(tmp_path / "input.csv.gz", demand, cov, grid, days[:5], days[5:])
    monkeypatch.setattr(chronos2, "names", names[:3])  # as if the GPU machine had an older list of inputs
    _, cov2, _, _, _ = load_export(tmp_path / "input.csv.gz")
    assert chronos2.names == names and np.allclose(cov2, cov, atol=1e-5)