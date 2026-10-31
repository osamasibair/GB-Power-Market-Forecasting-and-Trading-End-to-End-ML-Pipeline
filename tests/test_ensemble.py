import numpy as np
import pandas as pd
import pytest

from ensemble import (
    all_blends,
    best_weight,
    blend,
    mix,
    three_way_table,
    weight_table,
    weights,
)


def make_frame(rows=480, seed=0, lightgbm_noise=400, patchtst_noise=400):
    rng = np.random.default_rng(seed)
    actual = 30000 + 4000 * np.sin(np.arange(rows) / 7.6)
    lightgbm = actual + rng.normal(0, lightgbm_noise, rows)
    patchtst = actual + rng.normal(0, patchtst_noise, rows)
    return pd.DataFrame({
        "actual": actual,
        "lightgbm": lightgbm, "lightgbm_low": lightgbm - 600, "lightgbm_high": lightgbm + 600,
        "patchtst": patchtst, "patchtst_low": patchtst - 600, "patchtst_high": patchtst + 600,
    }, index=pd.date_range("2025-09-01", periods=rows, freq="30min"))


def test_blend_at_the_ends_is_each_model_alone():
    frame = make_frame()
    assert np.allclose(blend(frame, 0)["ensemble"], frame["lightgbm"])
    assert np.allclose(blend(frame, 1)["ensemble_high"], frame["patchtst_high"])


def test_blend_halfway_is_the_average():
    frame = make_frame()
    assert np.allclose(blend(frame, 0.5)["ensemble"], (frame["lightgbm"] + frame["patchtst"]) / 2)


def test_independent_errors_make_the_blend_beat_both_models():
    calib, test = make_frame(seed=1), make_frame(seed=2)
    table = weight_table(calib, test).set_index("weight")
    middle = table.loc[0.5, "test_mape"]
    assert middle < table.loc[0.0, "test_mape"] and middle < table.loc[1.0, "test_mape"]


def test_the_weight_is_chosen_on_the_calibration_year_only():
    calib = make_frame(seed=1, lightgbm_noise=100, patchtst_noise=900)
    test = make_frame(seed=2, lightgbm_noise=900, patchtst_noise=100)
    table = weight_table(calib, test)
    assert best_weight(table) < 0.3
    assert table.set_index("weight")["test_mape"].idxmin() > 0.7


def test_every_weight_is_recalibrated_to_about_80_percent():
    calib, test = make_frame(seed=1), make_frame(seed=2)
    table = weight_table(calib, test)
    assert len(table) == len(weights)
    assert table["test_coverage"].between(70, 90).all()
    assert table["weight"].tolist() == pytest.approx(list(weights))




def test_mix_with_two_models_matches_blend():
    frame = make_frame()
    assert np.allclose(mix(frame, {"patchtst": 0.3, "lightgbm": 0.7})["ensemble_low"], blend(frame, 0.3)["ensemble_low"])


def test_three_way_weights_add_up_to_one_and_cover_each_model_alone():
    calib, test = make_frame(seed=1), make_frame(seed=2)
    for frame in (calib, test):
        frame["chronos2_finetuned"] = frame["actual"] + 300
        frame["chronos2_finetuned_low"] = frame["chronos2_finetuned"] - 600
        frame["chronos2_finetuned_high"] = frame["chronos2_finetuned"] + 600
    table = three_way_table(calib, test, "chronos2_finetuned")
    total = table["lightgbm_weight"] + table["patchtst_weight"] + table["third_weight"]
    assert np.allclose(total, 1) and len(table) == 231
    assert (table["third_weight"] == 1).sum() == 1 and (table["lightgbm_weight"] == 1).sum() == 1


def test_every_blend_is_searched_and_sorted_by_the_calibration_year():
    calib, test = make_frame(seed=1), make_frame(seed=2)
    table = all_blends(calib, test, ["lightgbm", "patchtst"], steps=4)
    assert len(table) == 5 and np.allclose(table["lightgbm"] + table["patchtst"], 1)
    assert table["calibration_mape"].is_monotonic_increasing
    for frame in (calib, test):
        frame["chronos2_finetuned"] = frame["actual"] + 300
        frame["chronos2_finetuned_low"] = frame["chronos2_finetuned"] - 600
        frame["chronos2_finetuned_high"] = frame["chronos2_finetuned"] + 600
    three = all_blends(calib, test, ["lightgbm", "patchtst", "chronos2_finetuned"])
    assert len(three) == 231 and np.allclose(three[["lightgbm", "patchtst", "chronos2_finetuned"]].sum(axis=1), 1)