import pandas as pd
import pytest

from ablation import average, decide, merge, parse, run_row, run_settings


def test_values_from_the_command_line_become_numbers_where_they_can():
    assert parse("24") == 24 and parse("0.2") == 0.2 and parse("attention") == "attention"


def test_a_new_run_replaces_only_the_rows_for_its_own_setting():
    old = pd.DataFrame({"setting": ["pooling", "pooling", "patch_len"], "value": ["flatten", "mean", "24"], "mape": [6.0, 5.8, 5.9]})
    new = pd.DataFrame({"setting": ["pooling"], "value": ["attention"], "mape": [5.5]})
    table = merge(old, new)
    assert sorted(table["value"]) == ["24", "attention"]
    assert merge(None, new).equals(new)


def test_patch_length_runs_keep_patches_overlapping_by_half():
    settings = run_settings({"dropout": 0.2}, "patch_len", 48, seed=1)
    assert settings == {"dropout": 0.2, "patch_len": 48, "stride": 24, "seed": 1}


def test_a_run_is_scored_by_its_loss_averaged_over_both_validation_years():
    years = [{"loss": 0.10, "mape": 4.0, "best_epoch": 8, "parameters": 1000},
             {"loss": 0.12, "mape": 5.0, "best_epoch": 6, "parameters": 1000}]
    row = run_row(years)
    assert row["validation_loss"] == pytest.approx(0.11) and row["mape"] == 4.5
    assert row["loss_year1"] == 0.10 and row["best_epoch_year2"] == 6


def test_seeds_are_averaged_with_the_spread_of_the_deciding_loss():
    row = average([{"validation_loss": 0.10, "mape": 5.0}, {"validation_loss": 0.12, "mape": 6.0}])
    assert row["validation_loss"] == pytest.approx(0.11) and row["mape"] == 5.5 and row["seeds"] == 2
    assert round(row["validation_loss_std"], 4) == 0.0141 and round(row["mape_std"], 3) == 0.707


def comparison(setting, values, losses, seeds=5, tuned=False, std=0.001):
    return pd.DataFrame({"setting": setting, "value": values, "validation_loss": losses, "validation_loss_std": std,
                         "seeds": seeds, "tuned": tuned})


def test_the_design_takes_the_lowest_loss_for_each_setting_and_flags_choices_within_the_noise():
    table = pd.concat([comparison("pooling", ["flatten", "mean", "attention"], [0.12, 0.11, 0.10]),
                       comparison("revin", ["1", "0"], [0.105, 0.1045], std=0.002),
                       comparison("revin", ["0"], [0.05], tuned=True),          # a run from tuned settings doesn't count
                       comparison("lags", ["0", "1"], [0.1, 0.2], seeds=3)])     # nor one with too few seeds
    with pytest.raises(SystemExit, match="lags"):
        decide(table)
    chosen, report = decide(table, settings=["pooling", "revin"])
    assert chosen == {"pooling": "attention", "revin": 0}
    close = report.set_index("setting")
    assert close.loc["revin", "margin"] < close.loc["revin", "seed_spread"]
    assert close.loc["pooling", "margin"] > close.loc["pooling", "seed_spread"]