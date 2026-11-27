import pandas as pd
import pytest

from ablation import average, decide, merge, parse, run_row, run_settings


def test_values_from_the_command_line_become_numbers_where_they_can():
    assert parse("24") == 24 and parse("0.2") == 0.2 and parse("attention") == "attention"


def test_a_new_run_replaces_only_the_rows_for_its_own_setting_and_starting_point():
    old = pd.DataFrame({"setting": ["pooling", "pooling", "patch_len", "pooling"], "value": ["flatten", "mean", "24", "mean"],
                        "tuned": [False, False, False, True], "mape": [6.0, 5.8, 5.9, 5.0]})
    new = pd.DataFrame({"setting": ["pooling"], "value": ["attention"], "tuned": [False], "mape": [5.5]})
    table = merge(old, new)
    assert sorted(zip(table["value"], table["tuned"])) == [("24", False), ("attention", False), ("mean", True)]
    assert merge(None, new).equals(new)


def test_patch_length_runs_keep_patches_overlapping_by_half():
    settings = run_settings({"dropout": 0.2}, "patch_len", 48, seed=1)
    assert settings == {"dropout": 0.2, "patch_len": 48, "stride": 24, "seed": 1}


def test_a_run_is_scored_by_its_loss_averaged_over_both_validation_years():
    years = [{"loss": 0.10, "mape": 4.0, "epochs": 8, "parameters": 1000},
             {"loss": 0.12, "mape": 5.0, "epochs": 8, "parameters": 1000}]
    row = run_row(years)
    assert row["validation_loss"] == pytest.approx(0.11) and row["mape"] == 4.5
    assert row["loss_year1"] == 0.10 and row["epochs"] == 8


def test_seeds_are_averaged_with_the_spread_of_the_deciding_loss():
    row = average([{"validation_loss": 0.10, "mape": 5.0}, {"validation_loss": 0.12, "mape": 6.0}])
    assert row["validation_loss"] == pytest.approx(0.11) and row["mape"] == 5.5 and row["seeds"] == 2
    assert round(row["validation_loss_std"], 4) == 0.0141 and round(row["mape_std"], 3) == 0.707


def comparison(setting, values, losses, seeds=5, tuned=False, std=0.001):
    return pd.DataFrame({"setting": setting, "value": values, "validation_loss": losses, "validation_loss_std": std,
                         "seeds": seeds, "tuned": tuned})


def test_the_lookback_takes_the_lowest_loss_from_its_own_starting_point_and_flags_a_choice_within_the_noise():
    table = pd.concat([comparison("history_days", ["7", "14", "28"], [280.0, 279.5, 285.0], std=2.0),
                       comparison("history_days", ["42"], [200.0], tuned=True),   # from the tuned settings: the recheck
                       comparison("history_days", ["21"], [100.0], seeds=3),      # too few seeds doesn't count
                       comparison("pooling", ["mean"], [100.0])])
    days, report = decide(table)
    assert days == 14 and report["runner_up"] == "7" and report["margin"] < report["seed_spread"]
    with pytest.raises(SystemExit, match="history_days"):
        decide(table, tuned=True)  # one tuned value isn't a comparison
    tuned = pd.concat([table, comparison("history_days", ["7"], [210.0], tuned=True)])
    assert decide(tuned, tuned=True)[0] == 42