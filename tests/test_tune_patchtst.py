from types import SimpleNamespace

import optuna
import pytest

from patchtst import PatchTST, PatchTSTForecaster, defaults
from tune_patchtst import (
    best_settings,
    confirm,
    lookback_comparison,
    natural_ends,
    search,
    suggest,
)
from validation import edge_settings

base = {"history_days": 14}


def test_every_suggested_setting_is_a_valid_model_setting():
    trial = optuna.trial.FixedTrial({"pooling": "mean", "revin": 0, "lags": 1, "learning_rate": 1e-3, "weight_decay": 1e-2,
                                     "dropout": 0.1, "d_model": 32, "layers": 1, "hidden": 64, "patch_len": 48,
                                     "batch_size": 64, "epochs": 5})
    settings = suggest(trial)
    assert settings["stride"] == 24
    model = PatchTSTForecaster(**base, **settings)
    PatchTST(11, **{name: model.settings[name] for name in ["patch_len", "stride", "d_model", "layers", "hidden", "lags",
                                                             "pooling", "revin", "history_days"]})


def test_the_first_trial_can_be_the_current_defaults():
    trial = optuna.trial.FixedTrial({name: defaults[name] for name in search})
    settings = suggest(trial)
    assert all(settings[name] == defaults[name] for name in settings)


def test_only_the_lookback_is_chosen_before_tuning_and_the_head_revin_and_lags_are_searched():
    assert not set(base) & set(search)
    assert {"pooling", "revin", "lags"} <= set(search)


def test_best_settings_add_the_stride_back_and_keep_the_lookback():
    trial = SimpleNamespace(params={"patch_len": 12, "learning_rate": 1e-3})
    assert best_settings(trial, base) == {**base, "patch_len": 12, "learning_rate": 1e-3, "stride": 6}


def test_a_choice_between_every_option_there_is_is_never_an_edge_to_widen():
    trial = optuna.create_study().ask()
    suggest(trial)
    for pooling, revin, lags in [("flatten", 0, 0), ("attention", 1, 1)]:
        params = {**trial.params, "pooling": pooling, "revin": revin, "lags": lags}
        flagged = {name for name, _, _ in edge_settings(params, trial.distributions, natural_ends)}
        assert not flagged & {"pooling", "revin", "lags"}


def test_a_callback_can_stop_training_early():
    from test_patchtst import make_features

    from forecast_windows import day_windows, windows

    df = make_features(days=40)
    seen = []

    def stop_after_two(epoch, loss):
        seen.append(epoch)
        if epoch == 2:
            raise optuna.TrialPruned()

    model = PatchTSTForecaster(max_epochs=5, d_model=16, hidden=32)
    with pytest.raises(optuna.TrialPruned):
        model.fit(df, windows(df.index, last="2024-02-01", stride=16), valid=day_windows(df.index, first="2024-02-01"),
                  callback=stop_after_two)
    assert seen == [1, 2]


def test_the_best_trials_are_chosen_by_their_loss_averaged_over_every_seed():
    lucky = SimpleNamespace(number=1, value=260.0, params={"patch_len": 24, "dropout": 0.1})
    steady = SimpleNamespace(number=2, value=262.0, params={"patch_len": 24, "dropout": 0.2})
    later = {0.1: 280.0, 0.2: 263.0}  # the lucky trial's first seed was its best, the steady one's weren't
    seen = []

    def years(df, settings):
        seen.append(settings["seed"])
        return [{"loss": later[settings["dropout"]]}, {"loss": later[settings["dropout"]]}]

    results = confirm([lucky, steady], base, None, years)
    assert [r["trial"].number for r in results] == [2, 1]
    assert results[0]["mean"] == (262.0 + 4 * 263.0) / 5 and len(results[0]["losses"]) == 5
    assert seen == [1, 2, 3, 4] * 2


def test_the_widened_ranges_include_the_larger_models_and_more_dropout():
    trial = optuna.create_study().ask()
    suggest(trial)
    assert trial.distributions["d_model"].choices == (32, 64, 128, 256)
    assert trial.distributions["hidden"].choices == (64, 128, 256, 512)
    assert (trial.distributions["dropout"].low, trial.distributions["dropout"].high) == (0.0, 0.5)
    assert (trial.distributions["layers"].low, trial.distributions["layers"].high) == (1, 5)
    assert (trial.distributions["weight_decay"].low, trial.distributions["weight_decay"].high) == (1e-4, 1.0)
    assert (trial.distributions["epochs"].low, trial.distributions["epochs"].high) == (2, 40)


def test_every_lookback_is_trained_with_every_seed_and_ranked_by_its_loss_averaged_over_years_and_seeds():
    seen = []

    def years(df, settings):
        seen.append((settings["history_days"], settings["seed"]))
        loss = 300.0 - settings["history_days"] + settings["seed"]  # 42 days best; seeds move the loss
        return [{"loss": loss, "mape": 3.0}, {"loss": loss + 10, "mape": 4.0}]

    results = lookback_comparison(None, {"dropout": 0.2}, years=years)
    assert sorted(seen) == sorted((d, s) for d in (7, 14, 21, 28, 42) for s in (42, 1, 2, 3, 4))
    assert [r["days"] for r in results] == [42, 28, 21, 14, 7]
    best = results[0]
    assert best["loss_year1"] == 300.0 - 42 + 52 / 5 and best["loss"] == best["loss_year1"] + 5 and best["mape"] == 3.5
