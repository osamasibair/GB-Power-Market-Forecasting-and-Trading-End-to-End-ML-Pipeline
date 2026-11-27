"""Choose and tune the gradient boosting models, LightGBM and XGBoost, with the same budget and rules as every other
model, so the comparison between gradient boosting and deep learning is fair. Every choice is made on the two validation
years in validation.py (each forecast by a model trained on the data from the first training day every model shares up
to the year), and scored like every other model's: by the pinball loss in MW of the 10%, 50% and 90% forecasts
(validation.pinball_mw), averaged over the two years. Each model forecasts with three quantile models, and its forecast
is the 50% one, as for every other model. The test year stays unseen.

Both models get the same weather and calendar inputs as every other model (forecast_windows.known_inputs), with no
selection, and read past demand through lag columns. Three steps, in this order:
1. Lookback, for each model: how far back its lags reach, from the same options as every other model (7 to 42 days,
   forecast_windows.lookbacks). Beyond the 7 days of lags the models had, each extra week adds the demand at the same
   time that many weeks earlier. Default settings; the best is saved to models/<model>_lookback.json.
       python src/tune_boosting.py --model lightgbm --check-lookback
2. Settings: 40 optuna trials with that lookback, the same budget as the neural models. The first trial is the default
   settings, so the gain from tuning is visible. Trials train in seconds, so they aren't pruned. The study is saved in
   models/, so it can be stopped with Ctrl+C and resumed by running this again. Each trial uses one seed, so the best
   three are then scored with all five seeds, as for every tuned model, and the one with the lowest average wins. Its
   settings go to models/<model>_tuned.json, and the tuned model is then trained with five seeds. Their forecasts are
   averaged, the 80% range is calibrated on the calibration year like every other model's, and the result is added to
   transformer_forecasts and transformer_calibration as <model>_tuned, so run src/train_patchtst.py first.
       python src/tune_boosting.py --model lightgbm
3. Recheck, as for every tuned model: the lookback is compared again with the tuned settings (five seeds, since tuned
   settings sample rows and columns). If a different lookback wins, the tuned model reads that instead and its final
   five-seed model is trained again; the settings aren't tuned again, so it's one pass, as for every model.
       python src/tune_boosting.py --model lightgbm --recheck-lookback

With default settings neither model has any randomness (no row or column sampling), so the lookback check trains once
instead of once per seed: five seeds would give five identical results.

Doesn't import PyTorch: on macOS, LightGBM or XGBoost and PyTorch in one process can crash, because each brings its own
OpenMP library."""
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

from evaluate import metrics
from forecast_windows import first_training_day, known_inputs, lookbacks
from validation import before, edge_settings, fold_rows, folds, pinball_mw

models = Path("models")
docs = Path("docs")
seeds = [42, 1, 2, 3, 4]
quantiles = (0.1, 0.5, 0.9)
target = "demand"
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)

# the defaults: each library's own, apart from the number of trees and the learning rate, which start where
# src/train.py's LightGBM does
defaults = {
    "lightgbm": {"n_estimators": 500, "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 20,
                 "subsample": 1.0, "colsample_bytree": 1.0, "reg_lambda": 0.0},
    "xgboost": {"n_estimators": 500, "learning_rate": 0.05, "max_depth": 6, "min_child_weight": 1.0,
                "subsample": 1.0, "colsample_bytree": 1.0, "reg_lambda": 1.0},
}
# the lags every lookback has: same time yesterday (or two days ago), same time last week, and the 24 hour average
# ending two days earlier
lag_base = ["demand_lag_recent", "demand_lag_336", "demand_roll_96"]


def suggest(name, trial):
    """The settings to try in this trial: the same kinds of setting for both models, each model with its own ranges.
    The rule, the same for every model (validation.edge_settings): a range whose best setting comes out in its outer
    10% is widened and that model's search rerun, so each model's best version lies inside what is searched. Widened
    after the first search for both (fewer leaves, shallower trees, more rows per leaf, fewer columns, a higher learning
    rate), and after the second for LightGBM's leaves (down to 2) and XGBoost's learning rate (up to 1)."""
    common = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 5000, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.5 if name == "lightgbm" else 1.0, log=True),
    }
    if name == "lightgbm":
        size = {"num_leaves": trial.suggest_int("num_leaves", 2, 255, log=True),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 1000, log=True)}
    else:
        size = {"max_depth": trial.suggest_int("max_depth", 1, 12),
                "min_child_weight": trial.suggest_float("min_child_weight", 0.01, 100.0, log=True)}
    return {**common, **size,
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0)}


def make_model(name, settings, quantile, seed=42):
    """A quantile model. LightGBM bags every iteration, which does nothing when subsample is 1."""
    if name == "lightgbm":
        import lightgbm as lgb

        return lgb.LGBMRegressor(**settings, objective="quantile", alpha=quantile, subsample_freq=1, random_state=seed,
                                 verbose=-1)
    if name == "xgboost":
        import xgboost as xgb

        return xgb.XGBRegressor(**settings, objective="reg:quantileerror", quantile_alpha=quantile, tree_method="hist",
                                random_state=seed)
    raise ValueError(f"unknown model {name!r}")


def has_randomness(settings):
    return settings["subsample"] < 1 or settings["colsample_bytree"] < 1


def seeds_for(settings):
    """Five seeds when the settings sample rows or columns, one when they don't (the five would be identical)."""
    return seeds if has_randomness(settings) else seeds[:1]


def add_lags(df):
    """The demand at the same time 14, 21, ... 42 days earlier, for the longer lookbacks. Matched by time, so a gap in
    the data gives a missing value (which both models handle) rather than the wrong day."""
    df = df.copy()
    for days in range(14, max(lookbacks) + 1, 7):
        df[f"demand_lag_{days}d"] = df[target].shift(freq=pd.Timedelta(days=days)).reindex(df.index)
    return df


def lag_columns(days):
    """The lags for a lookback of `days`: the base ones, plus the same time each week back to `days`."""
    if days not in lookbacks:
        raise ValueError(f"lookback must be one of {lookbacks}, not {days!r}")
    return lag_base + [f"demand_lag_{d}d" for d in range(14, days + 1, 7)]


def columns_for(days):
    """Every model's weather and calendar inputs, plus the lags for a lookback of `days`."""
    return known_inputs + lag_columns(days)


def quantile_forecasts(name, settings, columns, fit_part, rows, seed=42):
    """The 10%, 50% and 90% forecasts for `rows`, shape (rows, 3), from three models trained on fit_part. Sorted along
    the last axis, so they never cross."""
    out = [make_model(name, settings, q, seed).fit(fit_part[columns], fit_part[target]).predict(rows[columns])
           for q in quantiles]
    return np.sort(np.stack(out, axis=1), axis=1)


def year_scores(name, settings, columns, years, seed=42):
    """For each validation year (data before it, its forecast rows): the pinball loss in MW and the median's MAPE."""
    out = []
    for fit_part, year in years:
        predicted = quantile_forecasts(name, settings, columns, fit_part, year, seed)
        actual = year[target].to_numpy()
        out.append({"loss": pinball_mw(actual, predicted, quantiles),
                    "mape": float(metrics(actual, predicted[:, 1])["mape"])})
    return out


def compare(name, settings, options, years):
    """Score each option, a {label: columns} dict, on both validation years: per year and on average (which decides),
    averaged over the seeds when the settings have randomness, with the spread between seeds. Best first; on an exact
    tie the simpler option (fewer columns) comes first."""
    results = []
    for label, columns in options.items():
        runs = [year_scores(name, settings, columns, years, seed) for seed in seeds_for(settings)]
        losses = np.array([[year["loss"] for year in run] for run in runs])
        mapes = np.array([[year["mape"] for year in run] for run in runs])
        both = losses.mean(axis=1)
        results.append({"option": label, "columns": columns,
                        **{f"loss_year{i + 1}": float(losses[:, i].mean()) for i in range(len(years))},
                        "loss": float(both.mean()), "loss_std": float(both.std(ddof=1)) if len(runs) > 1 else 0.0,
                        "mape": float(mapes.mean())})
    return sorted(results, key=lambda r: (r["loss"], len(r["columns"])))  # on a tie, the option with fewer columns


# each model's study is named after the last time its ranges were widened, so a model whose ranges didn't change keeps
# its finished trials
search_round = {"lightgbm": "wide2", "xgboost": "wide2"}
# ends of a range that can't go further, so a best setting there isn't a reason to widen: a tree needs at least 2
# leaves and 1 level, no L2 penalty is the least, and rows and columns can't be sampled above all of them
natural_ends = {("num_leaves", "low"): 2, ("max_depth", "low"): 1, ("reg_lambda", "low"): 0.0, ("subsample", "high"): 1.0,
                ("colsample_bytree", "high"): 1.0}


def storage_for(name, days):
    """Each model and lookback gets its own study, a new one each time that model's ranges are widened."""
    return f"sqlite:///{models / f'optuna_{name}_{days}d_{search_round[name]}.db'}"


def tuned_forecasts(name, df, calib, test, settings, columns, calibration_start, split_date, seeds=seeds):
    """The tuned model's forecasts and 80% range on the calibration and test years, made like every other model's:
    trained from the first training day to the calibration year to forecast it and calibrate the range, then to the test
    year to forecast that. Each seed's 10/50/90% forecasts are averaged; each seed's test MAPE is kept, to show the
    spread between seeds."""
    from train_patchtst import conformal_adjustment

    early_train = df[(df.index >= first_training_day) & (df.index < calibration_start)]
    train = df[(df.index >= first_training_day) & (df.index < split_date)]
    early_runs = [quantile_forecasts(name, settings, columns, early_train, calib, seed) for seed in seeds]
    final_runs = [quantile_forecasts(name, settings, columns, train, test, seed) for seed in seeds]
    early, final = np.mean(early_runs, axis=0), np.mean(final_runs, axis=0)
    adjustment = conformal_adjustment(calib[target].to_numpy(), early[:, 0], early[:, 2])
    actual = test[target].to_numpy()
    key = f"{name}_tuned"

    def interval(forecast):
        return {key: forecast[:, 1], f"{key}_low": np.minimum(forecast[:, 0] - adjustment, forecast[:, 1]),
                f"{key}_high": np.maximum(forecast[:, 2] + adjustment, forecast[:, 1])}

    return {"calibration": interval(early), "forecasts": interval(final), "adjustment": adjustment,
            "seed_mapes": [float(metrics(actual, run[:, 1])["mape"]) for run in final_runs]}


def confirm(name, trials, columns, years, scores=None):
    """Score each of `trials` with every seed and return them with the loss averaged over the seeds, best first, the
    same check every tuned model gets. Each trial's own run is its seed 42 result, so only the other seeds are trained;
    settings with no randomness give the same result for every seed, so they aren't trained again. `scores` (for tests)
    replaces year_scores."""
    scores = scores or year_scores
    results = []
    for trial in trials:
        settings = {k: trial.params[k] for k in defaults[name]}
        losses = [trial.value]
        for seed in seeds_for(settings)[1:]:
            losses.append(float(np.mean([year["loss"] for year in scores(name, settings, columns, years, seed)])))
        losses += [trial.value] * (len(seeds) - len(losses))  # no randomness: every seed gives the seed 42 result
        results.append({"trial": trial, "settings": settings, "losses": losses, "mean": float(np.mean(losses)),
                        "std": float(np.std(losses, ddof=1))})
    return sorted(results, key=lambda r: r["mean"])


def save_final(name, df, calib, test, settings, columns, calibration_start, split_date):
    """Train the final model with five seeds, average their forecasts, calibrate the range, add it to
    transformer_forecasts and transformer_calibration as <name>_tuned, and return its test scores and each seed's MAPE."""
    from chronos2 import add_columns, scored
    from db import load_table, save_table

    print(f"\ntraining the tuned settings with seeds {', '.join(map(str, seeds))} and averaging their forecasts")
    result = tuned_forecasts(name, df, calib, test, settings, columns, calibration_start, split_date)
    forecasts_table = add_columns(load_table("transformer_forecasts"), result["forecasts"])
    calibration_table = add_columns(load_table("transformer_calibration"), result["calibration"])
    save_table(forecasts_table, "transformer_forecasts")
    save_table(calibration_table, "transformer_calibration")
    m = scored(forecasts_table, [f"{name}_tuned"])[f"{name}_tuned"]
    spread = result["seed_mapes"]
    print(f"{name}_tuned (5 seeds averaged): test MAPE {m['mape']:.2f}%, coverage {m['coverage']:.1f}%, "
          f"width {m['width']:.0f} MW; single seeds {np.mean(spread):.2f}% ± {np.std(spread, ddof=1):.2f}")
    return m, spread


def print_comparison(title, results, label):
    print(f"\n{title}")
    print(f"{label:>48} {'year 1':>10} {'year 2':>10} {'average (decides)':>22} {'MAPE':>7}")
    for r in results:
        print(f"{r['option']:>48} {r['loss_year1']:>7.1f} MW {r['loss_year2']:>7.1f} MW "
              f"{r['loss']:>11.1f} ± {r['loss_std']:.1f} MW {r['mape']:>6.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=sorted(defaults), default="lightgbm")
    parser.add_argument("--check-lookback", action="store_true", help="step 1: choose this model's lookback, then stop")
    parser.add_argument("--recheck-lookback", action="store_true",
                        help="step 3: compare the lookbacks again with the tuned settings, then stop")
    parser.add_argument("--trials", type=int, default=40,
                        help="how many finished trials to reach in total (default 40, the same budget as every model)")
    parser.add_argument("--confirm", type=int, default=3,
                        help="how many of the best trials to score with every seed before choosing (default 3)")
    args = parser.parse_args()
    name = args.model

    from db import load_table
    from forecast_windows import day_windows
    from quantiles import calibration_start
    from train import split_date
    from train_patchtst import rows

    df = add_lags(load_table("features"))
    calib = df.iloc[rows(day_windows(df.index, calibration_start, split_date))]
    test = df.iloc[rows(day_windows(df.index, split_date))]
    years = [(before(df, start), fold_rows(df, start, end)) for start, end in folds]
    span = ", ".join(f"{start} to {end} ({len(year) // 48} days)" for (start, end), (_, year) in zip(folds, years))
    print(f"validation years {span}, each forecast by a model trained from {first_training_day} to the year")
    models.mkdir(exist_ok=True)

    lookback_path = models / f"{name}_lookback.json"

    if args.check_lookback:
        options = {f"{days} days": columns_for(days) for days in lookbacks}
        results = compare(name, defaults[name], options, years)
        print_comparison(f"{name} lookback, default settings ({len(seeds_for(defaults[name]))} run each):", results,
                         "lookback")
        best = int(results[0]["option"].split()[0])
        lookback_path.write_text(json.dumps({"lookback_days": best}, indent=2) + "\n")
        print(f"\nchosen: {best} days, saved to {lookback_path}")
        print(f"margin over the runner-up: {results[1]['loss'] - results[0]['loss']:.1f} MW")
        raise SystemExit

    if not lookback_path.exists():
        raise SystemExit(f"{lookback_path} not found, run python src/tune_boosting.py --model {name} --check-lookback first")
    days = json.loads(lookback_path.read_text())["lookback_days"]
    columns = columns_for(days)
    tuned_path = models / f"{name}_tuned.json"

    from tracking import log_run

    if args.recheck_lookback:
        if not tuned_path.exists():
            raise SystemExit(f"{tuned_path} not found, tune {name} first")
        settings = json.loads(tuned_path.read_text())
        options = {f"{d} days": columns_for(d) for d in lookbacks}
        results = compare(name, settings, options, years)
        print_comparison(f"{name} lookback rechecked with the tuned settings ({len(seeds_for(settings))} seeds each):",
                         results, "lookback")
        best_days = int(results[0]["option"].split()[0])
        print(f"\nmargin over the runner-up: {results[1]['loss'] - results[0]['loss']:.1f} MW")
        if best_days == days:
            print(f"the tuned model keeps {days} days")
            raise SystemExit
        lookback_path.write_text(json.dumps({"lookback_days": best_days, "before_recheck": days}, indent=2) + "\n")
        print(f"the tuned model now reads {best_days} days instead of {days}, saved to {lookback_path}")
        save_final(name, df, calib, test, settings, columns_for(best_days), calibration_start, split_date)
        raise SystemExit

    study = optuna.create_study(study_name=name, storage=storage_for(name, days), load_if_exists=True,
                                direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    # a resumed study gets a new sampler seed: the same seed would repeat the settings already tried
    study.sampler = optuna.samplers.TPESampler(seed=42 + len(study.trials))
    if not study.trials:
        study.enqueue_trial(defaults[name])
    done_before = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    print(f"{name}: lookback {days} days, tuning by the pinball loss in MW")

    def objective(trial):
        scores = year_scores(name, suggest(name, trial), columns, years)
        for i, year in enumerate(scores, start=1):
            trial.set_user_attr(f"loss_year{i}", year["loss"])
            trial.set_user_attr(f"mape_year{i}", year["mape"])
        return float(np.mean([year["loss"] for year in scores]))

    try:
        study.optimize(objective, n_trials=max(args.trials - done_before, 0))
    except KeyboardInterrupt:
        print("\nstopped, keeping the trials finished so far (run again to continue)")

    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not done:
        raise SystemExit("no finished trials yet")
    first, best = study.trials[0], study.best_trial
    print(f"\n{len(done)} finished trials: the defaults, then the best five")
    for trial in [first] + [t for t in sorted(done, key=lambda t: t.value) if t.number != first.number][:5]:
        a = trial.user_attrs
        note = "   <- default settings" if trial.number == first.number else ""
        print(f"  trial {trial.number:>3}: {trial.value:7.1f} MW (years {a['loss_year1']:.1f}, {a['loss_year2']:.1f}), "
              f"MAPE {np.mean([a['mape_year1'], a['mape_year2']]):.2f}%  {trial.params}{note}")
    top = sorted(done, key=lambda t: t.value)[:args.confirm]
    print(f"\nscoring the best {len(top)} trials with seeds {', '.join(map(str, seeds))}")
    confirmed = confirm(name, top, columns, years)
    print(f"{'trial':>7} {'one seed':>9} {'five seeds (decides)':>24}")
    for r in confirmed:
        print(f"{r['trial'].number:>7} {r['trial'].value:>6.1f} MW {r['mean']:>14.1f} ± {r['std']:.1f} MW")
    best, settings = confirmed[0]["trial"], confirmed[0]["settings"]
    tuned_path.write_text(json.dumps(settings, indent=2) + "\n")
    print(f"\nbest: trial {best.number}, {confirmed[0]['mean']:.1f} MW over five seeds vs {first.value:.1f} MW with the "
          f"defaults, saved to {tuned_path}")
    edges = edge_settings(best.params, best.distributions, natural_ends)
    print("best settings in the outer 10% of their range (widen and rerun): "
          + (", ".join(f"{n} {v:.4g} ({side})" for n, v, side in edges) if edges else "none"))

    m, spread = save_final(name, df, calib, test, settings, columns, calibration_start, split_date)

    if len(done) >= 5:
        import matplotlib.pyplot as plt

        docs.mkdir(exist_ok=True)
        ax = optuna.visualization.matplotlib.plot_param_importances(study)
        ax.figure.set_size_inches(7, 4)
        ax.figure.tight_layout()
        ax.figure.savefig(docs / f"{name}_tuning.png", dpi=150, bbox_inches="tight")
        plt.close(ax.figure)

    log_run(experiment=f"demand-{name}-tuning", params={**settings, "lookback_days": days},
            metrics={"validation_loss": best.value, "validation_loss_5_seeds": confirmed[0]["mean"],
                     "validation_loss_5_seeds_std": confirmed[0]["std"], "default_validation_loss": first.value,
                     "trials": len(done),
                     "test_mape": m["mape"], "test_coverage": m["coverage"], "test_width": m["width"],
                     "test_mape_single_seed_mean": float(np.mean(spread)),
                     "test_mape_single_seed_std": float(np.std(spread, ddof=1))},
            files=[tuned_path, lookback_path])