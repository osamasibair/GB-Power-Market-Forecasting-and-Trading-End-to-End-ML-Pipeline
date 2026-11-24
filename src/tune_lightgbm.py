"""Tune LightGBM's settings with optuna, with the same budget and rules as PatchTST (src/tune_patchtst.py), so the
comparison between gradient boosting and deep learning is fair, and choose its optional features first. Every choice is
made on the two validation years in validation.py (each forecast by a model trained only on the data before it, and
scored by the average of the two years), so the test year stays unseen.

Two steps, in this order:
1. Features. The sunshine (shortwave_radiation) and Christmas (is_christmas) features were added to train.py after
   looking at test year scores, so the validation years decide whether to keep them, in a controlled comparison: all
   four combinations, with train.py's settings, scored by their MAPE averaged over both validation years and five seeds.
   The best combination is saved to models/lightgbm_features.json. The same comparison with the previously tuned
   settings (if there are any) is printed too, to show whether the choice depends on the settings.
       python src/tune_lightgbm.py --check-features
2. Settings. 40 optuna trials with those features, each scored by its MAPE averaged over both validation years. The
   first trial is train.py's settings, so the gain from tuning is visible. LightGBM trains in seconds, so trials aren't
   pruned. Each choice of features has its own study in models/, so it can be
   stopped with Ctrl+C and resumed by running this again. The best settings are written to models/lightgbm_tuned.json.
       python src/tune_lightgbm.py

The tuned model is then trained with five random seeds, with the same 80% range as LightGBM in src/train_patchtst.py
(q10 and q90 models with the tuned settings, calibrated on the calibration year). The seed 42 forecasts are added to
transformer_forecasts and transformer_calibration as lightgbm_tuned, so run src/train_patchtst.py first, then this."""
import argparse
import itertools
import json
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna

from evaluate import metrics
from quantiles import calibration_start, quantiles
from train import features, split_date, target
from validation import before, fold_rows, folds

models = Path("models")
docs = Path("docs")
tuned_path = models / "lightgbm_tuned.json"
features_path = models / "lightgbm_features.json"
seeds = [42, 1, 2, 3, 4]
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
# the settings src/train.py uses: LightGBM's defaults, apart from the number of trees and the learning rate
defaults = {"n_estimators": 500, "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 20,
            "subsample": 1.0, "colsample_bytree": 1.0, "reg_lambda": 0.0}
optional = {"solar": "shortwave_radiation", "christmas": "is_christmas"}


def suggest(trial):
    """The settings to try in this trial."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 3000, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 200, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
    }


def used_features(choice):
    """train.py's features, without the optional ones switched off in `choice`, e.g. {"solar": 1, "christmas": 0}."""
    off = {column for name, column in optional.items() if not choice[name]}
    return [f for f in features if f not in off]


def storage_for(choice):
    """Each choice of features gets its own study, scored on both validation years."""
    tag = "".join(f"_no_{name}" for name, on in choice.items() if not on)
    return f"sqlite:///{models / f'optuna_lightgbm_2years{tag}.db'}"


def make_model(settings, seed=42, quantile=None):
    """A LightGBM model with `settings`. Bagging happens every iteration, which does nothing when subsample is 1."""
    extra = {"objective": "quantile", "alpha": quantile} if quantile is not None else {}
    return lgb.LGBMRegressor(**settings, **extra, subsample_freq=1, random_state=seed, verbose=-1)


def year_mape(settings, used, fit_part, year, seed=42):
    """MAPE on one validation year of a model trained on fit_part, the data before it."""
    model = make_model(settings, seed).fit(fit_part[used], fit_part[target])
    return float(metrics(year[target].to_numpy(), model.predict(year[used]))["mape"])


def validation_mapes(settings, used, years, seed=42):
    """MAPE on each validation year; years is a list of (data before the year, the year's forecast rows)."""
    return [year_mape(settings, used, fit_part, year, seed) for fit_part, year in years]


def check_features(settings, years, seeds=seeds):
    """For every combination of the optional features: the MAPE on each validation year averaged over the seeds, the
    average of the two years (which decides), and the spread of that average between seeds."""
    results = []
    for values in itertools.product([1, 0], repeat=len(optional)):
        choice = dict(zip(optional, values))
        runs = np.array([validation_mapes(settings, used_features(choice), years, seed) for seed in seeds])
        both = runs.mean(axis=1)
        results.append({**choice, **{f"year{i + 1}": float(runs[:, i].mean()) for i in range(len(years))},
                        "mape": float(both.mean()), "mape_std": float(both.std(ddof=1))})
    return results


def tuned_forecasts(df, calib, test, settings, seed, used, calibration_start=calibration_start, split_date=split_date):
    """The tuned model's forecasts and 80% range on the calibration and test years, made exactly like LightGBM's in
    src/train_patchtst.py: fitted before the calibration year to forecast it and calibrate the range, then refitted on
    everything before the test year to forecast that."""
    from train_patchtst import conformal_adjustment

    early_train = df[df.index < calibration_start]
    train = df[df.index < split_date]

    def fit(part):
        point = make_model(settings, seed).fit(part[used], part[target])
        q = {q: make_model(settings, seed, q).fit(part[used], part[target]) for q in quantiles if q != 0.5}
        return point, q

    early_point, early_q = fit(early_train)
    early = early_point.predict(calib[used])
    early_low, early_high = early_q[0.1].predict(calib[used]), early_q[0.9].predict(calib[used])
    adjustment = conformal_adjustment(calib[target].to_numpy(), early_low, early_high)
    final_point, final_q = fit(train)
    point = final_point.predict(test[used])
    low = np.minimum(final_q[0.1].predict(test[used]) - adjustment, point)
    high = np.maximum(final_q[0.9].predict(test[used]) + adjustment, point)
    return {
        "calibration": {"lightgbm_tuned": early, "lightgbm_tuned_low": np.minimum(early_low - adjustment, early),
                        "lightgbm_tuned_high": np.maximum(early_high + adjustment, early)},
        "forecasts": {"lightgbm_tuned": point, "lightgbm_tuned_low": low, "lightgbm_tuned_high": high},
    }


def describe(trial):
    p = trial.params
    years = " ".join(f"{trial.user_attrs.get(f'year{i}', float('nan')):>6.2f}%" for i in (1, 2))
    return (f"{trial.number:>5} {trial.value:>6.2f}% {years} {p['n_estimators']:>6} {p['learning_rate']:>7.3f} {p['num_leaves']:>6} "
            f"{p['min_child_samples']:>9} {p['subsample']:>9.2f} {p['colsample_bytree']:>8.2f} {p['reg_lambda']:>7.2f}")


def print_check(title, results):
    print(f"\n{title}")
    print(f"{'sunshine':>9} {'christmas':>9} {'year 1':>8} {'year 2':>8} {'average (decides)':>20}")
    for r in sorted(results, key=lambda r: r["mape"]):
        print(f"{'yes' if r['solar'] else 'no':>9} {'yes' if r['christmas'] else 'no':>9} {r['year1']:>7.3f}% "
              f"{r['year2']:>7.3f}% {r['mape']:>11.3f}% ± {r['mape_std']:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check-features", action="store_true",
                        help="step 1: choose the optional features on both validation years, then stop")
    parser.add_argument("--trials", type=int, default=40,
                        help="how many finished trials to reach in total (default 40, the same budget as PatchTST)")
    args = parser.parse_args()

    from db import load_table

    df = load_table("features")
    from forecast_windows import day_windows
    from train_patchtst import rows

    calib = df.iloc[rows(day_windows(df.index, calibration_start, split_date))]
    test = df.iloc[rows(day_windows(df.index, split_date))]
    years = [(before(df, start), fold_rows(df, start, end)) for start, end in folds]
    span = ", ".join(f"{start} to {end} ({len(year) // 48} days)" for (start, end), (_, year) in zip(folds, years))
    models.mkdir(exist_ok=True)

    if args.check_features:
        print(f"scored on the validation years {span}, each by a model trained on the data before it, "
              f"{len(seeds)} seeds each")
        chosen_by = check_features(defaults, years)
        print_check("with train.py's settings (this decides):", chosen_by)
        if tuned_path.exists():
            previous = {k: v for k, v in json.loads(tuned_path.read_text()).items() if k in defaults}
            print_check(f"with the settings in {tuned_path} (a check that the choice doesn't depend on the settings):",
                        check_features(previous, years))
        best = min(chosen_by, key=lambda r: r["mape"])
        choice = {name: int(best[name]) for name in optional}
        features_path.write_text(json.dumps(choice, indent=2) + "\n")
        print(f"\nchosen: {', '.join(f'{name} {'yes' if on else 'no'}' for name, on in choice.items())}, "
              f"saved to {features_path}")
        print(f"features: {', '.join(used_features(choice))}")
        raise SystemExit

    if not features_path.exists():
        raise SystemExit(f"{features_path} not found, run python src/tune_lightgbm.py --check-features first")
    choice = json.loads(features_path.read_text())
    used = used_features(choice)

    from chronos2 import add_columns, scored
    from db import save_table
    from tracking import log_run

    study = optuna.create_study(study_name="lightgbm", storage=storage_for(choice), load_if_exists=True,
                                direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    if not study.trials:
        study.enqueue_trial(defaults)
    done_before = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    print(f"features: {', '.join(used)}")
    print(f"tuning on the validation years {span}, each trial scored by the average of the two")

    def objective(trial):
        mapes = validation_mapes(suggest(trial), used, years)
        for i, value in enumerate(mapes):
            trial.set_user_attr(f"year{i + 1}", value)
        return float(np.mean(mapes))

    try:
        study.optimize(objective, n_trials=max(args.trials - done_before, 0))
    except KeyboardInterrupt:
        print("\nstopped, keeping the trials finished so far (run again to continue)")

    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not done:
        raise SystemExit("no finished trials yet")
    print(f"\n{len(done)} finished trials")
    print(f"{'trial':>5} {'MAPE':>7} {'year 1':>7} {'year 2':>7} {'trees':>6} {'lr':>7} {'leaves':>6} {'min child':>9} {'subsample':>9} {'columns':>8} {'lambda':>7}")
    print(describe(study.trials[0]) + "   <- train.py's settings")
    for trial in sorted(done, key=lambda t: t.value)[:5]:
        print(describe(trial))

    best = study.best_trial
    settings = {k: best.params[k] for k in defaults}
    tuned_path.write_text(json.dumps(settings, indent=2) + "\n")
    print(f"\nbest: trial {best.number}, validation MAPE {best.value:.2f}% vs {study.trials[0].value:.2f}% with train.py's "
          f"settings, saved to {tuned_path}")

    print(f"\ntraining the tuned settings with seeds {', '.join(map(str, seeds))} (point, q10 and q90 models, twice each)")
    forecasts_table = load_table("transformer_forecasts")
    calibration_table = load_table("transformer_calibration")
    runs = {}
    for seed in seeds:
        result = tuned_forecasts(df, calib, test, settings, seed, used)
        if seed == seeds[0]:
            first = result
        runs[seed] = scored(add_columns(forecasts_table, result["forecasts"]), ["lightgbm_tuned"])["lightgbm_tuned"]
        m = runs[seed]
        print(f"  seed {seed:>2}: test MAPE {m['mape']:.2f}%, coverage {m['coverage']:.1f}%, width {m['width']:.0f} MW")

    forecasts_table = add_columns(forecasts_table, first["forecasts"])
    calibration_table = add_columns(calibration_table, first["calibration"])
    save_table(forecasts_table, "transformer_forecasts")
    save_table(calibration_table, "transformer_calibration")
    print("added lightgbm_tuned (seed 42) to transformer_forecasts and transformer_calibration")

    mapes = [runs[s]["mape"] for s in seeds]
    everything = scored(forecasts_table, ["baseline", "lightgbm", "lightgbm_tuned"])
    print(f"\n{'model':15} {'MAE':>7} {'RMSE':>7} {'MAPE':>7} {'in range':>9} {'width':>8}")
    for name, m in everything.items():
        interval = f"{m['coverage']:8.1f}% {m['width']:6.0f} MW" if "coverage" in m else f"{'-':>9} {'-':>8}"
        print(f"{name:15} {m['mae']:7.0f} {m['rmse']:7.0f} {m['mape']:6.2f}% {interval}")
    print(f"\ntuned LightGBM over {len(seeds)} seeds: test MAPE {np.mean(mapes):.2f}% ± {np.std(mapes, ddof=1):.2f}")

    if len(done) >= 5:
        import matplotlib.pyplot as plt

        docs.mkdir(exist_ok=True)
        ax = optuna.visualization.matplotlib.plot_param_importances(study)
        ax.figure.set_size_inches(7, 4)
        ax.figure.tight_layout()
        ax.figure.savefig(docs / "lightgbm_tuning.png", dpi=150, bbox_inches="tight")
        plt.close(ax.figure)
        print("saved docs/lightgbm_tuning.png")

    log_run(experiment="demand-lightgbm-tuning", params={**settings, **choice},
            metrics={"validation_mape": best.value, "default_validation_mape": study.trials[0].value, "trials": len(done),
                     **{f"validation_mape_year{i}": best.user_attrs[f"year{i}"] for i in (1, 2)},
                     "test_mape": everything["lightgbm_tuned"]["mape"], "test_mape_mean": float(np.mean(mapes)),
                     "test_mape_std": float(np.std(mapes, ddof=1)), "test_coverage": everything["lightgbm_tuned"]["coverage"]},
            files=[tuned_path, features_path])