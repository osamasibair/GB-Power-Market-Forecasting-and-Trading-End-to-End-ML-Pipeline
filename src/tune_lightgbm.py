"""Tune LightGBM's settings with optuna, with the same budget and rules as PatchTST (src/tune_patchtst.py), so the
comparison between gradient boosting and deep learning is fair: 40 trials, chosen on the calibration year only, so the
test year stays unseen. Each trial trains on the data before the calibration year and is scored by its MAPE on the
calibration year's forecast days. The first trial is the current settings in src/train.py, so the gain from tuning is
visible. LightGBM trains in seconds, so trials aren't pruned.

The study is saved in models/optuna_lightgbm.db, so it can be stopped with Ctrl+C and resumed by running this again,
and the best settings are written to models/lightgbm_tuned.json. The tuned settings are then trained with three random
seeds, with the same 80% range as LightGBM in src/train_patchtst.py (q10 and q90 models with the tuned settings,
calibrated on the calibration year). The seed 42 forecasts are added to transformer_forecasts and
transformer_calibration as lightgbm_tuned, so run src/train_patchtst.py first (it rewrites those tables), then this.
    python src/tune_lightgbm.py"""
import argparse
import json
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna

from evaluate import metrics
from quantiles import calibration_start, quantiles
from train import features, split_date, target

models = Path("models")
docs = Path("docs")
tuned_path = models / "lightgbm_tuned.json"
storage = f"sqlite:///{models / 'optuna_lightgbm.db'}"
seeds = [42, 1, 2]
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
# the settings src/train.py uses: LightGBM's defaults, apart from the number of trees and the learning rate
defaults = {"n_estimators": 500, "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 20,
            "subsample": 1.0, "colsample_bytree": 1.0, "reg_lambda": 0.0}
search = list(defaults)


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


def make_model(settings, seed=42, quantile=None):
    """A LightGBM model with `settings`. Bagging happens every iteration, which does nothing when subsample is 1."""
    extra = {"objective": "quantile", "alpha": quantile} if quantile is not None else {}
    return lgb.LGBMRegressor(**settings, **extra, subsample_freq=1, random_state=seed, verbose=-1)


def calibration_mape(settings, fit_part, calib, seed=42):
    model = make_model(settings, seed).fit(fit_part[features], fit_part[target])
    return float(metrics(calib[target].to_numpy(), model.predict(calib[features]))["mape"])


def objective(trial, fit_part, calib):
    return calibration_mape(suggest(trial), fit_part, calib)


def tuned_forecasts(df, calib, test, settings, seed, calibration_start=calibration_start, split_date=split_date):
    """The tuned model's forecasts and 80% range on the calibration and test years, made exactly like LightGBM's in
    src/train_patchtst.py: fitted before the calibration year to forecast it and calibrate the range, then refitted on
    everything before the test year to forecast that."""
    from train_patchtst import conformal_adjustment

    early_train = df[df.index < calibration_start]
    train = df[df.index < split_date]

    def fit(part):
        point = make_model(settings, seed).fit(part[features], part[target])
        q = {q: make_model(settings, seed, q).fit(part[features], part[target]) for q in quantiles if q != 0.5}
        return point, q

    early_point, early_q = fit(early_train)
    early = early_point.predict(calib[features])
    early_low, early_high = early_q[0.1].predict(calib[features]), early_q[0.9].predict(calib[features])
    adjustment = conformal_adjustment(calib[target].to_numpy(), early_low, early_high)
    final_point, final_q = fit(train)
    point = final_point.predict(test[features])
    low = np.minimum(final_q[0.1].predict(test[features]) - adjustment, point)
    high = np.maximum(final_q[0.9].predict(test[features]) + adjustment, point)
    return {
        "calibration": {"lightgbm_tuned": early, "lightgbm_tuned_low": np.minimum(early_low - adjustment, early),
                        "lightgbm_tuned_high": np.maximum(early_high + adjustment, early)},
        "forecasts": {"lightgbm_tuned": point, "lightgbm_tuned_low": low, "lightgbm_tuned_high": high},
    }


def describe(trial):
    p = trial.params
    return (f"{trial.number:>5} {trial.value:>6.2f}% {p['n_estimators']:>6} {p['learning_rate']:>7.3f} {p['num_leaves']:>6} "
            f"{p['min_child_samples']:>9} {p['subsample']:>9.2f} {p['colsample_bytree']:>8.2f} {p['reg_lambda']:>7.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trials", type=int, default=40, help="how many finished trials to reach in total (default 40, the same budget as PatchTST)")
    args = parser.parse_args()

    from chronos2 import add_columns, scored
    from db import load_table, save_table
    from forecast_windows import day_windows
    from tracking import log_run
    from train_patchtst import rows

    df = load_table("features")
    calib = df.iloc[rows(day_windows(df.index, calibration_start, split_date))]
    test = df.iloc[rows(day_windows(df.index, split_date))]
    fit_part = df[df.index < calibration_start]

    models.mkdir(exist_ok=True)
    study = optuna.create_study(study_name="lightgbm", storage=storage, load_if_exists=True, direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    if not study.trials:
        study.enqueue_trial(defaults)
    done_before = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    print(f"tuning on {len(fit_part):,} half hours before {calibration_start}, scored on {len(calib) // 48} calibration days")
    try:
        study.optimize(lambda trial: objective(trial, fit_part, calib), n_trials=max(args.trials - done_before, 0))
    except KeyboardInterrupt:
        print("\nstopped, keeping the trials finished so far (run again to continue)")

    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not done:
        raise SystemExit("no finished trials yet")
    print(f"\n{len(done)} finished trials")
    print(f"{'trial':>5} {'MAPE':>7} {'trees':>6} {'lr':>7} {'leaves':>6} {'min child':>9} {'subsample':>9} {'columns':>8} {'lambda':>7}")
    print(describe(study.trials[0]) + "   <- current settings")
    for trial in sorted(done, key=lambda t: t.value)[:5]:
        print(describe(trial))

    best = study.best_trial
    tuned_path.write_text(json.dumps(best.params, indent=2) + "\n")
    print(f"\nbest: trial {best.number}, calibration MAPE {best.value:.2f}% vs {study.trials[0].value:.2f}% now, "
          f"saved its settings to {tuned_path}")

    print(f"\ntraining the tuned settings with seeds {', '.join(map(str, seeds))} (point, q10 and q90 models, twice each)")
    forecasts_table = load_table("transformer_forecasts")
    calibration_table = load_table("transformer_calibration")
    runs = {}
    for seed in seeds:
        result = tuned_forecasts(df, calib, test, best.params, seed)
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

    log_run(experiment="demand-lightgbm-tuning", params=best.params,
            metrics={"calibration_mape": best.value, "default_calibration_mape": study.trials[0].value, "trials": len(done),
                     "test_mape": everything["lightgbm_tuned"]["mape"], "test_mape_mean": float(np.mean(mapes)),
                     "test_mape_std": float(np.std(mapes, ddof=1)), "test_coverage": everything["lightgbm_tuned"]["coverage"]},
            files=[tuned_path])