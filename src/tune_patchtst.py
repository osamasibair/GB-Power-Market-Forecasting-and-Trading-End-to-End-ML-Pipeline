"""Tune PatchTST's size and training settings with optuna, on the two validation years (validation.py), so the test year
stays unseen. The design (head, RevIN, lags and the sunshine and Christmas inputs) is chosen first by controlled
comparisons in src/ablation.py and read from models/patchtst_design.json, and this search starts from it.

Each trial trains once for each validation year on the data before it, with early stopping on that year, and is scored
by its best pinball loss averaged over the two years. A trial whose loss is worse than the median of earlier trials at
the same epoch is stopped early (median pruning). The first trial is the default settings with the chosen design, so
the gain from tuning is visible. The study is saved in models/optuna_patchtst_2years.db, so it can be stopped with Ctrl+C
and resumed by running this again, and the best settings (design included) are written to models/patchtst_tuned.json
for train_patchtst.py --tuned and ablation.py --tuned.
    python src/tune_patchtst.py --trials 40"""
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import optuna

from ablation import design_path
from db import load_table
from tracking import log_run
from train_patchtst import models, tuned_path
from validation import patchtst_years

docs = Path("docs")
storage = f"sqlite:///{models / 'optuna_patchtst_2years.db'}"
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
search = ["learning_rate", "weight_decay", "dropout", "d_model", "layers", "hidden", "patch_len", "batch_size"]
steps_per_year = 1000  # epochs of the second year are reported after those of the first, so the pruner compares like with like


def suggest(trial):
    """The settings to try in this trial. Patches always overlap by half, as in the defaults."""
    patch_len = trial.suggest_categorical("patch_len", [12, 24, 48])
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True),
        "dropout": trial.suggest_float("dropout", 0.0, 0.3),
        "d_model": trial.suggest_categorical("d_model", [32, 64, 128]),
        "layers": trial.suggest_int("layers", 1, 3),
        "hidden": trial.suggest_categorical("hidden", [64, 128, 256]),
        "patch_len": patch_len,
        "stride": patch_len // 2,
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
    }


def load_design():
    if not design_path.exists():
        raise SystemExit(f"{design_path} not found: compare the design settings with src/ablation.py, "
                         "then run python src/ablation.py --decide")
    return json.loads(design_path.read_text())


def objective(trial, df, design):
    def report(year, epoch, loss):
        trial.report(loss, year * steps_per_year + epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    years = patchtst_years(df, {**design, **suggest(trial)}, report)
    for i, year in enumerate(years, start=1):
        trial.set_user_attr(f"loss_year{i}", year["loss"])
        trial.set_user_attr(f"mape_year{i}", year["mape"])
        trial.set_user_attr(f"best_epoch_year{i}", year["best_epoch"])
    return float(np.mean([year["loss"] for year in years]))


def best_settings(trial, design):
    return {**design, **trial.params, "stride": trial.params["patch_len"] // 2}


def describe(trial):
    p, a = trial.params, trial.user_attrs
    return (f"{trial.number:>5} {trial.value:>8.4f} {a.get('loss_year1', float('nan')):>8.4f} {a.get('loss_year2', float('nan')):>8.4f} "
            f"{(a.get('mape_year1', float('nan')) + a.get('mape_year2', float('nan'))) / 2:>6.2f}% "
            f"{p['learning_rate']:>8.1e} {p['dropout']:>7.2f} {p['d_model']:>7} {p['layers']:>6} {p['hidden']:>6} "
            f"{p['patch_len']:>5} {p['batch_size']:>5}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trials", type=int, default=40, help="how many trials to reach in total, finished or stopped early "
                        "(default 40, the same budget as LightGBM)")
    parser.add_argument("--timeout", type=float, help="stop starting new trials after this many minutes")
    args = parser.parse_args()

    from patchtst import defaults

    design = load_design()
    print(f"design from {design_path}: {design}")
    df = load_table("features")
    models.mkdir(exist_ok=True)
    study = optuna.create_study(
        study_name="patchtst_2years", storage=storage, load_if_exists=True, direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42), pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5))
    if not study.trials:
        study.enqueue_trial({name: defaults[name] for name in search})
    started = sum(t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED) for t in study.trials)
    try:
        study.optimize(lambda trial: objective(trial, df, design), n_trials=max(args.trials - started, 0),
                       timeout=args.timeout * 60 if args.timeout else None)
    except KeyboardInterrupt:
        print("\nstopped, keeping the trials finished so far (run again to continue)")

    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not done:
        raise SystemExit("no finished trials yet")
    pruned = sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials)
    print(f"\n{len(done)} finished trials, {pruned} stopped early")
    print(f"{'trial':>5} {'loss':>8} {'year 1':>8} {'year 2':>8} {'MAPE':>7} {'lr':>8} {'dropout':>7} {'d_model':>7} "
          f"{'layers':>6} {'hidden':>6} {'patch':>5} {'batch':>5}")
    if study.trials[0].state == optuna.trial.TrialState.COMPLETE:
        print(describe(study.trials[0]) + "   <- defaults")
    for trial in sorted(done, key=lambda t: t.value)[:5]:
        print(describe(trial))

    best = study.best_trial
    tuned_path.write_text(json.dumps(best_settings(best, design), indent=2) + "\n")
    print(f"\nbest: trial {best.number}, saved its settings (with the design) to {tuned_path}")

    if len(done) >= 5:
        import matplotlib.pyplot as plt

        docs.mkdir(exist_ok=True)
        ax = optuna.visualization.matplotlib.plot_param_importances(study)
        ax.figure.set_size_inches(7, 4)
        ax.figure.tight_layout()
        ax.figure.savefig(docs / "patchtst_tuning.png", dpi=150, bbox_inches="tight")
        plt.close(ax.figure)
        print("saved docs/patchtst_tuning.png")

    first = study.trials[0]
    log_run(experiment="demand-patchtst-tuning", params=best_settings(best, design),
            metrics={"validation_loss": best.value, **{k: v for k, v in best.user_attrs.items()},
                     "default_validation_loss": first.value if first.value is not None else float("nan"),
                     "trials": len(done)},
            files=[tuned_path])