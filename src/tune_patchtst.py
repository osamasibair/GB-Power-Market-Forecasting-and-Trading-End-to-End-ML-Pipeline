"""Tune PatchTST's settings with optuna, on the two validation years (validation.py), so the test year stays unseen.
The lookback (history_days) is chosen first with the default settings, as for every model (src/ablation.py), and read
from models/patchtst_lookback.json. Every other setting is searched here: the head (pooling), RevIN and lags along with
the model's size and training settings, so PatchTST's choices all come from the same 40 trials as every model's.

Each trial trains once for each validation year on the data before it, for exactly its number of epochs, and is scored
by its pinball loss in MW averaged over the two years. The number of epochs is one of the settings searched, as the
number of trees is for the boosting models, so nothing is chosen by looking at the year being scored (no early stopping
on it). Every trial runs in full, as every other model's do: none is stopped early for looking worse than earlier
trials partway through. The first trial is the default settings with the chosen lookback, so the gain from tuning is
visible. The study is saved in models/optuna_patchtst_all.db, so it can be stopped with Ctrl+C and resumed by running
this again. The rule, the same for every model (validation.edge_settings): a range whose best setting comes out in its
outer 10% is widened and the search rerun. Widened twice: after the first search (the largest model sizes, dropout near
its top) and the second (the most layers, weight decay near its top).

Each trial is trained with one seed, and one seed moves the loss by several MW, more than the gaps between the best
trials. So the best few trials are then trained with the other four seeds too, and the one with the lowest loss averaged
over all five seeds wins. Its settings (lookback included) are written to models/patchtst_tuned.json for
train_patchtst.py --tuned and the lookback recheck (src/ablation.py --tuned).
    python src/tune_patchtst.py --trials 40"""
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import optuna

from ablation import lookback_path, seeds
from db import load_table
from tracking import log_run
from train_patchtst import models, tuned_path
from validation import edge_settings, patchtst_years

docs = Path("docs")
# a new study each time the search changes: earlier trials searched different ranges, were stopped early, used early
# stopping on the year being scored or didn't search the head, RevIN and lags
storage = f"sqlite:///{models / 'optuna_patchtst_all.db'}"
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
search = ["pooling", "revin", "lags", "learning_rate", "weight_decay", "dropout", "d_model", "layers", "hidden",
          "patch_len", "batch_size", "epochs"]
# ends of a range that can't go further: no dropout, one layer, (almost) no weight decay; and the head, RevIN and lags,
# whose choices are every option there is
natural_ends = {("dropout", "low"): 0.0, ("layers", "low"): 1, ("weight_decay", "low"): 1e-4,
                ("pooling", "low"): "flatten", ("pooling", "high"): "attention",
                ("revin", "low"): 0, ("revin", "high"): 1, ("lags", "low"): 0, ("lags", "high"): 1}


def suggest(trial):
    """The settings to try in this trial. Patches always overlap by half, as in the defaults."""
    patch_len = trial.suggest_categorical("patch_len", [12, 24, 48])
    return {
        "pooling": trial.suggest_categorical("pooling", ["flatten", "mean", "attention"]),
        "revin": trial.suggest_categorical("revin", [0, 1]),
        "lags": trial.suggest_categorical("lags", [0, 1]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-4, 1.0, log=True),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "d_model": trial.suggest_categorical("d_model", [32, 64, 128, 256]),
        "layers": trial.suggest_int("layers", 1, 5),
        "hidden": trial.suggest_categorical("hidden", [64, 128, 256, 512]),
        "patch_len": patch_len,
        "stride": patch_len // 2,
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
        "epochs": trial.suggest_int("epochs", 2, 40, log=True),
    }


def load_lookback():
    """The lookback chosen with the default settings, as the setting PatchTST reads it by."""
    if not lookback_path.exists():
        raise SystemExit(f"{lookback_path} not found: compare the lookbacks with python src/ablation.py history_days "
                         "7 14 21 28 42, then run python src/ablation.py --decide")
    return {"history_days": json.loads(lookback_path.read_text())["lookback_days"]}


def objective(trial, df, base):
    years = patchtst_years(df, {**base, **suggest(trial)})
    for i, year in enumerate(years, start=1):
        trial.set_user_attr(f"loss_year{i}", year["loss"])
        trial.set_user_attr(f"mape_year{i}", year["mape"])
    return float(np.mean([year["loss"] for year in years]))


def best_settings(trial, base):
    return {**base, **trial.params, "stride": trial.params["patch_len"] // 2}


def confirm(trials, base, df, years=None):
    """Train each of `trials` with every seed and return them with the loss averaged over the seeds, best first. Each
    trial's own run is its seed 42 result, so only the other seeds are trained. `years` (for tests) replaces
    patchtst_years."""
    years = years or patchtst_years
    results = []
    for trial in trials:
        losses = [trial.value]
        for seed in seeds[1:]:
            print(f"\ntrial {trial.number}, seed {seed}")
            runs = years(df, {**best_settings(trial, base), "seed": seed})
            losses.append(float(np.mean([year["loss"] for year in runs])))
        results.append({"trial": trial, "losses": losses, "mean": float(np.mean(losses)),
                        "std": float(np.std(losses, ddof=1))})
    return sorted(results, key=lambda r: r["mean"])


def describe(trial):
    p, a = trial.params, trial.user_attrs
    return (f"{trial.number:>5} {trial.value:>8.1f} {a.get('loss_year1', float('nan')):>8.1f} {a.get('loss_year2', float('nan')):>8.1f} "
            f"{(a.get('mape_year1', float('nan')) + a.get('mape_year2', float('nan'))) / 2:>6.2f}% "
            f"{p['pooling']:>9} {p['revin']:>5} {p['lags']:>4} "
            f"{p['learning_rate']:>8.1e} {p['dropout']:>7.2f} {p['d_model']:>7} {p['layers']:>6} {p['hidden']:>6} "
            f"{p['patch_len']:>5} {p['batch_size']:>5} {p['epochs']:>6}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trials", type=int, default=40,
                        help="how many finished trials to reach in total (default 40, the same budget as every model)")
    parser.add_argument("--timeout", type=float, help="stop starting new trials after this many minutes")
    parser.add_argument("--confirm", type=int, default=3,
                        help="how many of the best trials to train with every seed before choosing (default 3)")
    args = parser.parse_args()

    from patchtst import defaults

    base = load_lookback()
    print(f"lookback from {lookback_path}: {base['history_days']} days")
    df = load_table("features")
    models.mkdir(exist_ok=True)
    study = optuna.create_study(
        study_name="patchtst_all", storage=storage, load_if_exists=True, direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42), pruner=optuna.pruners.NopPruner())
    # a resumed study gets a new sampler seed: the same seed would repeat the settings already tried
    study.sampler = optuna.samplers.TPESampler(seed=42 + len(study.trials))
    if not study.trials:
        study.enqueue_trial({name: defaults[name] for name in search})
    started = sum(t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED) for t in study.trials)
    try:
        study.optimize(lambda trial: objective(trial, df, base), n_trials=max(args.trials - started, 0),
                       timeout=args.timeout * 60 if args.timeout else None)
    except KeyboardInterrupt:
        print("\nstopped, keeping the trials finished so far (run again to continue)")

    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not done:
        raise SystemExit("no finished trials yet")
    pruned = sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials)
    print(f"\n{len(done)} finished trials, {pruned} stopped early")
    print(f"{'trial':>5} {'MW loss':>8} {'year 1':>8} {'year 2':>8} {'MAPE':>7} {'pooling':>9} {'revin':>5} {'lags':>4} "
          f"{'lr':>8} {'dropout':>7} {'d_model':>7} "
          f"{'layers':>6} {'hidden':>6} {'patch':>5} {'batch':>5} {'epochs':>6}")
    if study.trials[0].state == optuna.trial.TrialState.COMPLETE:
        print(describe(study.trials[0]) + "   <- defaults")
    for trial in sorted(done, key=lambda t: t.value)[:5]:
        print(describe(trial))

    top = sorted(done, key=lambda t: t.value)[:args.confirm]
    print(f"\ntraining the best {len(top)} trials with seeds {', '.join(map(str, seeds))}")
    confirmed = confirm(top, base, df)
    print(f"\n{'trial':>5} {'one seed':>9} {'five seeds (decides)':>24}")
    for r in confirmed:
        print(f"{r['trial'].number:>5} {r['trial'].value:>6.1f} MW {r['mean']:>14.1f} ± {r['std']:.1f} MW")
    best = confirmed[0]["trial"]
    tuned_path.write_text(json.dumps(best_settings(best, base), indent=2) + "\n")
    print(f"\nbest: trial {best.number}, {confirmed[0]['mean']:.1f} MW over five seeds, saved its settings (with the "
          f"lookback) to {tuned_path}")
    edges = edge_settings(best.params, best.distributions, natural_ends)
    print("best settings in the outer 10% of their range (widen and rerun): "
          + (", ".join(f"{n} {v:.4g} ({side})" if isinstance(v, (int, float)) else f"{n} {v} ({side})"
                       for n, v, side in edges) if edges else "none"))

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
    log_run(experiment="demand-patchtst-tuning", params=best_settings(best, base),
            metrics={"validation_loss": best.value, "validation_loss_5_seeds": confirmed[0]["mean"],
                     "validation_loss_5_seeds_std": confirmed[0]["std"], **{k: v for k, v in best.user_attrs.items()},
                     "default_validation_loss": first.value if first.value is not None else float("nan"),
                     "trials": len(done)},
            files=[tuned_path])