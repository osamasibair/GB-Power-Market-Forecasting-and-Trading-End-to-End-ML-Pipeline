"""Choose and tune PatchTST's settings on the two validation years (validation.py), so the test year stays unseen, with
the same steps, budget and rules as every model. One command runs all the steps, in this order:
1. Lookback: 7 to 42 days (forecast_windows.lookbacks), as for every model, with the default settings and five seeds.
2. Settings: 60 optuna trials with that lookback, covering every other setting: the head (pooling), RevIN and lags along
   with the model's size and training settings, so PatchTST's choices all come from the same search budget as every
   model's. The trials run 4 at a time, like every model's (src/parallel.py).
3. Confirm: each trial uses one seed, so the best three are trained with the other four seeds too, and the one with the
   lowest loss averaged over all five wins.
4. Recheck: the lookbacks are compared again with the tuned settings (five seeds each) and the winner is kept; the
   settings aren't tuned again, so it's one pass, as for every model.
5-9. The final model, as for every model (src/final_model.py): the tuned settings trained with each of the five seeds up
   to the calibration year and up to the test year, the forecasts averaged, the 80% range calibrated on the calibration
   year, and the result added to transformer_forecasts and transformer_calibration as patchtst_tuned.
Every step's runs are split across the workers. Worker 0 also saves the lookback (models/patchtst_lookback.json) and the
settings (models/patchtst_tuned.json).

Each run trains once for each validation year on the data before it, for exactly its number of epochs, and is scored by
its pinball loss in MW averaged over the two years. The number of epochs is one of the settings searched, as the number
of trees is for the boosting models, so nothing is chosen by looking at the year being scored (no early stopping on it).
Every trial runs in full, as every other model's do: none is stopped early for looking worse than earlier trials. The
first trial run is the default settings, so the gain from tuning is visible. The study is kept in the shared database,
so it can be stopped with Ctrl+C and resumed by running this again. The rule, the same for every model
(validation.edge_settings): a range whose best setting comes out in its outer 10% is widened and the search rerun.
Widened four times, each by the measure in validation.edge_settings: after the first search (the largest model sizes,
dropout near its top), the second (the most layers, weight decay near its top), the third, the first run 4 trials at a
time (the longest patches, the largest hidden size), and the fourth (the longest patches again, the highest learning
rate, the smallest d_model, the smallest batch).
    PATCHTST_DEVICE=cuda python src/tune_patchtst.py --workers 4 --worker 0     (and --worker 1, 2, 3 at the same time)"""
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import optuna

import parallel
from db import load_table
from forecast_windows import lookbacks
from tracking import log_run
from train_patchtst import models, seeds, tuned_path
from validation import best_so_far, edge_settings, patchtst_years

docs = Path("docs")
lookback_path = models / "patchtst_lookback.json"
# a new study each time the search changes: earlier ones searched different ranges, were stopped early, used early
# stopping on the year being scored, didn't search the head, RevIN and lags, ran one trial at a time, or (all_4w) didn't
# offer patches of 96 or a hidden size of 1024, or (wide3_4w) patches of 192, a d_model of 16, batches of 32 or a learning
# rate above 0.003
search_round = "wide4_4w"
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
search = ["pooling", "revin", "lags", "learning_rate", "weight_decay", "dropout", "d_model", "layers", "hidden",
          "patch_len", "batch_size", "epochs"]
# ends of a range that can't go further: no dropout, one layer, (almost) no weight decay, patches of 192 (the longest
# that fit the shortest lookback, 7 days, twice over); and the head, RevIN and lags, whose choices are every option
natural_ends = {("dropout", "low"): 0.0, ("layers", "low"): 1, ("weight_decay", "low"): 1e-4, ("patch_len", "high"): 192,
                ("pooling", "low"): "flatten", ("pooling", "high"): "attention",
                ("revin", "low"): 0, ("revin", "high"): 1, ("lags", "low"): 0, ("lags", "high"): 1}


def suggest(trial):
    """The settings to try in this trial. Patches always overlap by half, as in the defaults."""
    patch_len = trial.suggest_categorical("patch_len", [12, 24, 48, 96, 192])
    return {
        "pooling": trial.suggest_categorical("pooling", ["flatten", "mean", "attention"]),
        "revin": trial.suggest_categorical("revin", [0, 1]),
        "lags": trial.suggest_categorical("lags", [0, 1]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-4, 1.0, log=True),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "d_model": trial.suggest_categorical("d_model", [16, 32, 64, 128, 256]),
        "layers": trial.suggest_int("layers", 1, 5),
        "hidden": trial.suggest_categorical("hidden", [64, 128, 256, 512, 1024]),
        "patch_len": patch_len,
        "stride": patch_len // 2,
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        "epochs": trial.suggest_int("epochs", 2, 40, log=True),
    }


def lookback_comparison(df, settings, board=None, tag="", worker=0, workers=1, years=None):
    """Every lookback trained with `settings` and each of the five seeds, split across the workers (parallel.shared):
    per year and on average (which decides), with the spread between seeds and the MAPE, best first. `years` (for
    tests) replaces patchtst_years."""
    years = years or patchtst_years
    jobs = [[days, seed] for days in lookbacks for seed in seeds]
    runs = parallel.shared(board, tag, jobs, lambda job: years(df, {**settings, "history_days": job[0], "seed": job[1]}),
                           worker, workers)
    results = []
    for days in lookbacks:
        losses = np.array([[year["loss"] for year in run] for job, run in zip(jobs, runs) if job[0] == days])
        mapes = np.array([[year["mape"] for year in run] for job, run in zip(jobs, runs) if job[0] == days])
        both = losses.mean(axis=1)
        results.append({"days": days, "loss_year1": float(losses[:, 0].mean()), "loss_year2": float(losses[:, 1].mean()),
                        "loss": float(both.mean()), "loss_std": float(both.std(ddof=1)), "mape": float(mapes.mean())})
    return sorted(results, key=lambda r: r["loss"])


def objective(trial, df, base):
    years = patchtst_years(df, {**base, **suggest(trial)})
    for i, year in enumerate(years, start=1):
        trial.set_user_attr(f"loss_year{i}", year["loss"])
        trial.set_user_attr(f"mape_year{i}", year["mape"])
    return float(np.mean([year["loss"] for year in years]))


def best_settings(trial, base):
    return {**base, **trial.params, "stride": trial.params["patch_len"] // 2}


def confirm(trials, base, df, years=None, board=None, tag="", worker=0, workers=1):
    """Train each of `trials` with every seed and return them with the loss averaged over the seeds, best first. Each
    trial's own run is its seed 42 result, so only the other seeds are trained, split across the workers
    (parallel.shared). `years` (for tests) replaces patchtst_years."""
    years = years or patchtst_years
    by_number = {trial.number: trial for trial in trials}
    jobs = [[trial.number, seed] for trial in trials for seed in seeds[1:]]
    runs = parallel.shared(board, tag, jobs, lambda job: float(np.mean(
        [year["loss"] for year in years(df, {**best_settings(by_number[job[0]], base), "seed": job[1]})])),
        worker, workers)
    results = []
    for trial in trials:
        losses = [trial.value] + [loss for job, loss in zip(jobs, runs) if job[0] == trial.number]
        results.append({"trial": trial, "losses": losses, "mean": float(np.mean(losses)),
                        "std": float(np.std(losses, ddof=1))})
    return sorted(results, key=lambda r: r["mean"])


def final_forecasts(df, settings, board=None, tag="", worker=0, workers=1):
    """Steps 5 to 9's training runs (src/final_model.py): each seed trained from the first training day to the
    calibration year, for exactly its number of epochs, to forecast it, and to the test year to forecast that, split
    across the workers; combined into the averaged forecasts and calibrated 80% range."""
    from final_model import combine, seed_runs
    from forecast_windows import (
        day_windows,
        first_training_day,
        longest_history,
        windows,
    )
    from patchtst import PatchTSTForecaster
    from train_patchtst import rows
    from validation import folds

    calibration_start, split_date = folds[-1]  # the calibration year is the second validation year, up to the test year
    target = "demand"
    ends = {"calibration": calibration_start, "test": split_date}
    days = {"calibration": day_windows(df.index, calibration_start, split_date), "test": day_windows(df.index, split_date)}

    def train_and_forecast(seed, part):
        model = PatchTSTForecaster(**{**settings, "seed": seed})
        model.fit(df, windows(df.index, first=first_training_day, last=ends[part], stride=model.settings["train_stride"],
                              history=longest_history), epochs=model.settings["epochs"])
        return model.predict(df, days[part]).reshape(-1, 3)

    calibration_runs, test_runs = seed_runs(train_and_forecast, seeds, board, tag, worker, workers)
    actual = {part: df[target].to_numpy()[rows(days[part])] for part in days}
    return combine("patchtst_tuned", calibration_runs, test_runs, actual["calibration"], actual["test"])


def describe(trial):
    p, a = trial.params, trial.user_attrs
    return (f"{trial.number:>5} {trial.value:>8.1f} {a.get('loss_year1', float('nan')):>8.1f} {a.get('loss_year2', float('nan')):>8.1f} "
            f"{(a.get('mape_year1', float('nan')) + a.get('mape_year2', float('nan'))) / 2:>6.2f}% "
            f"{p['pooling']:>9} {p['revin']:>5} {p['lags']:>4} "
            f"{p['learning_rate']:>8.1e} {p['dropout']:>7.2f} {p['d_model']:>7} {p['layers']:>6} {p['hidden']:>6} "
            f"{p['patch_len']:>5} {p['batch_size']:>5} {p['epochs']:>6}")


def print_lookbacks(title, results):
    print(f"\n{title}")
    print(f"{'lookback':>10} {'year 1':>10} {'year 2':>10} {'average (decides)':>22} {'MAPE':>7}")
    for r in results:
        print(f"{r['days']:>5} days {r['loss_year1']:>7.1f} MW {r['loss_year2']:>7.1f} MW "
              f"{r['loss']:>11.1f} ± {r['loss_std']:.1f} MW {r['mape']:>6.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trials", type=int, default=60,
                        help="how many finished trials, across all the workers (default 60, the same budget as every model)")
    parser.add_argument("--confirm", type=int, default=3,
                        help="how many of the best trials to train with every seed before choosing (default 3)")
    parser.add_argument("--workers", type=int, default=1, help="how many workers share the runs (4 for every model)")
    parser.add_argument("--worker", type=int, default=0, help="this worker's number, 0 to --workers - 1")
    args = parser.parse_args()
    worker, workers = args.worker, args.workers
    if not 0 <= worker < workers:
        raise SystemExit(f"--worker must be 0 to {workers - 1}")

    from patchtst import defaults

    df = load_table("features")
    models.mkdir(exist_ok=True)
    print(f"worker {worker} of {workers}")
    shared_db = parallel.storage(models / "optuna_patchtst.db")
    board = parallel.Board(models / "optuna_patchtst.db", f"patchtst_{search_round}")

    # 1. lookback, with the default settings
    results = lookback_comparison(df, {}, board, "lookback", worker, workers)
    print_lookbacks("1. patchtst lookback, default settings (5 seeds each):", results)
    days = results[0]["days"]
    print(f"\nchosen: {days} days, margin over the runner-up {results[1]['loss'] - results[0]['loss']:.1f} MW")
    base = {"history_days": days}

    # 2. the settings: the shared search
    study = optuna.create_study(study_name=f"patchtst_{days}d_{search_round}", storage=shared_db, load_if_exists=True,
                                direction="minimize", pruner=optuna.pruners.NopPruner())
    study.sampler = parallel.sampler(study, worker)
    if not study.trials and worker == 0:
        study.enqueue_trial({name: defaults[name] for name in search})
    print(f"\n2. patchtst: lookback {days} days, {args.trials} trials shared by {workers} worker(s)")
    try:
        done = parallel.search(study, lambda trial: objective(trial, df, base), args.trials)
    except KeyboardInterrupt:
        raise SystemExit("\nstopped, keeping the trials finished so far (run again to continue)")
    first = next((t for t in done if t.params == {name: defaults[name] for name in search}), None)
    print(f"\n{len(done)} finished trials")
    print(f"{'trial':>5} {'MW loss':>8} {'year 1':>8} {'year 2':>8} {'MAPE':>7} {'pooling':>9} {'revin':>5} {'lags':>4} "
          f"{'lr':>8} {'dropout':>7} {'d_model':>7} "
          f"{'layers':>6} {'hidden':>6} {'patch':>5} {'batch':>5} {'epochs':>6}")
    if first:
        print(describe(first) + "   <- defaults")
    for trial in sorted(done, key=lambda t: t.value)[:5]:
        print(describe(trial))
    print("best loss after: " + ", ".join(f"{n} trials {v:.1f} MW" for n, v in best_so_far([t.value for t in done])))

    # 3. confirm the best few with every seed
    top = sorted(done, key=lambda t: t.value)[:args.confirm]
    print(f"\n3. training the best {len(top)} trials with seeds {', '.join(map(str, seeds))}")
    confirmed = confirm(top, base, df, board=board, tag=f"confirm|{study.study_name}", worker=worker, workers=workers)
    print(f"\n{'trial':>5} {'one seed':>9} {'five seeds (decides)':>24}")
    for r in confirmed:
        print(f"{r['trial'].number:>5} {r['trial'].value:>6.1f} MW {r['mean']:>14.1f} ± {r['std']:.1f} MW")
    best = confirmed[0]["trial"]
    settings = best_settings(best, base)
    default_loss = first.value if first else float("nan")
    print(f"\nbest: trial {best.number}, {confirmed[0]['mean']:.1f} MW over five seeds vs {default_loss:.1f} MW with the "
          "defaults")
    edges = edge_settings(best.params, best.distributions, natural_ends)
    print("best settings in the outer 10% of their range (widen and rerun): "
          + (", ".join(f"{n} {v:.4g} ({side})" if isinstance(v, (int, float)) else f"{n} {v} ({side})"
                       for n, v, side in edges) if edges else "none"))

    # 4. recheck the lookback with the tuned settings
    tuned = {k: v for k, v in settings.items() if k != "history_days"}
    rechecked = lookback_comparison(df, tuned, board, f"recheck|{study.study_name}|{best.number}", worker, workers)
    print_lookbacks("4. patchtst lookback rechecked with the tuned settings (5 seeds each):", rechecked)
    final_days = rechecked[0]["days"]
    print(f"\nmargin over the runner-up: {rechecked[1]['loss'] - rechecked[0]['loss']:.1f} MW; the tuned model reads "
          f"{final_days} days" + (f" instead of {days}" if final_days != days else ""))
    settings = {**settings, "history_days": final_days}

    # 5-9. the final model: the ten runs split across the workers, then worker 0 saves it with the choices
    print(f"\n5-9. training the tuned settings with seeds {', '.join(map(str, seeds))} and averaging their forecasts")
    result = final_forecasts(df, settings, board, f"final|{study.study_name}|{best.number}|{final_days}", worker,
                             workers)
    if worker != 0:
        print(f"worker {worker} done: worker 0 saves the choices and the final model")
        raise SystemExit  # finished without an error, so a command chained after this one runs

    from final_model import save

    lookback_path.write_text(json.dumps({"lookback_days": final_days, "before_recheck": days}, indent=2) + "\n")
    tuned_path.write_text(json.dumps(settings, indent=2) + "\n")
    print(f"saved {lookback_path}: {lookback_path.read_text().strip()}")
    print(f"saved {tuned_path}: {tuned_path.read_text().strip()}")
    parallel.keep_file(board, lookback_path)
    parallel.keep_file(board, tuned_path)
    m = save("patchtst_tuned", result)
    spread = result["seed_mapes"]

    import matplotlib.pyplot as plt

    docs.mkdir(exist_ok=True)
    ax = optuna.visualization.matplotlib.plot_param_importances(study)
    ax.figure.set_size_inches(7, 4)
    ax.figure.tight_layout()
    ax.figure.savefig(docs / "patchtst_tuning.png", dpi=150, bbox_inches="tight")
    plt.close(ax.figure)

    log_run(experiment="demand-patchtst-tuning", params={**settings, "workers": workers},
            metrics={"validation_loss": best.value, "validation_loss_5_seeds": confirmed[0]["mean"],
                     "validation_loss_5_seeds_std": confirmed[0]["std"], **{k: v for k, v in best.user_attrs.items()},
                     "default_validation_loss": default_loss, "trials": len(done),
                     "test_mape": m["mape"], "test_coverage": m["coverage"], "test_width": m["width"],
                     "test_mape_single_seed_mean": float(np.mean(spread)),
                     "test_mape_single_seed_std": float(np.std(spread, ddof=1))},
            files=[tuned_path, lookback_path])
