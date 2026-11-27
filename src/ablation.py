"""Compare PatchTST's settings on the two validation years (validation.py), one setting at a time: train it once for
each value of one setting, everything else unchanged, with several seeds, and score each run by its pinball loss in MW
on each validation year (each year forecast by a model trained only on the data before it, for its set number of
epochs), averaged over both years and the seeds. Results go to the transformer_ablation table (one row per setting,
value and starting point) and to the MLflow experiment demand-patchtst-ablation. The test year is never used here.

Only the lookback (history_days) is chosen here, as for every model: first with the default settings (step 1), then
rechecked with the tuned ones (step 3). Every other setting, pooling, RevIN and lags included, is chosen in the 40
tuning trials (src/tune_patchtst.py), the same budget as every model. Comparisons of other settings are only for the
write-up: they show what each choice does and choose nothing.
1. Lookback with the default settings, saved to models/patchtst_lookback.json for src/tune_patchtst.py:
    python src/ablation.py history_days 7 14 21 28 42
    python src/ablation.py --decide
3. Recheck with the tuned settings: if a different lookback wins, it's saved to models/patchtst_lookback.json and
   models/patchtst_tuned.json; the settings aren't tuned again, so it's one pass, as for every model.
    python src/ablation.py history_days 7 14 21 28 42 --tuned
    python src/ablation.py --decide --tuned"""
import argparse
import json
from pathlib import Path

import pandas as pd

from db import load_table, save_table
from tracking import log_run
from train_patchtst import base_settings, seeds, tuned_path
from validation import patchtst_years

lookback_path = Path("models") / "patchtst_lookback.json"
min_seeds = len(seeds)


def dates():
    """The calibration and test dates. Read in a separate process because the modules that hold them import LightGBM."""
    from quantiles import calibration_start
    from train import split_date, target

    return calibration_start, split_date, target


def parse(value):
    for kind in (int, float):
        try:
            return kind(value)
        except ValueError:
            pass
    return value


def run_settings(base, setting, value, seed):
    """The settings for one run: the base settings with one setting changed. Patches always overlap by half."""
    settings = {**base, setting: value, "seed": seed}
    if setting == "patch_len":
        settings["stride"] = value // 2
    return settings


def run_row(years):
    """One run's results from patchtst_years: each year's loss and MAPE, their averages, and the number of epochs."""
    row = {"parameters": years[0]["parameters"]}
    for i, year in enumerate(years, start=1):
        row.update({f"loss_year{i}": year["loss"], f"mape_year{i}": year["mape"]})
    row["epochs"] = years[0]["epochs"]
    row["validation_loss"] = sum(y["loss"] for y in years) / len(years)
    row["mape"] = sum(y["mape"] for y in years) / len(years)
    return row


def average(runs):
    """One row from several seeds: the mean of every number, plus the spread between seeds of the validation loss
    (which decides) and of the MAPE."""
    frame = pd.DataFrame(runs)
    row = frame.mean(numeric_only=True).to_dict()
    for column in ("validation_loss", "mape"):
        row[f"{column}_std"] = float(frame[column].std()) if len(frame) > 1 else 0.0
    row["seeds"] = len(frame)
    return row


def merge(old, new):
    """Replace the rows for the setting just tested from the same starting point (defaults or tuned settings) and keep
    every other row."""
    if old is None:
        return new
    same = (old["setting"] == new["setting"].iloc[0]) & (old["tuned"].astype(bool) == bool(new["tuned"].iloc[0]))
    return pd.concat([old[~same], new], ignore_index=True)


def decide(table, setting="history_days", tuned=False, min_seeds=min_seeds):
    """The value of `setting` with the lowest validation loss, from runs that started from the defaults (or, with tuned,
    from the tuned settings) with at least min_seeds seeds. Also returns the winner's margin over the runner-up and the
    larger of their seed spreads, so a choice closer than the noise is visible."""
    rows = table[(table["setting"] == setting) & (table["tuned"].astype(bool) == tuned) & (table["seeds"] >= min_seeds)]
    if "validation_loss" not in rows or len(rows) < 2:
        raise SystemExit(f"no complete comparison for {setting}: run python src/ablation.py {setting} ... "
                         f"{'--tuned ' if tuned else ''}--seeds {' '.join(map(str, seeds))}")
    rows = rows.sort_values("validation_loss")
    best, second = rows.iloc[0], rows.iloc[1]
    return parse(str(best["value"])), {"runner_up": second["value"],
                                       "margin": second["validation_loss"] - best["validation_loss"],
                                       "seed_spread": max(best["validation_loss_std"], second["validation_loss_std"])}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("setting", nargs="?", help="the PatchTST setting to vary, e.g. pooling")
    parser.add_argument("values", nargs="*", help="the values to try")
    parser.add_argument("--seeds", nargs="+", type=int, default=seeds,
                        help="train once per seed and average (default 42 1 2 3 4)")
    parser.add_argument("--tuned", action="store_true", help=f"start from the settings in {tuned_path}")
    parser.add_argument("--decide", action="store_true",
                        help=f"choose the lookback from the saved comparison and save it to {lookback_path} (with "
                             f"--tuned: the recheck, which also updates {tuned_path})")
    args = parser.parse_args()

    if args.decide:
        days, report = decide(load_table("transformer_ablation"), tuned=args.tuned)
        print(f"lookback {days} days, runner-up {report['runner_up']} days, margin {report['margin']:.1f} MW, "
              f"spread between seeds {report['seed_spread']:.1f} MW")
        if report["margin"] < report["seed_spread"]:
            print("closer than the spread between seeds (still decided by the average)")
        lookback_path.parent.mkdir(exist_ok=True)
        if not args.tuned:
            lookback_path.write_text(json.dumps({"lookback_days": days}, indent=2) + "\n")
            print(f"saved {days} days to {lookback_path}")
            raise SystemExit
        tuned = json.loads(tuned_path.read_text())
        if tuned["history_days"] == days:
            print(f"the tuned model keeps {days} days")
            raise SystemExit
        lookback_path.write_text(json.dumps({"lookback_days": days, "before_recheck": tuned["history_days"]}, indent=2)
                                 + "\n")
        print(f"the tuned model now reads {days} days instead of {tuned['history_days']}, saved to {lookback_path} "
              f"and {tuned_path}")
        tuned_path.write_text(json.dumps({**tuned, "history_days": days}, indent=2) + "\n")
        raise SystemExit

    from patchtst import defaults

    if args.setting not in defaults or not args.values:
        raise SystemExit(f"give a setting and its values, e.g. pooling flatten mean attention; settings: {sorted(defaults)}")

    df = load_table("features")
    base = base_settings(args.tuned)
    results = []
    for value in map(parse, args.values):
        runs = []
        for seed in args.seeds:
            print(f"\n{args.setting} = {value}, seed {seed}")
            settings = run_settings(base, args.setting, value, seed)
            run = run_row(patchtst_years(df, settings))
            runs.append(run)
            log_run(experiment="demand-patchtst-ablation", params={**settings, "setting": args.setting}, metrics=run, files=[])
        results.append({"setting": args.setting, "value": str(value), "tuned": args.tuned, **average(runs)})

    table = pd.DataFrame(results)
    print(f"\n{args.setting:>12} {'parameters':>11} {'loss year 1':>12} {'loss year 2':>12} {'average (decides)':>22} "
          f"{'MAPE':>15}   ({len(args.seeds)} seed{'s' if len(args.seeds) > 1 else ''}"
          f"{', from the tuned settings' if args.tuned else ', from the defaults'})")
    for _, r in table.iterrows():
        print(f"{r['value']:>12} {r['parameters']:>11,.0f} {r['loss_year1']:>9.1f} MW {r['loss_year2']:>9.1f} MW "
              f"{r['validation_loss']:>10.1f} ± {r['validation_loss_std']:.1f} MW {r['mape']:>7.2f}% ± {r['mape_std']:.2f}")

    try:
        old = load_table("transformer_ablation").reset_index(drop=True)
    except ValueError:
        old = None
    save_table(merge(old, table), "transformer_ablation")
    print("saved transformer_ablation")