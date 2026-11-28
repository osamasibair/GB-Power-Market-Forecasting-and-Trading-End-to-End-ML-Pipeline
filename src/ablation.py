"""Compare PatchTST's settings on the two validation years (validation.py) one at a time, for the write-up: train it once
for each value of one setting, everything else unchanged, with several seeds, and score each run by its pinball loss in
MW on each validation year (each year forecast by a model trained only on the data before it, for its set number of
epochs), averaged over both years and the seeds. It chooses nothing: every PatchTST setting is chosen by
src/tune_patchtst.py, as every model's are by its own tuning. Results go to the transformer_ablation table (one row per
setting, value and starting point) and to the MLflow experiment demand-patchtst-ablation. The test year is never used.
    python src/ablation.py pooling flatten mean attention --tuned"""
import argparse

import pandas as pd

from db import load_table, save_table
from tracking import log_run
from train_patchtst import base_settings, seeds, tuned_path
from validation import patchtst_years


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("setting", nargs="?", help="the PatchTST setting to vary, e.g. pooling")
    parser.add_argument("values", nargs="*", help="the values to try")
    parser.add_argument("--seeds", nargs="+", type=int, default=seeds,
                        help="train once per seed and average (default 42 1 2 3 4)")
    parser.add_argument("--tuned", action="store_true", help=f"start from the settings in {tuned_path}")
    args = parser.parse_args()

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