"""Choose PatchTST's design on the two validation years (validation.py), one setting at a time: train it once for each
value of one setting, everything else unchanged, with several seeds, and score each run by its best pinball loss on
each validation year (each year forecast by a model trained only on the data before it). The value with the lowest
loss averaged over both years and the seeds wins. Results go to the transformer_ablation table (one row per setting and
value) and to the MLflow experiment demand-patchtst-ablation. The test year is never used here.
    python src/ablation.py pooling flatten mean attention --seeds 42 1 2 3 4
After the design settings have been compared (pooling, revin, lags), choose them all and save them to
models/patchtst_design.json for src/tune_patchtst.py:
    python src/ablation.py --decide"""
import argparse
import json
from pathlib import Path

import pandas as pd

from db import load_table, save_table
from tracking import log_run
from train_patchtst import base_settings
from validation import patchtst_years

# the sunshine and Christmas inputs aren't chosen here: every model gets the same inputs, chosen on the validation years
# by LightGBM's feature check (src/tune_lightgbm.py --check-features), so differences between models come from the models
design_settings = ["pooling", "revin", "lags"]
design_path = Path("models") / "patchtst_design.json"
seeds = [42, 1, 2, 3, 4]
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
    """One run's results from patchtst_years: each year's loss, MAPE and best epoch, and their averages."""
    row = {"parameters": years[0]["parameters"]}
    for i, year in enumerate(years, start=1):
        row.update({f"loss_year{i}": year["loss"], f"mape_year{i}": year["mape"], f"best_epoch_year{i}": year["best_epoch"]})
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
    """Replace the rows for the setting just tested and keep the rows for other settings."""
    if old is None:
        return new
    old = old[old["setting"] != new["setting"].iloc[0]]
    return pd.concat([old, new], ignore_index=True)


def decide(table, settings=design_settings, min_seeds=min_seeds):
    """For each design setting, the value with the lowest validation loss, from runs that started from the defaults
    (not the tuned settings) with at least min_seeds seeds. Also returns, for each setting, the winner's margin over the
    runner-up and the larger of their seed spreads, so a choice closer than the noise is visible."""
    chosen, report = {}, []
    for setting in settings:
        rows = table[(table["setting"] == setting) & (~table["tuned"].astype(bool)) & (table["seeds"] >= min_seeds)]
        if "validation_loss" not in rows or len(rows) < 2:
            raise SystemExit(f"no complete comparison for {setting}: run python src/ablation.py {setting} ... "
                             f"--seeds {' '.join(map(str, seeds))}")
        rows = rows.sort_values("validation_loss")
        best, second = rows.iloc[0], rows.iloc[1]
        chosen[setting] = parse(str(best["value"]))
        noise = max(best["validation_loss_std"], second["validation_loss_std"])
        report.append({"setting": setting, "chosen": best["value"], "runner_up": second["value"],
                       "margin": second["validation_loss"] - best["validation_loss"], "seed_spread": noise})
    return chosen, pd.DataFrame(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("setting", nargs="?", help="the PatchTST setting to vary, e.g. pooling")
    parser.add_argument("values", nargs="*", help="the values to try")
    parser.add_argument("--seeds", nargs="+", type=int, default=seeds,
                        help="train once per seed and average (default 42 1 2 3 4)")
    parser.add_argument("--tuned", action="store_true", help="start from the settings in models/patchtst_tuned.json")
    parser.add_argument("--from-design", action="store_true",
                        help=f"start from the design in {design_path} instead of the defaults, to recheck one choice "
                             "given the others")
    parser.add_argument("--decide", action="store_true",
                        help=f"choose {', '.join(design_settings)} from the saved comparisons and save them to {design_path}")
    args = parser.parse_args()

    if args.decide:
        chosen, report = decide(load_table("transformer_ablation"))
        print(report.round(4).to_string(index=False))
        close = report[report["margin"] < report["seed_spread"]]
        if len(close):
            print(f"\ncloser than the spread between seeds (still decided by the average): {', '.join(close['setting'])}")
        design_path.parent.mkdir(exist_ok=True)
        design_path.write_text(json.dumps(chosen, indent=2) + "\n")
        print(f"\nsaved {chosen} to {design_path}")
        raise SystemExit

    from patchtst import defaults

    if args.setting not in defaults or not args.values:
        raise SystemExit(f"give a setting and its values, e.g. pooling flatten mean attention; settings: {sorted(defaults)}")

    df = load_table("features")
    base = base_settings(args.tuned)
    if args.from_design:
        if not design_path.exists():
            raise SystemExit(f"{design_path} not found, run python src/ablation.py --decide first")
        base = {**base, **json.loads(design_path.read_text())}
        print(f"starting from the design in {design_path}: { {k: base[k] for k in design_settings} }")
    results = []
    for value in map(parse, args.values):
        runs = []
        for seed in args.seeds:
            print(f"\n{args.setting} = {value}, seed {seed}")
            settings = run_settings(base, args.setting, value, seed)
            run = run_row(patchtst_years(df, settings))
            runs.append(run)
            log_run(experiment="demand-patchtst-ablation", params={**settings, "setting": args.setting}, metrics=run, files=[])
        results.append({"setting": args.setting, "value": str(value), "tuned": args.tuned,
                        "from_design": args.from_design, **average(runs)})

    table = pd.DataFrame(results)
    print(f"\n{args.setting:>12} {'parameters':>11} {'loss year 1':>12} {'loss year 2':>12} {'average (decides)':>22} "
f"{'MAPE':>15}   ({len(args.seeds)} seed{'s' if len(args.seeds) > 1 else ''}{', tuned' if args.tuned else ''}"
          f"{', from the design' if args.from_design else ''})")
    for _, r in table.iterrows():
        print(f"{r['value']:>12} {r['parameters']:>11,.0f} {r['loss_year1']:>12.4f} {r['loss_year2']:>12.4f} "
              f"{r['validation_loss']:>13.4f} ± {r['validation_loss_std']:.4f} {r['mape']:>7.2f}% ± {r['mape_std']:.2f}")

    try:
        old = load_table("transformer_ablation").reset_index(drop=True)
    except ValueError:
        old = None
    save_table(merge(old, table), "transformer_ablation")
    print("saved transformer_ablation")