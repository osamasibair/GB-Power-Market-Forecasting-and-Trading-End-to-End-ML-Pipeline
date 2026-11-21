"""Chronos-2, Amazon's pretrained time series model, which unlike Chronos-Bolt can also read covariates. Three versions
are compared with the other models on the same days:
- zero shot, demand only (like the Chronos-Bolt comparison)
- zero shot, with tomorrow's weather and calendar as known future covariates
- fine tuned on this data, with the same covariates

Every version sees what the live job has at 08:00 the day before: demand up to then, plus the weather and calendar for
the 16 hour gap and the forecast day. Each forecasts the 80 half hours from 08:00 and keeps the last 48. Every version
gets the same conformal calibration as the other models. The fine-tuned version is fitted twice, like PatchTST: on data
before the calibration year (giving the calibration-year forecasts the calibration and the ensemble weights use), then
on all data before the test year (giving the test forecasts).

Results are added as columns to the transformer_forecasts and transformer_calibration tables, so run
src/train_patchtst.py first (it rewrites those tables), then this.
    python src/chronos2.py                  # all three versions
    python src/chronos2.py --steps 20       # a quick check that fine-tuning runs, before the full run
    python src/chronos2.py --skip-finetune  # only the two zero shot versions
Fine-tuning a 120M parameter model needs a real GPU, so it can also run on another machine (such as Google Colab)
that has no access to the database. Export the inputs, fine-tune there, and import the forecasts:
    python src/chronos2.py --export chronos2_input.csv.gz                                   # here
    python src/chronos2.py --from-file chronos2_input.csv.gz --to-file chronos2_finetuned.csv.gz   # on the GPU
    python src/chronos2.py --import chronos2_finetuned.csv.gz                               # here again
To measure how much fine tuning varies between runs, repeat it with other random seeds and import each run under its own
name, which adds it as extra columns without replacing the main one:
    python src/chronos2.py --from-file chronos2_input.csv.gz --to-file chronos2_finetuned_seed1.csv.gz --seed 1
    python src/chronos2.py --import chronos2_finetuned_seed1.csv.gz --name chronos2_finetuned_seed1"""""
import argparse
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from forecast_windows import (
    covariate_matrix,
    covariates,
    cycles,
    day_windows,
    gap_steps,
    horizon,
)

model_name = "amazon/chronos-2"
names = covariates + [f"{kind}_{name}" for name in cycles for kind in ("sin", "cos")]
steps_ahead = gap_steps + horizon  # the 16 hour gap plus tomorrow
quantile_levels = [0.1, 0.5, 0.9]
versions = {
    "chronos2": ("Chronos-2, zero shot, demand only", False, False),
    "chronos2_covariates": ("Chronos-2, zero shot, with weather and calendar", True, False),
    "chronos2_finetuned": ("Chronos-2, fine-tuned, with weather and calendar", True, True),
}


def regular_grid(df):
    """Demand and covariates on an unbroken half hourly grid, so that a difference in position is a difference in time.
    Missing demand stays NaN, which Chronos-2 handles; missing covariates are filled from their neighbours."""
    full = pd.date_range(df.index.min(), df.index.max(), freq="30min")
    cov = pd.DataFrame(covariate_matrix(df), index=df.index, columns=names).reindex(full)
    cov = cov.interpolate(limit_direction="both")
    return df["demand"].reindex(full).to_numpy(np.float32), cov.to_numpy(np.float32), full


def day_inputs(demand, cov, positions, context, use_covariates):
    """One input per forecast day starting at `positions`: demand up to the 08:00 cutoff, and with covariates their past
    values plus their values for the next 80 half hours (the gap and the day), which are known in advance."""
    inputs = []
    for p in positions:
        cut = p - gap_steps
        first = max(0, cut - context)
        item = {"target": demand[first:cut]}
        if use_covariates:
            item["past_covariates"] = {name: cov[first:cut, i] for i, name in enumerate(names)}
            item["future_covariates"] = {name: cov[cut:p + horizon, i] for i, name in enumerate(names)}
        inputs.append(item)
    return inputs


def forecast(pipeline, demand, cov, positions, context, use_covariates):
    """Forecasts in MW for each day, shape (days, 48, 3): the 10%, 50% and 90% quantiles of tomorrow's half hours."""
    inputs = day_inputs(demand, cov, positions, context, use_covariates)
    predicted, _ = pipeline.predict_quantiles(inputs, prediction_length=steps_ahead, quantile_levels=quantile_levels)
    # sorted so the 10% forecast can never sit above the median or the median above the 90%
    return np.sort(np.stack([p[0, gap_steps:, :].numpy() for p in predicted]), axis=-1)


def training_series(demand, cov, end):
    """All history before position `end` as one long series. Fine-tuning samples random windows from it."""
    return [{"target": demand[:end], "past_covariates": {name: cov[:end, i] for i, name in enumerate(names)}}]


def fine_tune(pipeline, demand, cov, end, context, steps, learning_rate, batch_size, accumulate, seed, output_dir):
    """A fine tuned copy of `pipeline`, trained only on data before position `end`. Every covariate is marked as known
    in advance, as weather forecasts and the calendar are. Gradients from `accumulate` batches are added up before each
    update, which trains like a batch that many times bigger without needing the memory for it. `seed` sets which
    training windows are sampled and the dropout, so runs with different seeds show how much fine tuning varies."""
    from chronos.chronos2.preprocess import from_list_of_dicts

    inputs = from_list_of_dicts(training_series(demand, cov, end), prediction_length=steps_ahead, known_covariates_names=names)
    return pipeline.fit(inputs, prediction_length=steps_ahead, context_length=context, num_steps=steps,
                        learning_rate=learning_rate, batch_size=batch_size, output_dir=str(output_dir),
                        gradient_accumulation_steps=accumulate, seed=seed)


def load_pipeline(device):
    import torch
    from chronos import BaseChronosPipeline

    print(f"loading {model_name} on {device}")
    return BaseChronosPipeline.from_pretrained(model_name, device_map=device, torch_dtype=torch.float32)


def fine_tuned_forecasts(pretrained, demand, cov, calib_positions, test_positions, args):
    """Fine tune twice, then forecast the calibration days with the model trained before them and the test days with the
    model trained on everything before the test year. Each model only sees data up to its first forecast's 08:00 cutoff."""
    settings = (args.context, args.steps, args.learning_rate, args.batch_size, args.accumulate, args.seed)
    print(f"fine-tuning for {args.steps} steps with seed {args.seed} on data before the calibration year")
    early = fine_tune(pretrained, demand, cov, calib_positions[0] - gap_steps, *settings, Path("models") / "chronos2-early")
    print("fine-tuning again on all data before the test year")
    final = fine_tune(pretrained, demand, cov, test_positions[0] - gap_steps, *settings, Path("models") / "chronos2-final")
    return (forecast(early, demand, cov, calib_positions, args.context, True),
            forecast(final, demand, cov, test_positions, args.context, True))


def export(path, demand, cov, grid, calib_positions, test_positions):
    """Everything fine tuning needs, for running it on a machine with a GPU: the half hourly grid of demand and
    covariates, with the start of every calibration and test day marked."""
    frame = pd.DataFrame(cov, index=pd.Index(grid, name="timestamp"), columns=names)
    frame.insert(0, "demand", demand)
    frame["role"] = ""
    frame.iloc[calib_positions, frame.columns.get_loc("role")] = "calibration"
    frame.iloc[test_positions, frame.columns.get_loc("role")] = "test"
    frame.to_csv(path)


def load_export(path):
    frame = pd.read_csv(path, index_col="timestamp", parse_dates=True)
    role = frame["role"].fillna("")
    return (frame["demand"].to_numpy(np.float32), frame[names].to_numpy(np.float32), frame.index,
            np.flatnonzero(role == "calibration"), np.flatnonzero(role == "test"))


def to_rows(forecasts, grid, positions, role):
    """Day forecasts of shape (days, 48, 3) as one row per half hour, with its timestamp."""
    quantiles = forecasts.reshape(-1, 3)
    times = grid[(np.asarray(positions)[:, None] + np.arange(horizon)).ravel()]
    return pd.DataFrame({"role": role, "low": quantiles[:, 0], "median": quantiles[:, 1], "high": quantiles[:, 2]},
                        index=pd.Index(times, name="timestamp"))


def read_forecasts(path, calib_times, test_times):
    """The calibration and test forecasts made on another machine, checked against the days this machine expects."""
    frame = pd.read_csv(path, index_col="timestamp", parse_dates=True)
    out = []
    for role, expected in (("calibration", calib_times), ("test", test_times)):
        part = frame[frame["role"] == role]
        if not part.index.equals(pd.DatetimeIndex(expected)):
            raise SystemExit(f"the {role} days in {path} don't match the database, export the inputs again")
        out.append(part[["low", "median", "high"]].to_numpy())
    return out


def add_columns(table, forecasts):
    """The saved forecast table with this model's columns added, replacing any from an earlier run."""
    table = table.copy()
    for name, values in forecasts.items():
        if len(values) != len(table):
            raise SystemExit(f"{name} has {len(values)} rows but the table has {len(table)}, run src/train_patchtst.py again")
        table[name] = values
    return table


def scored(table, keys):
    from train_patchtst import summary
    actual = table["actual"].to_numpy()
    out = {}
    for key in keys:
        if key in table:
            low, high = (table[f"{key}_low"].to_numpy(), table[f"{key}_high"].to_numpy()) if f"{key}_low" in table else (None, None)
            out[key] = summary(actual, table[key].to_numpy(), low, high)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--context", type=int, default=1024, help="half hours of past demand each forecast sees (default 1024, about 3 weeks)")
    parser.add_argument("--steps", type=int, default=500, help="fine-tuning steps (default 500)")
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="fine-tuning learning rate (default 1e-5)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="fine-tuning batch size, counting each covariate as a series, so 32 is about 3 windows (default 32)")
    parser.add_argument("--accumulate", type=int, default=4,
                        help="batches whose gradients are added up before each update, so 32 x 4 trains like 128 (default 4)")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for fine-tuning (default 42, the Hugging Face default the first run used)")
    parser.add_argument("--device", default=None, help="cpu, mps (Apple GPU) or cuda (default: cuda or mps if there is one, otherwise cpu)")
    parser.add_argument("--skip-finetune", action="store_true", help="only run the two zero shot versions")
    parser.add_argument("--export", metavar="FILE", help="save the inputs for fine-tuning on another machine, then stop")
    parser.add_argument("--from-file", metavar="FILE", help="on the GPU machine: fine-tune from an exported file, no database needed")
    parser.add_argument("--to-file", metavar="FILE", default="chronos2_finetuned.csv.gz", help="where --from-file saves the forecasts")
    parser.add_argument("--import", dest="import_file", metavar="FILE", help="add fine-tuned forecasts made on another machine")
    parser.add_argument("--name", default="chronos2_finetuned",
                        help="column name for --import, such as chronos2_finetuned_seed1 for a run with another seed (default chronos2_finetuned)")
    args = parser.parse_args()

    if args.from_file:
        import torch

        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        demand, cov, grid, calib_positions, test_positions = load_export(args.from_file)
        started = time.perf_counter()
        calib, test = fine_tuned_forecasts(load_pipeline(device), demand, cov, calib_positions, test_positions, args)
        pd.concat([to_rows(calib, grid, calib_positions, "calibration"), to_rows(test, grid, test_positions, "test")]).to_csv(args.to_file)
        raise SystemExit(f"saved {args.to_file} ({time.perf_counter() - started:.0f}s), import it with --import {args.to_file}")

    from ablation import dates
    from db import load_table, save_table
    from tracking import log_run
    from train_patchtst import conformal_adjustment, rows, summary

    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as pool:
        calibration_start, split_date, target = pool.submit(dates).result()
    df = load_table("features")
    calib_days = day_windows(df.index, calibration_start, split_date)
    test_days = day_windows(df.index, split_date)
    demand, cov, grid = regular_grid(df)
    calib_positions = grid.get_indexer(df.index[calib_days])
    test_positions = grid.get_indexer(df.index[test_days])
    actual_calib = df[target].to_numpy()[rows(calib_days)]
    actual_test = df[target].to_numpy()[rows(test_days)]

    if args.export:
        export(args.export, demand, cov, grid, calib_positions, test_positions)
        raise SystemExit(f"saved {args.export} ({len(calib_days)} calibration and {len(test_days)} test days)")

    raw = {}
    if args.import_file:
        raw[args.name] = read_forecasts(args.import_file, df.index[rows(calib_days)], df.index[rows(test_days)])
        device = "imported"
    else:
        import torch

        device = args.device or ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        pretrained = load_pipeline(device)
        for key, (label, use_covariates, tuned) in versions.items():
            started = time.perf_counter()
            if tuned:
                if args.skip_finetune:
                    continue
                calib, test = fine_tuned_forecasts(pretrained, demand, cov, calib_positions, test_positions, args)
            else:
                print(f"{key}: forecasting {len(calib_days)} calibration days and {len(test_days)} test days")
                calib = forecast(pretrained, demand, cov, calib_positions, args.context, use_covariates)
                test = forecast(pretrained, demand, cov, test_positions, args.context, use_covariates)
            raw[key] = (calib.reshape(-1, 3), test.reshape(-1, 3))
            print(f"{key}: done ({time.perf_counter() - started:.0f}s)")

    test_forecasts, calib_forecasts, results = {}, {}, {}
    for key, (calib, test) in raw.items():
        adjustment = conformal_adjustment(actual_calib, calib[:, 0], calib[:, 2])
        calib_forecasts.update({key: calib[:, 1], f"{key}_low": calib[:, 0] - adjustment, f"{key}_high": calib[:, 2] + adjustment})
        test_forecasts.update({key: test[:, 1], f"{key}_low": test[:, 0] - adjustment, f"{key}_high": test[:, 2] + adjustment})
        results[key] = summary(actual_test, test[:, 1], test[:, 0] - adjustment, test[:, 2] + adjustment)

    forecasts_table = add_columns(load_table("transformer_forecasts"), test_forecasts)
    calibration_table = add_columns(load_table("transformer_calibration"), calib_forecasts)
    save_table(forecasts_table, "transformer_forecasts")
    save_table(calibration_table, "transformer_calibration")
    print(f"added {', '.join(raw)} to transformer_forecasts and transformer_calibration")

    seeds = sorted(c for c in forecasts_table if c.startswith("chronos2_finetuned_seed") and not c.endswith(("_low", "_high")))
    everything = scored(forecasts_table, ["baseline", "chronos", *versions, *seeds, "patchtst", "lightgbm"])
    print(f"\n{'model':22} {'MAE':>7} {'RMSE':>7} {'MAPE':>7} {'in range':>9} {'width':>8}")
    for name, m in everything.items():
        interval = f"{m['coverage']:8.1f}% {m['width']:6.0f} MW" if "coverage" in m else f"{'-':>9} {'-':>8}"
        print(f"{name:22} {m['mae']:7.0f} {m['rmse']:7.0f} {m['mape']:6.2f}% {interval}")

    log_run(experiment="demand-chronos2",
            params={"model": model_name, "context": args.context, "steps": args.steps, "learning_rate": args.learning_rate,
                    "batch_size": args.batch_size, "accumulate": args.accumulate, "device": device},
            metrics={f"{name}_{k}": v for name, m in results.items() for k, v in m.items()}, files=[])

    runs = [everything[k]["mape"] for k in ["chronos2_finetuned", *seeds] if k in everything]
    if len(runs) > 1:
        print(f"\nfine-tuned Chronos-2 over {len(runs)} seeds: test MAPE {np.mean(runs):.2f}% ± {np.std(runs, ddof=1):.2f}"
              f" (runs: {', '.join(f'{m:.2f}%' for m in runs)})")