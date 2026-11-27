"""Compare the PatchTST transformer with LightGBM and a pretrained Chronos model on the same test days. LightGBM is scored in a separate process that never loads pytorch, so each library brings its own openmp threading
library, and on macOS having both in one process makes it crash or freeze."""
import argparse
import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from db import load_table, save_table
from evaluate import metrics
from forecast_windows import (
    day_windows,
    first_training_day,
    horizon,
    longest_history,
    windows,
)
from tracking import log_run

models = Path("models")
chronos_model = "amazon/chronos-bolt-small"

tuned_path = models / "patchtst_tuned.json"
seeds = [42, 1, 2, 3, 4]


def base_settings(tuned):
    """The PatchTST settings to start from: the defaults, or with `tuned` the settings optuna chose."""
    if not tuned:
        return {}
    if not tuned_path.exists():
        raise SystemExit("models/patchtst_tuned.json not found, run src/tune_patchtst.py first")
    return json.loads(tuned_path.read_text())


def rows(starts):
    return (starts[:, None] + np.arange(horizon)).ravel()


def conformal_adjustment(actual, low, high, coverage=0.8):
    return float(np.quantile(np.maximum(low - actual, actual - high), coverage))


def summary(actual, middle, low=None, high=None):
    result = metrics(actual, middle)
    if low is not None:
        result["coverage"] = ((actual >= low) & (actual <= high)).mean() * 100
        result["width"] = (high - low).mean()
    return {name: float(value) for name, value in result.items()}


def lightgbm_part():
    """Score the baseline and LightGBM, and return the dates and target the comparison uses. Runs in its own process."""
    from quantiles import calibration_start, train_quantile_models
    from train import features, split_date, target, train_model

    df = load_table("features")
    calib = df.iloc[rows(day_windows(df.index, calibration_start, split_date))]
    test = df.iloc[rows(day_windows(df.index, split_date))]
    train = df[df.index < split_date]

    point = train_model(train).predict(test[features])
    early_train = train[train.index < calibration_start]
    early = train_quantile_models(early_train)
    early_low, early_high = early[0.1].predict(calib[features]), early[0.9].predict(calib[features])
    adjustment = conformal_adjustment(calib[target].to_numpy(), early_low, early_high)
    early_point = train_model(early_train).predict(calib[features])
    final = train_quantile_models(train)
    low = np.minimum(final[0.1].predict(test[features]) - adjustment, point)
    high = np.maximum(final[0.9].predict(test[features]) + adjustment, point)
    return {
        "split_date": split_date,
        "calibration_start": calibration_start,
        "target": target,
        "baseline": summary(test[target].to_numpy(), test["demand_lag_336"].to_numpy()),
        "lightgbm": summary(test[target].to_numpy(), point, low, high),
        "forecasts": {"baseline": test["demand_lag_336"].to_numpy(), "lightgbm": point, "lightgbm_low": low, "lightgbm_high": high},
        "calibration": {"lightgbm": early_point, "lightgbm_low": np.minimum(early_low - adjustment, early_point),
        "lightgbm_high": np.maximum(early_high + adjustment, early_point)},
    }


def patchtst(df, calib_days, test_days, actual_calib, actual_test, calibration_start, split_date, settings,
             seeds=(42,)):
    """PatchTST's forecasts and 80% range on the calibration and test years, made like every other model's: trained from
    the first training day to the calibration year to forecast it and calibrate the range, then to the test year to
    forecast that, each for exactly its number of epochs. With several seeds, each seed's 10/50/90% forecasts are
    averaged, as for every tuned model. Returns the scores, the first seed's final model and calibration-year learning
    curve (for the live forecasts and the figures), the range adjustment, the forecasts and each seed's test MAPE."""
    from patchtst import PatchTSTForecaster

    started = time.perf_counter()
    epochs = PatchTSTForecaster(**settings).settings["epochs"]
    early_runs, final_runs, models_kept, losses_kept = [], [], [], []
    for seed in seeds:
        early = PatchTSTForecaster(**{**settings, "seed": seed})
        print(f"patchtst, seed {seed}: training on data before the calibration year for {epochs} epochs (the calibration "
              "loss is only recorded, for the learning curve, not used to choose anything)")
        early.fit(df, windows(df.index, first=first_training_day, last=calibration_start,
                              stride=early.settings["train_stride"], history=longest_history),
                  valid=calib_days, epochs=epochs, verbose=True, keep_best=False)
        early_runs.append(early.predict(df, calib_days).reshape(-1, 3))
        losses_kept.append(early.valid_losses)

        print(f"patchtst, seed {seed}: training on all data before {split_date} for {epochs} epochs")
        final = PatchTSTForecaster(**{**settings, "seed": seed})
        final.fit(df, windows(df.index, first=first_training_day, last=split_date, stride=final.settings["train_stride"],
                              history=longest_history), epochs=epochs, verbose=True)
        final_runs.append(final.predict(df, test_days).reshape(-1, 3))
        models_kept.append(final)
    calib, test = np.mean(early_runs, axis=0), np.mean(final_runs, axis=0)
    adjustment = conformal_adjustment(actual_calib, calib[:, 0], calib[:, 2])
    print(f"patchtst: {len(seeds)} seed{'s' if len(seeds) > 1 else ''} trained in {time.perf_counter() - started:.0f}s, "
          f"conformal adjustment {adjustment:.0f} MW")

    def interval(forecast):
        # the same range as every model's: widened (or narrowed) by the adjustment, never past the median
        return {"patchtst": forecast[:, 1], "patchtst_low": np.minimum(forecast[:, 0] - adjustment, forecast[:, 1]),
                "patchtst_high": np.maximum(forecast[:, 2] + adjustment, forecast[:, 1])}

    forecasts, calibration = interval(test), interval(calib)
    result = summary(actual_test, forecasts["patchtst"], forecasts["patchtst_low"], forecasts["patchtst_high"])
    seed_mapes = [float(metrics(actual_test, run[:, 1])["mape"]) for run in final_runs]
    return result, models_kept[0], adjustment, losses_kept[0], forecasts, calibration, seed_mapes



def chronos(df, calib_days, test_days, actual_calib, actual_test):
    try:
        import torch
        from chronos import BaseChronosPipeline

        from patchtst import chronos_forecast
    except ImportError:
        print("chronos forecasting isnt installed, so the Chronos comparison is skipped")
        return None
    print(f"chronos: forecasting {len(calib_days) + len(test_days)} days with {chronos_model}")
    pipeline = BaseChronosPipeline.from_pretrained(chronos_model, device_map="cpu", torch_dtype=torch.float32)
    calib = chronos_forecast(df, calib_days, pipeline).reshape(-1, 3)
    adjustment = conformal_adjustment(actual_calib, calib[:, 0], calib[:, 2])
    test = chronos_forecast(df, test_days, pipeline).reshape(-1, 3)
    forecasts = {"chronos": test[:, 1], "chronos_low": test[:, 0] - adjustment, "chronos_high": test[:, 2] + adjustment} 
    return summary(actual_test, forecasts["chronos"], forecasts["chronos_low"], forecasts["chronos_high"]), forecasts  


def keep_other_models(table, name):
    """`table` with the columns of models this script doesn't make (Chronos-2, ridge, tuned LightGBM), carried over from
    the saved table `name`, so retraining PatchTST doesn't mean rerunning or re-importing them."""
    try:
        old = load_table(name)
    except ValueError:
        return table
    if not old.index.equals(table.index):
        print(f"{name} covers different days, so the other models' columns are dropped: rerun them")
        return table
    for column in old.columns:
        if column not in table:
            table[column] = old[column].to_numpy()
    return table


def forecast_table(index, actual, forecasts):  
    """Every model's test-period forecasts in one table, for the dashboard.""" 
    return pd.DataFrame({"actual": actual, **forecasts}, index=index) 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-chronos", action="store_true", help="don't run the pretrained Chronos comparison")
    parser.add_argument("--tuned", action="store_true", help="train with the settings in models/patchtst_tuned.json")
    args = parser.parse_args()

    print("lightgbm: scoring the baseline and LightGBM in a separate process")
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as pool:
        first = pool.submit(lightgbm_part).result()
    split_date, calibration_start, target = first["split_date"], first["calibration_start"], first["target"]
    results = {"baseline": first["baseline"], "lightgbm": first["lightgbm"]}
    forecasts = dict(first["forecasts"])
    print(f"lightgbm: done, MAPE {results['lightgbm']['mape']:.2f}%")

    df = load_table("features")
    calib_days = day_windows(df.index, calibration_start, split_date)
    test_days = day_windows(df.index, split_date)
    actual_calib = df[target].to_numpy()[rows(calib_days)]
    actual_test = df[target].to_numpy()[rows(test_days)]
    print(f"calibration: {len(calib_days)} days from {calibration_start}, test: {len(test_days)} days from {split_date}")

    # the tuned model is a final model, so it's trained with five seeds and averaged, as every model's final model is
    results["patchtst"], model, adjustment, early_losses, patchtst_forecasts, patchtst_calibration, seed_mapes = patchtst(
        df, calib_days, test_days, actual_calib, actual_test, calibration_start, split_date, base_settings(args.tuned),
        seeds=seeds if args.tuned else (42,))
    if len(seed_mapes) > 1:
        print(f"patchtst ({len(seed_mapes)} seeds averaged): test MAPE {results['patchtst']['mape']:.2f}%; single seeds "
              f"{np.mean(seed_mapes):.2f}% ± {np.std(seed_mapes, ddof=1):.2f}")
    forecasts.update(patchtst_forecasts)
    if not args.skip_chronos:
        chronos_result = chronos(df, calib_days, test_days, actual_calib, actual_test)
        if chronos_result is not None:
            results["chronos"], chronos_forecasts = chronos_result 
            forecasts.update(chronos_forecasts) 

    print(f"\n{'model':10} {'MAE':>7} {'RMSE':>7} {'MAPE':>7} {'in range':>9} {'width':>8}")
    for name, m in results.items():
        interval = f"{m['coverage']:8.1f}% {m['width']:6.0f} MW" if "coverage" in m else f"{'-':>9} {'-':>8}"
        print(f"{name:10} {m['mae']:7.0f} {m['rmse']:7.0f} {m['mape']:6.2f}% {interval}")

    models.mkdir(exist_ok=True)
    model.save(models / "patchtst.pt", adjustment=adjustment, valid_losses=early_losses, split_date=split_date)
    print("saved models/patchtst.pt")

    save_table(keep_other_models(forecast_table(df.index[rows(test_days)], actual_test, forecasts), "transformer_forecasts"),
               "transformer_forecasts")
    calibration = {**first["calibration"], **patchtst_calibration}
    save_table(keep_other_models(forecast_table(df.index[rows(calib_days)], actual_calib, calibration), "transformer_calibration"),
               "transformer_calibration")
    save_table(pd.DataFrame({"loss": early_losses}, index=range(1, len(early_losses) + 1)), "transformer_training") 
    print("saved transformer_forecasts and transformer_training for the dashboard") 

    log_run(
        experiment="demand-patchtst",
        params={**model.settings, "best_epochs": model.best_epochs, "split_date": split_date,
                "calibration_start": calibration_start, "test_days": len(test_days)},
        metrics={f"{name}_{key}": value for name, m in results.items() for key, value in m.items()},
        files=[models / "patchtst.pt"],
    )