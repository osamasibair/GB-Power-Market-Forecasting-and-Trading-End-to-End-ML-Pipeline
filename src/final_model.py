"""Steps 5 to 9, the same for every tuned model: train the chosen settings with each of the five seeds, once on the data
up to the calibration year (to forecast it) and once on the data up to the test year (to forecast that); average the
seeds' 10%, 50% and 90% forecasts; calibrate the 80% range on the calibration year (never past the median); and add the
result to transformer_forecasts and transformer_calibration as <model>_tuned. The ten training runs are split across the
workers like every other step (src/parallel.py). Doesn't import PyTorch, so the boosting models can use it."""
import base64
import io
import zlib

import numpy as np

from parallel import shared

parts = ("calibration", "test")


def pack(array):
    """An array as text, to keep in the shared database."""
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(array, dtype=np.float64))
    return base64.b64encode(zlib.compress(buffer.getvalue())).decode()


def unpack(text):
    return np.load(io.BytesIO(zlib.decompress(base64.b64decode(text))))


def seed_runs(train_and_forecast, seeds, board=None, tag="", worker=0, workers=1):
    """train_and_forecast(seed, part) for every seed and both parts, split across the workers: each a (rows, 3) array of
    the 10%, 50% and 90% forecasts. Returns the calibration-year runs and the test-year runs, in seed order."""
    jobs = [[seed, part] for seed in seeds for part in parts]
    runs = shared(board, tag, jobs, lambda job: pack(train_and_forecast(job[0], job[1])), worker, workers)
    by_part = {part: [unpack(run) for job, run in zip(jobs, runs) if job[1] == part] for part in parts}
    return by_part["calibration"], by_part["test"]


def combine(key, calibration_runs, test_runs, actual_calibration, actual_test):
    """The seeds' forecasts averaged, with the 80% range calibrated on the calibration year (conformal: widened or
    narrowed until 80% of the actual demand falls inside it), never past the median, for both years; the share of the
    calibration year inside the range before and after calibrating; and each seed's test MAPE, to show the spread
    between seeds."""
    from evaluate import metrics
    from train_patchtst import conformal_adjustment

    early, final = np.mean(calibration_runs, axis=0), np.mean(test_runs, axis=0)
    adjustment = conformal_adjustment(actual_calibration, early[:, 0], early[:, 2])

    def interval(forecast):
        return {key: forecast[:, 1], f"{key}_low": np.minimum(forecast[:, 0] - adjustment, forecast[:, 1]),
                f"{key}_high": np.maximum(forecast[:, 2] + adjustment, forecast[:, 1])}

    def inside(low, high):
        return float(((actual_calibration >= low) & (actual_calibration <= high)).mean() * 100)

    calibration = interval(early)
    return {"calibration": calibration, "forecasts": interval(final), "adjustment": adjustment,
            "calibration_coverage_before": inside(early[:, 0], early[:, 2]),
            "calibration_coverage": inside(calibration[f"{key}_low"], calibration[f"{key}_high"]),
            "seed_mapes": [float(metrics(actual_test, run[:, 1])["mape"]) for run in test_runs]}


def save(key, result):
    """Add a combined result to transformer_forecasts and transformer_calibration and return its test scores. Models
    finishing at the same time on different machines take turns (a lock in the shared database), so one model's columns
    can't be overwritten by another's."""
    from sqlalchemy import text

    from chronos2 import add_columns, scored
    from db import engine, load_table, save_table

    with engine.connect() as lock:
        if engine.dialect.name == "postgresql":
            lock.execute(text("SELECT pg_advisory_lock(424242)"))
        try:
            forecasts_table = add_columns(load_table("transformer_forecasts"), result["forecasts"])
            calibration_table = add_columns(load_table("transformer_calibration"), result["calibration"])
            save_table(forecasts_table, "transformer_forecasts")
            save_table(calibration_table, "transformer_calibration")
        finally:
            if engine.dialect.name == "postgresql":
                lock.execute(text("SELECT pg_advisory_unlock(424242)"))
    m = scored(forecasts_table, [key])[key]
    spread, adjustment = result["seed_mapes"], result["adjustment"]
    print(f"5. trained with {len(spread)} seeds, each up to the calibration year and up to the test year")
    print(f"6. seeds averaged: test MAPE {m['mape']:.2f}% (single seeds {np.mean(spread):.2f}% ± "
          f"{np.std(spread, ddof=1):.2f})")
    print(f"7. 80% range calibrated on the calibration year: {result['calibration_coverage_before']:.1f}% inside "
          f"before, {'widened' if adjustment >= 0 else 'narrowed'} by {abs(adjustment):.0f} MW at each end, "
          f"{result['calibration_coverage']:.1f}% inside after (never past the median)")
    print(f"8. applied to the test year: coverage {m['coverage']:.1f}%, width {m['width']:.0f} MW")
    print(f"9. saved as {key} in transformer_forecasts and transformer_calibration")
    return m
