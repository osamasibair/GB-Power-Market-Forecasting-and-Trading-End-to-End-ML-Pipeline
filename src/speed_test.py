"""Time PatchTST's training on the CPU and on a GPU, to know how long tuning will take on each. Trains on every 8th of
the windows the first validation year's model trains on (from the first training day to 1 September 2023, the 42-day
lookback), for 2 epochs, with the default settings and with the largest settings the search can pick, and prints the
time a full epoch would take on each (8 times the sample's). Nothing is saved.
    python src/speed_test.py --device mps     (a Mac's GPU)
    python src/speed_test.py --device cuda    (an Nvidia GPU, e.g. on Colab)"""
import argparse
import os
import time

from db import load_table
from forecast_windows import first_training_day, longest_history, windows
from validation import folds

settings = {
    "defaults": {"history_days": 42},
    "largest": {"history_days": 42, "d_model": 256, "layers": 5, "hidden": 512, "patch_len": 12, "stride": 6,
                "batch_size": 64},
}


def seconds_per_epoch(df, starts, where, config, epochs=2):
    from patchtst import PatchTSTForecaster

    os.environ["PATCHTST_DEVICE"] = where
    started = time.perf_counter()
    PatchTSTForecaster(**config).fit(df, starts, epochs=epochs)
    return (time.perf_counter() - started) / epochs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="mps", help="the GPU to compare with the CPU: mps or cuda")
    args = parser.parse_args()

    df = load_table("features")
    # every 8th of the windows training uses (which are 2 half hours apart)
    starts = windows(df.index, first=first_training_day, last=folds[0][0], stride=16, history=longest_history)
    print(f"{len(starts):,} sample windows, 2 epochs each; time per full epoch:\n")
    print(f"{'settings':>10} {'cpu':>12} {args.device:>12} {'faster':>8}")
    for name, config in settings.items():
        cpu = 8 * seconds_per_epoch(df, starts, "cpu", config)
        gpu = 8 * seconds_per_epoch(df, starts, args.device, config)
        print(f"{name:>10} {cpu:>10.0f} s {gpu:>10.0f} s {cpu / gpu:>7.1f}x")