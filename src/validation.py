"""The two validation years every choice is made on. Each year is forecast by a model trained only on the data before
it, and a choice's score is its average over the two years, so no choice rests on one year. The 80% ranges are still
calibrated on the second year alone (calibration_start in quantiles.py), the year closest to the test year, and the
test year (from split_date in train.py) is never used here."""
import numpy as np

from forecast_windows import day_windows, horizon

folds = [("2023-09-01", "2024-09-01"), ("2024-09-01", "2025-09-01")]


def fold_rows(df, start, end):
    """The rows of df forecast on the days of the validation year from start to end: every half hour of each forecast
    day, the same days every model is scored on."""
    starts = day_windows(df.index, start, end)
    return df.iloc[(starts[:, None] + np.arange(horizon)).ravel()]


def before(df, start):
    """The data a model scored on the validation year starting at start may be trained on."""
    return df[df.index < start]


def patchtst_years(df, settings, report=None, target="demand"):
    """Train PatchTST with settings once for each validation year, on the data before it, early stopping on the year
    itself as train_patchtst.py does with the calibration year. For each year: the best pinball loss on that year
    (the score every PatchTST choice is made by), the epoch it came at, the median forecast's MAPE and the model's size.
    report(year, epoch, loss), if given, is called after every epoch, which lets optuna stop poor trials early."""
    from evaluate import metrics
    from forecast_windows import windows
    from patchtst import PatchTSTForecaster

    results = []
    for i, (start, end) in enumerate(folds):
        days = day_windows(df.index, start, end)
        model = PatchTSTForecaster(**settings)
        callback = None if report is None else (lambda epoch, loss, year=i: report(year, epoch, loss))
        model.fit(df, windows(df.index, last=start, stride=model.settings["train_stride"]), valid=days, callback=callback)
        median = model.predict(df, days)[..., 1].ravel()
        actual = df[target].to_numpy()[(days[:, None] + np.arange(horizon)).ravel()]
        results.append({"loss": float(min(model.valid_losses)), "best_epoch": model.best_epochs,
                        "mape": float(metrics(actual, median)["mape"]),
                        "parameters": sum(p.numel() for p in model.net.parameters())})
        print(f"  validation year {start} to {end}: loss {results[-1]['loss']:.4f}, MAPE {results[-1]['mape']:.2f}%, "
              f"best epoch {model.best_epochs}")
    return results