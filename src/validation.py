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
    """The data a model scored on the validation year starting at start is trained on: from the first training day
    every model shares (forecast_windows.py) to the start of the year."""
    from forecast_windows import first_training_day

    return df[(df.index >= first_training_day) & (df.index < start)]


def pinball_mw(actual, predicted, quantiles):
    """The pinball loss in MW, averaged over the quantiles and every forecast half hour: the score every choice is made
    by, for every model. It isn't scaled by anything that depends on the model, such as the spread of the history it
    reads, so models and settings that read different amounts of history are scored on the same scale."""
    error = actual[..., None] - predicted
    q = np.asarray(quantiles)
    return float(np.maximum(q * error, (q - 1) * error).mean())


def patchtst_years(df, settings, report=None, target="demand"):
    """Train PatchTST with settings once for each validation year, on the data before it, early stopping on the year
    itself as train_patchtst.py does with the calibration year. For each year: the pinball loss in MW on that year (the
    score every PatchTST choice is made by), the epoch early stopping chose and its training loss (scaled by each
    window's history, so not comparable between history lengths), the median forecast's MAPE and the model's size.
    report(year, epoch, loss), if given, is called after every epoch, which lets optuna stop poor trials early."""
    from evaluate import metrics
    from forecast_windows import first_training_day, longest_history, windows
    from patchtst import PatchTSTForecaster, quantiles

    results = []
    for i, (start, end) in enumerate(folds):
        days = day_windows(df.index, start, end)
        model = PatchTSTForecaster(**settings)
        callback = None if report is None else (lambda epoch, loss, year=i: report(year, epoch, loss))
        train = windows(df.index, first=first_training_day, last=start, stride=model.settings["train_stride"],
                        history=longest_history)
        model.fit(df, train, valid=days, callback=callback)
        predicted = model.predict(df, days)
        actual = df[target].to_numpy()[days[:, None] + np.arange(horizon)]
        results.append({"loss": pinball_mw(actual, predicted, quantiles), "best_epoch": model.best_epochs,
                        "training_loss": float(min(model.valid_losses)),
                        "mape": float(metrics(actual.ravel(), predicted[..., 1].ravel())["mape"]),
                        "parameters": sum(p.numel() for p in model.net.parameters())})
        print(f"  validation year {start} to {end}: loss {results[-1]['loss']:.1f} MW, MAPE {results[-1]['mape']:.2f}%, "
              f"best epoch {model.best_epochs}")
    return results