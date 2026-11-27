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


def edge_settings(params, distributions, natural=None, margin=0.1):
    """The rule every model's search follows, applied the same way to each: the best settings `params` that lie in the
    outer `margin` (10%) of their search range, measured on the scale searched (log or linear), or at the first or last
    of a list of choices. Such a range is widened and that model's search rerun. `natural` gives the limits a setting
    can't go past, as {(setting, "low" or "high"): limit} (no dropout, a full sample of rows): an end already at its
    limit isn't flagged. `distributions` are optuna's, from a finished trial. Returns (setting, value, "low" or "high")
    for each setting at an edge."""
    import math

    natural = natural or {}
    out = []
    for name, d in distributions.items():
        value = params[name]
        if hasattr(d, "choices"):
            choices = list(d.choices)
            side = "low" if value == choices[0] else "high" if value == choices[-1] else None
            end = choices[0] if side == "low" else choices[-1]
        else:
            if getattr(d, "log", False):
                position = math.log(value / d.low) / math.log(d.high / d.low)
            else:
                position = (value - d.low) / (d.high - d.low)
            side = "low" if position < margin else "high" if position > 1 - margin else None
            end = d.low if side == "low" else d.high
        if side and natural.get((name, side)) != end:
            out.append((name, value, side))
    return out


def patchtst_years(df, settings, target="demand"):
    """Train PatchTST with settings once for each validation year, on the data before it, for exactly its number of
    epochs (a setting, chosen by tuning like the boosting models' number of trees): nothing is chosen by looking at the
    year being scored. For each year: the pinball loss in MW on that year (the score every PatchTST choice is made by),
    the median forecast's MAPE, the number of epochs and the model's size."""
    from evaluate import metrics
    from forecast_windows import first_training_day, longest_history, windows
    from patchtst import PatchTSTForecaster, quantiles

    results = []
    for start, end in folds:
        days = day_windows(df.index, start, end)
        model = PatchTSTForecaster(**settings)
        train = windows(df.index, first=first_training_day, last=start, stride=model.settings["train_stride"],
                        history=longest_history)
        model.fit(df, train, epochs=model.settings["epochs"])
        predicted = model.predict(df, days)
        actual = df[target].to_numpy()[days[:, None] + np.arange(horizon)]
        results.append({"loss": pinball_mw(actual, predicted, quantiles), "epochs": model.settings["epochs"],
                        "mape": float(metrics(actual.ravel(), predicted[..., 1].ravel())["mape"]),
                        "parameters": sum(p.numel() for p in model.net.parameters())})
        print(f"  validation year {start} to {end}: loss {results[-1]['loss']:.1f} MW, MAPE {results[-1]['mape']:.2f}%, "
              f"{model.settings['epochs']} epochs")
    return results