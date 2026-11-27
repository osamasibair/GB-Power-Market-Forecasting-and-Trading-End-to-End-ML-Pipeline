"""Draw the PatchTST figures for the README into docs/:
- a week of test period forecasts from PatchTST and LightGBM against actual demand, with PatchTST's 80% range
- the learning curve: the loss on the calibration year after each epoch, recorded but not used to choose anything
- which half hours of the past week each of tomorrow  s forecasts depends on most (the gradient of the forecast with respect to each past value, averaged over every test day)"""
import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter

from db import load_table, save_table
from forecast_windows import day_windows, gap_steps, history_steps, horizon

docs, models = Path("docs"), Path("models")
ink, muted, grid = "#0b0b0b", "#52514e", "#e4e3df"
blue, blue_band, orange = "#2a78d6", "#cde2fb", "#eb6834"
influence_map = LinearSegmentedColormap.from_list(
    "influence", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
thousands = FuncFormatter(lambda value, _: f"{value:,.0f}")


def lightgbm_forecast(positions):
    """LightGBM trained only on data before the test period, forecasting the given rows. Runs in its own process."""
    from train import features, split_date, train_model

    df = load_table("features")
    return train_model(df[df.index < split_date]).predict(df.iloc[positions][features])


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(grid)
    ax.tick_params(colors=muted, labelsize=9)
    ax.grid(axis="y", color=grid, linewidth=0.6)
    ax.set_axisbelow(True)


def pick_week(df, test_days, week):
    local = df.index[test_days].tz_localize("UTC").tz_convert("Europe/London")
    if week is None:
        peaks = [df["demand"].iloc[s:s + horizon].max() for s in test_days]
        busiest = local[int(np.argmax(peaks))]
        start = (busiest - pd.Timedelta(days=busiest.dayofweek)).normalize()
    else:
        start = pd.Timestamp(week, tz="Europe/London")
    chosen = test_days[(local >= start) & (local < start + pd.Timedelta(days=7))]
    if len(chosen) == 0:
        raise SystemExit(f"no test days in the week from {start:%Y-%m-%d}")
    return start, chosen


def forecast_week(df, model, adjustment, starts, lightgbm, start):
    rows = (starts[:, None] + np.arange(horizon)).ravel()
    predicted = model.predict(df, starts).reshape(-1, 3)
    frame = pd.DataFrame({
        "low": predicted[:, 0] - adjustment, "patchtst": predicted[:, 1], "high": predicted[:, 2] + adjustment,
        "lightgbm": lightgbm, "actual": df["demand"].to_numpy()[rows],
    }, index=df.index[rows].tz_localize("UTC").tz_convert("Europe/London"))
    frame = frame.reindex(pd.date_range(frame.index.min(), frame.index.max(), freq="30min"))

    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=150)
    ax.fill_between(frame.index, frame["low"], frame["high"], color=blue_band, linewidth=0, label="PatchTST 80% range")
    ax.plot(frame.index, frame["lightgbm"], color=orange, linewidth=2, linestyle=(0, (4, 2)), label="LightGBM")
    ax.plot(frame.index, frame["patchtst"], color=blue, linewidth=2, label="PatchTST")
    ax.plot(frame.index, frame["actual"], color=ink, linewidth=1.5, label="Actual")
    style(ax)
    ax.yaxis.set_major_formatter(thousands)
    ax.xaxis.set_major_locator(mdates.DayLocator(tz=frame.index.tz))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d %b", tz=frame.index.tz))
    ax.set_ylabel("Demand (MW)", color=muted, fontsize=9)
    ax.set_title(f"Day-ahead forecasts for the week of {start:%d %b %Y}, made at 08:00 the day before",
                 loc="left", fontsize=11, color=ink)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.1), ncol=4, frameon=False, fontsize=9, labelcolor=muted)
    fig.tight_layout()
    fig.savefig(docs / "patchtst_forecast_week.png", bbox_inches="tight")
    plt.close(fig)


def learning_curve(losses):
    epochs = np.arange(1, len(losses) + 1)
    best = len(losses) - 1  # the number of epochs is a tuned setting, so the last epoch is the one kept
    fig, ax = plt.subplots(figsize=(6, 3.2), dpi=150)
    ax.plot(epochs, losses, color=blue, linewidth=2, marker="o", markersize=4)
    ax.plot(epochs[best], losses[best], marker="o", markersize=9, color=blue, markeredgecolor="#fcfcfb", markeredgewidth=2)
    ax.annotate(f"kept: epoch {epochs[best]}", (epochs[best], losses[best]), xytext=(0, 36),
                textcoords="offset points", ha="center", fontsize=9, color=ink,
                arrowprops={"arrowstyle": "-", "color": muted, "linewidth": 0.6})
    style(ax)
    ax.set_xlabel("Epoch", color=muted, fontsize=9)
    ax.set_ylabel("Calibration-year loss", color=muted, fontsize=9)
    ax.set_title("Calibration-year loss by epoch (recorded, not used to choose)", loc="left", fontsize=11, color=ink)
    fig.tight_layout()
    fig.savefig(docs / "patchtst_learning_curve.png", bbox_inches="tight")
    plt.close(fig)


def influence(df, model, starts):
    """Mean absolute gradient of each of tomorrow's 48 median forecasts with respect to each of the 336 past values:
    how many MW the forecast moves per MW change in that past half hour."""
    import torch

    history, future, _ = model.tensors(df, starts)
    history.requires_grad_(True)
    median = model.net(history, future)[..., 1]
    grads = np.zeros((horizon, history_steps))
    for step in range(horizon):
        (gradient,) = torch.autograd.grad(median[:, step].sum(), history, retain_graph=True)
        grads[step] = gradient.abs().mean(dim=0).numpy()
    return grads

def influence_table(grads):
    """The influence map in long form for the dashboard, one row per forecast half hour and past half hour."""
    step, past = np.meshgrid(np.arange(horizon), np.arange(history_steps), indexing="ij")
    return pd.DataFrame({
        "forecast_time": [f"{s // 2:02d}:{30 * (s % 2):02d}" for s in step.ravel()],
        "hours_before_midnight": (gap_steps + history_steps - past.ravel()) / 2,
        "influence": grads.ravel(),
    })

def influence_plot(grads):
    first = -(gap_steps + history_steps) / 2
    last = -gap_steps / 2
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(10, 5.6), dpi=150, sharex=True,
                                      gridspec_kw={"height_ratios": [3, 1]})
    image = top.imshow(grads, aspect="auto", cmap=influence_map, extent=[first, last, 24, 0], interpolation="nearest")
    top.set_ylabel("Forecast time tomorrow", color=muted, fontsize=9)
    top.set_yticks(range(0, 25, 6), [f"{hour:02d}:00" for hour in range(0, 25, 6)])
    top.set_title("Which past half-hours each of tomorrow's forecasts depends on", loc="left", fontsize=11, color=ink)
    for ax in (top, bottom):
        ax.tick_params(colors=muted, labelsize=9)
        for side in ax.spines:
            ax.spines[side].set_color(grid)
    colourbar = fig.colorbar(image, ax=[top, bottom], fraction=0.025, pad=0.02)
    colourbar.set_label("MW change per MW", color=muted, fontsize=9)
    colourbar.ax.tick_params(colors=muted, labelsize=8)
    colourbar.outline.set_visible(False)

    hours = np.linspace(first, last, history_steps, endpoint=False) + 0.25
    bottom.plot(hours, grads.mean(axis=0), color=blue, linewidth=1.5)
    style(bottom)
    bottom.set_ylabel("Average", color=muted, fontsize=9)
    ticks = np.arange(-168, last + 1, 24)
    labels = [f"{-int(t) // 24} day{'s' if t < -24 else ''} before" for t in ticks]
    bottom.set_xticks(ticks, labels)
    bottom.set_xlabel("Past demand, by time before tomorrow's midnight (history ends 16 hours before)", color=muted, fontsize=9)
    fig.savefig(docs / "patchtst_influence.png", bbox_inches="tight")
    plt.close(fig)


def summary_text(model):
    parameters = sum(p.numel() for p in model.net.parameters())
    lines = [f"PatchTST: {parameters:,} trainable parameters"]
    try:
        from torchinfo import summary
    except ImportError:
        lines.append("(install torchinfo for the table)")
    else:
        n_covariates = len(model.cov_mean)
        lines.append(str(summary(model.net, input_size=[(1, history_steps), (1, horizon, n_covariates)], verbose=0)))
    (docs / "patchtst_summary.txt").write_text("\n".join(lines) + "\n")
    return parameters


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--week", help="Monday of the week to plot, YYYY-MM-DD (default: the week with the highest test-period peak)")
    args = parser.parse_args()

    saved_path = models / "patchtst.pt"
    if not saved_path.exists():
        raise SystemExit("models/patchtst.pt not found, run src/train_patchtst.py first")
    df = load_table("features")
    docs.mkdir(exist_ok=True)

    from patchtst import PatchTSTForecaster

    model, saved = PatchTSTForecaster.load(saved_path)
    if "split_date" not in saved:
        raise SystemExit("models/patchtst.pt is from an older version, run src/train_patchtst.py again")
    test_days = day_windows(df.index, saved["split_date"])
    start, week = pick_week(df, test_days, args.week)
    rows = (week[:, None] + np.arange(horizon)).ravel()

    print(f"lightgbm: forecasting the week of {start:%d %b %Y} in a separate process")
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as pool:
        lightgbm = pool.submit(lightgbm_forecast, rows).result()

    forecast_week(df, model, saved["adjustment"], week, lightgbm, start)
    print("saved docs/patchtst_forecast_week.png")
    if saved.get("valid_losses"):
        learning_curve(saved["valid_losses"])
        print("saved docs/patchtst_learning_curve.png")
    grads = influence(df, model, test_days)
    influence_plot(grads)
    print(f"saved docs/patchtst_influence.png ({len(test_days)} test days)")
    save_table(influence_table(grads), "transformer_influence")
    print("saved transformer_influence for the dashboard")
    print(f"saved docs/patchtst_summary.txt ({summary_text(model):,} parameters)")