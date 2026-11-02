"""Gradient boosting vs deep learning page: LightGBM against PatchTST, Chronos Bolt and Chronos 2, and their blends, over the test year."""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from db import load_table
from ensemble import score, weights

alt.data_transformers.disable_max_rows()
names = {
    "baseline": "Same time last week",
    "chronos": "Chronos (pretrained, zero shot)",
    "chronos2": "Chronos-2 (zero shot, demand only)",
    "chronos2_covariates": "Chronos-2 (zero shot, with weather)",
    "chronos2_finetuned": "Chronos-2 (fine tuned, with weather)",
    "patchtst": "PatchTST (trained from scratch)",
    "lightgbm": "LightGBM",
    "blend": "Best blend (35% LightGBM, 5% PatchTST, 60% fine tuned Chronos-2)",
}
colours = {"Actual": "#888780", "LightGBM": "#eb6834", "PatchTST": "#2a78d6",
           "Chronos-2 fine tuned": "#7f56d9", "Best blend": "#1baf7a", "Ensemble": "#e87ba4"}
best_blend = {"lightgbm": 0.35, "patchtst": 0.05, "chronos2_finetuned": 0.60}
blend_names = {"lightgbm": "LightGBM", "patchtst": "PatchTST", "chronos2_finetuned": "fine tuned Chronos-2"}


def load_optional(name):
    try:
        return load_table(name)
    except ValueError:
        return None


def scores(forecasts):
    actual = forecasts["actual"]
    rows = []
    for key, label in names.items():
        if key not in forecasts:
            continue
        error = forecasts[key] - actual
        row = {"Model": label, "MAE (MW)": error.abs().mean(), "RMSE (MW)": (error ** 2).mean() ** 0.5,
               "MAPE (%)": (error.abs() / actual).mean() * 100}
        if f"{key}_low" in forecasts:
            low, high = forecasts[f"{key}_low"], forecasts[f"{key}_high"]
            row["In 80% range (%)"] = ((actual >= low) & (actual <= high)).mean() * 100
            row["Average width (MW)"] = (high - low).mean()
        rows.append(row)
    table = pd.DataFrame(rows).set_index("Model").sort_values("MAPE (%)", ascending=False)
    return table.round({"MAE (MW)": 0, "RMSE (MW)": 0, "MAPE (%)": 2, "In 80% range (%)": 1, "Average width (MW)": 0})

def add_best_blend(forecasts, calibration):
    """Add the final blend to the test year forecasts, with its 80% range recalibrated on the calibration year like the other models, so it matches src/ensemble.py."""
    if calibration is None or any(m not in forecasts or m not in calibration for m in best_blend):
        return forecasts
    def mix(frame, part):
        return sum(weight * frame[f"{model}{part}"] for model, weight in best_blend.items())
    low, high = mix(calibration, "_low"), mix(calibration, "_high")
    adjustment = float(np.quantile(np.maximum(low - calibration["actual"], calibration["actual"] - high), 0.8))
    return forecasts.assign(blend=mix(forecasts, ""), blend_low=mix(forecasts, "_low") - adjustment,
                            blend_high=mix(forecasts, "_high") + adjustment)

def week_chart(week, chosen):
    data = week.rename_axis("time").reset_index()
    band = alt.Chart(data).mark_area(opacity=0.35, color=colours["PatchTST"]).encode(
        x=alt.X("time:T", title=None, axis=alt.Axis(format="%a %d %b", tickCount="day")),
        y=alt.Y("patchtst_low:Q", title="Demand (MW)", scale=alt.Scale(zero=False)), y2="patchtst_high:Q")
    columns = {key: name for key, name in {"actual": "Actual", "lightgbm": "LightGBM", "patchtst": "PatchTST",
               "chronos2_finetuned": "Chronos-2 fine tuned", "blend": "Best blend",
               "ensemble": "Ensemble"}.items() if key in data and (name == "Actual" or name in chosen)}
    lines = data.melt("time", list(columns), var_name="model", value_name="demand")
    lines["model"] = lines["model"].map(columns)
    shown = [name for name in colours if name in set(lines["model"])]
    line = alt.Chart(lines).mark_line(strokeWidth=1.8).encode(
        x="time:T", y="demand:Q",
        color=alt.Color("model:N", title=None, sort=shown,
                        scale=alt.Scale(domain=shown, range=[colours[name] for name in shown])),
        tooltip=[alt.Tooltip("time:T", format="%a %d %b %H:%M"), "model:N", alt.Tooltip("demand:Q", format=",.0f")])
    chart = band + line if "PatchTST" in chosen else line
    return chart.properties(height=420)

def rebalance(moved):
    """Share what the moved slider leaves between the other two in their current ratio, so the shares always add up to 100%."""
    others = [model for model in best_blend if model != moved]
    left = round(1 - st.session_state[f"share_{moved}"], 2)
    a, b = (st.session_state[f"share_{model}"] for model in others)
    first = left / 2 if a + b == 0 else left * a / (a + b)
    first = round(round(first / 0.05) * 0.05, 2)
    st.session_state[f"share_{others[0]}"], st.session_state[f"share_{others[1]}"] = first, round(left - first, 2)


def reset_shares():
    st.session_state.update({f"share_{model}": weight for model, weight in best_blend.items()})


def share_table(calibration, forecasts, model, shares):
    """The error for every share of `model`, with the other two models sharing the rest in their current ratio."""
    others = {m: s for m, s in shares if m != model}
    total = sum(others.values())
    rows = []
    for weight in weights:
        mix = {model: weight, **{m: (1 - weight) * (s / total if total else 0.5) for m, s in others.items()}}
        rows.append({"weight": weight, **score(calibration, forecasts, {m: w for m, w in mix.items() if w > 0.0001})})
    return pd.DataFrame(rows).set_index("weight")


def weight_chart(table, weight, name, label):
    data = table.reset_index().melt("weight", ["calibration_mape", "test_mape"], var_name="year", value_name="mape")
    data["year"] = data["year"].map({"calibration_mape": "Calibration year", "test_mape": "Test year"})
    lines = alt.Chart(data).mark_line(point=True, strokeWidth=2).encode(
        x=alt.X("weight:Q", title=label, axis=alt.Axis(labels=False, ticks=False)),
        y=alt.Y("mape:Q", title="MAPE (%)", scale=alt.Scale(zero=False)),
        color=alt.Color("year:N", title=None, scale=alt.Scale(range=["#888780", colours["Ensemble"]]), legend=alt.Legend(orient="top")),
        tooltip=[alt.Tooltip("weight:Q", title=f"{name} share", format=".0%"), "year:N", alt.Tooltip("mape:Q", title="MAPE", format=".2f")])
    slider_rule = alt.Chart(pd.DataFrame({"weight": [weight]})).mark_rule(color=colours["Ensemble"], strokeWidth=2).encode(x="weight:Q")
    return (lines + slider_rule).properties(height=300)


def month_chart(forecasts):
    """The test MAPE in each month for the slider blend, the best blend and the three models in it."""
    lines = {"ensemble": "Ensemble", "blend": "Best blend", "chronos2_finetuned": "Chronos-2 fine tuned",
             "lightgbm": "LightGBM", "patchtst": "PatchTST"}
    actual = forecasts["actual"]
    errors = pd.DataFrame({name: (forecasts[key] - actual).abs() / actual * 100 for key, name in lines.items()})
    data = errors.groupby(errors.index.to_period("M")).mean()
    data = data.set_axis(data.index.to_timestamp()).rename_axis("month").reset_index().melt("month", var_name="model", value_name="mape")
    shown = list(lines.values())
    return alt.Chart(data).mark_line(point=True, strokeWidth=2).encode(
        x=alt.X("month:T", title=None, axis=alt.Axis(format="%b %Y", tickCount="month")),
        y=alt.Y("mape:Q", title="MAPE (%)", scale=alt.Scale(zero=False)),
        color=alt.Color("model:N", title=None, sort=shown, legend=alt.Legend(orient="top"),
                        scale=alt.Scale(domain=shown, range=[colours[name] for name in shown])),
        tooltip=[alt.Tooltip("month:T", format="%B %Y"), "model:N", alt.Tooltip("mape:Q", title="MAPE", format=".2f")],
    ).properties(height=300)


def influence_chart(influence):
    data = influence.assign(start=-influence["hours_before_midnight"], end=-influence["hours_before_midnight"] + 0.5)
    blues = alt.Scale(scheme="blues")
    return alt.Chart(data).mark_rect(strokeWidth=1).encode(
        x=alt.X("start:Q", title="Hours before tomorrow's midnight", scale=alt.Scale(domain=[-184, -16], nice=False),
                axis=alt.Axis(values=list(range(-168, 0, 24)), labelExpr="-datum.value")),
        x2="end:Q",
        y=alt.Y("forecast_time:O", title="Forecast time tomorrow", axis=alt.Axis(values=[f"{h:02d}:00" for h in range(0, 24, 3)])),
        color=alt.Color("influence:Q", title="MW per MW", scale=blues),
        stroke=alt.Stroke("influence:Q", scale=blues, legend=None),
        tooltip=["forecast_time:O", alt.Tooltip("hours_before_midnight:Q", title="hours before midnight"),
                 alt.Tooltip("influence:Q", format=".4f")],
    ).properties(height=380)


st.title("Gradient Boosting vs Deep Learning")
st.caption("LightGBM, the gradient boosting model in production, against three kinds of deep learning model: PatchTST, a transformer implemented in PyTorch and trained from scratch on this data, Chronos-Bolt and Chronos-2, Amazon's pretrained time series models, used without training, and Chronos-2 fine tuned on this data. Every model sees demand up to 08:00 the day before plus tomorrow's weather and calendar (except the demand only versions), gets the same conformal calibration, and is scored on the same test days (1 September 2025 to 31 August 2026, clock change days skipped).")

forecasts = load_optional("transformer_forecasts")
if forecasts is None:
    st.info("No transformer results yet. Run src/train_patchtst.py to create them.")
    st.stop()
calibration = load_optional("transformer_calibration")
forecasts = add_best_blend(forecasts, calibration)
forecasts = forecasts.set_axis(forecasts.index.tz_localize("UTC").tz_convert("Europe/London").tz_localize(None))

st.dataframe(scores(forecasts))
if "blend" in forecasts:
    actual = forecasts["actual"]
    def mape(key):
        return ((forecasts[key] - actual).abs() / actual).mean() * 100
    def width(key):
        return (forecasts[f"{key}_high"] - forecasts[f"{key}_low"]).mean()
    covered = ((actual >= forecasts["blend_low"]) & (actual <= forecasts["blend_high"])).mean() * 100
    left, middle, right = st.columns(3)
    left.metric("Best blend: test MAPE", f"{mape('blend'):.2f}%", f"{mape('blend') - mape('lightgbm'):+.2f} vs LightGBM", delta_color="inverse")
    middle.metric("In 80% range", f"{covered:.1f}%", "target 80%", delta_color="off")
    right.metric("Average range width", f"{width('blend'):,.0f} MW", f"{width('blend') - width('lightgbm'):+,.0f} MW vs LightGBM", delta_color="inverse")
st.caption("Lower error is better, and a well calibrated 80% range contains about 80% of actual values. Fine tuned Chronos-2 is the only single model that beats LightGBM, and a blend of LightGBM, PatchTST and fine tuned Chronos-2 does better still (4.36% MAPE). LightGBM stays in production until the blend passes a live shadow test.")

if "blend" in forecasts:
    st.subheader("Blending LightGBM, PatchTST and fine tuned Chronos-2")
    for model, name in blend_names.items():
        st.session_state.setdefault(f"share_{model}", best_blend[model])
        st.slider(f"Share of {name} in the blend", 0.0, 1.0, step=0.05, key=f"share_{model}", on_change=rebalance, args=(model,))
    st.button("Reset to the best blend", on_click=reset_shares)
    shares = {model: round(st.session_state[f"share_{model}"], 2) for model in best_blend}
    row = score(calibration, forecasts, {m: w for m, w in shares.items() if w > 0})
    lightgbm_row = score(calibration, forecasts, {"lightgbm": 1.0})
    forecasts["ensemble"] = sum(weight * forecasts[model] for model, weight in shares.items())
    left, middle, right = st.columns(3)
    left.metric("Test MAPE", f"{row['test_mape']:.2f}%", f"{row['test_mape'] - lightgbm_row['test_mape']:+.2f} vs LightGBM", delta_color="inverse")
    middle.metric("In 80% range", f"{row['test_coverage']:.1f}%", "target 80%", delta_color="off")
    right.metric("Average range width", f"{row['test_width']:,.0f} MW", f"{row['test_width'] - lightgbm_row['test_width']:+,.0f} MW vs LightGBM", delta_color="inverse")
    moved = "chronos2_finetuned"
    table = st.cache_data(share_table)(calibration, forecasts, moved, tuple(shares.items()))
    label = ", ".join(f"{shares[model]:.0%} {name}" for model, name in blend_names.items())
    st.altair_chart(weight_chart(table, shares[moved], blend_names[moved], label), width="stretch")
    st.caption(f"The three shares always add up to 100%: moving one slider shares the rest between the other two in their current ratio. The chart moves the share of fine tuned Chronos-2 from 0% to 100%, with LightGBM and PatchTST sharing the rest in their current ratio. The defaults are the best blend, chosen on the calibration year ({best_blend['lightgbm']:.0%} LightGBM, {best_blend['patchtst']:.0%} PatchTST, {best_blend['chronos2_finetuned']:.0%} fine tuned Chronos-2) and scored once on the test year. The test-year line is shown for transparency: choosing the weights from it would be cheating. Each blend's 80% range is recalibrated on the calibration year.")

first, last = forecasts.index.min().date(), forecasts.index.max().date()
default = min(max(pd.Timestamp("2026-01-05").date(), first), last)
start = st.date_input("Week starting", value=default, min_value=first, max_value=last)
week = forecasts.loc[pd.Timestamp(start):pd.Timestamp(start) + pd.Timedelta(days=7) - pd.Timedelta(minutes=30)]
available = [name for key, name in [("blend", "Best blend"), ("chronos2_finetuned", "Chronos-2 fine tuned"), ("lightgbm", "LightGBM"),
             ("patchtst", "PatchTST"), ("ensemble", "Ensemble")] if key in forecasts]
chosen = st.multiselect("Models to show", available, default=[name for name in ("Best blend", "LightGBM") if name in available])
st.altair_chart(week_chart(week, chosen), width="stretch")
st.caption("Forecasts against actual demand (the PatchTST 80% range is shaded when PatchTST is shown). Pick any week in the test year: the models often make different mistakes, which is why blending them beats each of them alone. Ensemble is the blend picked with the sliders above.")

if "ensemble" in forecasts:
    st.subheader("Error by month")
    st.altair_chart(month_chart(forecasts), width="stretch")
    st.caption("The test-year error in each month for the blend picked with the sliders above (Ensemble), the best blend, and LightGBM, PatchTST and fine tuned Chronos-2 on their own. The pink and green lines overlap until the sliders are moved away from the best blend.")

influence = load_optional("transformer_influence")
if influence is not None:
    st.subheader("What the transformer looks at")
    st.altair_chart(influence_chart(influence), width="stretch")
    st.caption("How many MW each of tomorrow's forecasts moves per MW change in each past half-hour, averaged over every test day (the gradient of the forecast). The influence peaks at the same times every day across the week and is stronger for daytime forecasts, so the transformer matches times of day, as LightGBM's same time yesterday and last week features do. The most recent full day matters most.")

training = load_optional("transformer_training")
if training is not None:
    st.subheader("Training")
    st.line_chart(training.rename_axis("epoch").rename(columns={"loss": "calibration-year loss"}), x_label="Epoch", y_label="Loss")
    st.caption(f"The loss on the calibration year after each pass over the training data. Training stopped when it stopped improving, and the best epoch ({int(training['loss'].idxmin())}) was kept.")