"""Transformer page, the PatchTST transformer against LightGBM and a pretrained Chronos model over the test year."""
import altair as alt
import pandas as pd
import streamlit as st

from db import load_table

alt.data_transformers.disable_max_rows()
names = {
    "baseline": "Same time last week",
    "chronos": "Chronos (pretrained, zero shot)",
    "chronos2": "Chronos-2 (zero shot, demand only)",
    "chronos2_covariates": "Chronos-2 (zero shot, with weather)",
    "chronos2_finetuned": "Chronos-2 (fine tuned, with weather)",
    "patchtst": "PatchTST (trained from scratch)",
    "lightgbm": "LightGBM",
}
colours = {"Actual": "#888780", "LightGBM": "#eb6834", "PatchTST": "#2a78d6", "Chronos": "#1baf7a", "Ensemble": "#e87ba4"}


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


def week_chart(week, chosen):
    data = week.rename_axis("time").reset_index()
    band = alt.Chart(data).mark_area(opacity=0.35, color=colours["PatchTST"]).encode(
        x=alt.X("time:T", title=None, axis=alt.Axis(format="%a %d %b", tickCount="day")),
        y=alt.Y("patchtst_low:Q", title="Demand (MW)", scale=alt.Scale(zero=False)), y2="patchtst_high:Q")
    columns = {key: name for key, name in {"actual": "Actual", "lightgbm": "LightGBM", "patchtst": "PatchTST",
               "chronos": "Chronos", "ensemble": "Ensemble"}.items() if key in data and (name == "Actual" or name in chosen)}
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

def weight_chart(table, chosen_weight, weight):
    data = table.reset_index().melt("weight", ["calibration_mape", "test_mape"], var_name="year", value_name="mape")
    data["year"] = data["year"].map({"calibration_mape": "Calibration year", "test_mape": "Test year"})
    lines = alt.Chart(data).mark_line(point=True, strokeWidth=2).encode(
        x=alt.X("weight:Q", title="Share of PatchTST in the blend", axis=alt.Axis(format="%")),
        y=alt.Y("mape:Q", title="MAPE (%)", scale=alt.Scale(zero=False)),
        color=alt.Color("year:N", title=None, scale=alt.Scale(range=["#888780", colours["Ensemble"]]), legend=alt.Legend(orient="top")),
        tooltip=[alt.Tooltip("weight:Q", title="PatchTST share", format=".0%"), "year:N", alt.Tooltip("mape:Q", title="MAPE", format=".2f")])
    chosen_rule = alt.Chart(pd.DataFrame({"weight": [chosen_weight]})).mark_rule(strokeDash=[4, 4], color="#888780").encode(x="weight:Q")
    slider_rule = alt.Chart(pd.DataFrame({"weight": [weight]})).mark_rule(color=colours["Ensemble"], strokeWidth=2).encode(x="weight:Q")
    return (lines + chosen_rule + slider_rule).properties(height=300)


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
forecasts = forecasts.set_axis(forecasts.index.tz_localize("UTC").tz_convert("Europe/London").tz_localize(None))

st.dataframe(scores(forecasts))
st.caption("Lower error is better, and a well calibrated 80% range contains about 80% of actual values. Fine tuned Chronos-2 is the only single model that beats LightGBM, and a blend of LightGBM, PatchTST and fine tuned Chronos-2 does better still (4.36% MAPE). LightGBM stays in production until the blend passes a live shadow test.")

ensemble = load_optional("ensemble_weights")
if ensemble is not None:
    ensemble = ensemble.set_index("weight")
    ensemble.index = ensemble.index.round(2)
    chosen_weight = float(ensemble["calibration_mape"].idxmin())
    st.subheader("Blending LightGBM and PatchTST")
    weight = round(st.slider("Share of PatchTST in the blend", 0.0, 1.0, chosen_weight, 0.05), 2)
    row, lightgbm_row = ensemble.loc[weight], ensemble.loc[0.0]
    forecasts["ensemble"] = weight * forecasts["patchtst"] + (1 - weight) * forecasts["lightgbm"]
    left, middle, right = st.columns(3)
    left.metric("Test MAPE", f"{row['test_mape']:.2f}%", f"{row['test_mape'] - lightgbm_row['test_mape']:+.2f} vs LightGBM", delta_color="inverse")
    middle.metric("In 80% range", f"{row['test_coverage']:.1f}%", "target 80%", delta_color="off")
    right.metric("Average range width", f"{row['test_width']:,.0f} MW", f"{row['test_width'] - lightgbm_row['test_width']:+,.0f} MW vs LightGBM", delta_color="inverse")
    st.altair_chart(weight_chart(ensemble, chosen_weight, weight), width="stretch")
    st.caption(f"0% is LightGBM alone and 100% is PatchTST alone. The weight is chosen on the calibration year ({1 - chosen_weight:.0%} LightGBM, {chosen_weight:.0%} PatchTST, dashed line) and scored once on the test year. The test-year line is shown for transparency: choosing the weight from it would be cheating. Each blend's 80% range is recalibrated on the calibration year.")

first, last = forecasts.index.min().date(), forecasts.index.max().date()
default = min(max(pd.Timestamp("2026-01-05").date(), first), last)
start = st.date_input("Week starting", value=default, min_value=first, max_value=last)
week = forecasts.loc[pd.Timestamp(start):pd.Timestamp(start) + pd.Timedelta(days=7) - pd.Timedelta(minutes=30)]
available = [name for key, name in [("lightgbm", "LightGBM"), ("patchtst", "PatchTST"), ("chronos", "Chronos"), ("ensemble", "Ensemble")] if key in forecasts]
chosen = st.multiselect("Models to show", available, default=[name for name in ("LightGBM", "PatchTST") if name in available])
st.altair_chart(week_chart(week, chosen), width="stretch")
st.caption("Forecasts against actual demand, with the PatchTST 80% range shaded. Pick any week in the test year: the models often make different mistakes, which is why blending them beats both.")

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