"""Inside the Transformer: a step by step walk through what PatchTST does for one test day, from the week it reads to
the forecast it makes, using the real inputs and attention saved by src/walkthrough.py."""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from db import load_table

colours = {"line": "#2a78d6", "actual": "#888780", "lanes": ["#2a78d6", "#7f56d9"], "forecast": "#eb6834"}
steps = 5
step_titles = ["1. What it reads: the past week", "2. Rescaling the week (RevIN)", "3. Cutting the week into patches",
               "4. Where each of tomorrow's half hours looks (cross attention)", "5. The forecast"]


def to_london(index):
    return pd.DatetimeIndex(index).tz_localize("UTC").tz_convert("Europe/London").tz_localize(None)


@st.cache_data
def load():
    try:
        inputs, attention = load_table("transformer_inputs"), load_table("transformer_attention")
        forecasts = load_table("transformer_forecasts")[["actual", "patchtst", "patchtst_low", "patchtst_high"]]
    except ValueError:
        return None
    inputs = inputs.set_axis(to_london(inputs.index))
    inputs["history_start"] = to_london(inputs["history_start"])
    return inputs, attention.set_axis(to_london(attention.index)), forecasts.set_axis(to_london(forecasts.index))


def week_frame(row):
    demand = row.filter(regex=r"^d\d{3}$").to_numpy(float)
    times = pd.date_range(row["history_start"], periods=len(demand), freq="30min")
    scaled = (demand - row["mean"]) / row["std"] * row["revin_weight"] + row["revin_bias"]
    return pd.DataFrame({"time": times, "demand": demand, "scaled": scaled})


def patch_frame(row, n_patches):
    """Each patch's start and end, and its middle stretch, which tiles the week without overlapping."""
    half_hour = pd.Timedelta(minutes=30)
    start = row["history_start"] + np.arange(n_patches) * int(row["stride"]) * half_hour
    length = int(row["patch_len"]) * half_hour
    middle = (length - int(row["stride"]) * half_hour) / 2
    return pd.DataFrame({"patch": np.arange(1, n_patches + 1), "start": start, "end": start + length,
                         "mid_start": start + middle, "mid_end": start + middle + int(row["stride"]) * half_hour,
                         "lane": np.arange(n_patches) % 2})


x_axis = alt.Axis(format="%a %H:%M", labelAngle=0, tickCount="day")
y_width = 70  # the same y axis width on stacked charts, so their time axes line up


def week_chart(week, field, title, cutoff=None, day=None, domain=None):
    scale = alt.Scale(domain=domain) if domain else alt.Scale()
    line = alt.Chart(week).mark_line(color=colours["line"], strokeWidth=1.6).encode(
        x=alt.X("time:T", title=None, axis=x_axis, scale=scale),
        y=alt.Y(f"{field}:Q", title=title, scale=alt.Scale(zero=False), axis=alt.Axis(minExtent=y_width)),
        tooltip=[alt.Tooltip("time:T", format="%a %d %b %H:%M"), alt.Tooltip(f"{field}:Q", format=",.2f")])
    layers = [line]
    if day is not None:
        shade = pd.DataFrame({"start": [day], "end": [day + pd.Timedelta(days=1)]})
        layers.insert(0, alt.Chart(shade).mark_rect(color=colours["forecast"], opacity=0.20).encode(x="start:T", x2="end:T"))
    if cutoff is not None:
        rule = pd.DataFrame({"time": [cutoff], "label": ["08:00 cutoff"]})
        layers.append(alt.Chart(rule).mark_rule(strokeDash=[4, 3], color=colours["actual"]).encode(x="time:T"))
        layers.append(alt.Chart(rule).mark_text(align="left", baseline="top", dx=4, color=colours["actual"]).encode(
            x="time:T", y=alt.value(4), text="label:N"))
    return alt.layer(*layers).properties(height=260)


def lanes_chart(patches, domain):
    x = alt.X("start:T", title=None, axis=x_axis, scale=alt.Scale(domain=domain))
    bars = alt.Chart(patches).mark_rect(opacity=0.55, cornerRadius=3).encode(
        x=x, x2="end:T", y=alt.Y("lane:O", title=None, axis=alt.Axis(labels=False, ticks=False, domain=False, minExtent=y_width)),
        color=alt.Color("lane:N", legend=None, scale=alt.Scale(range=colours["lanes"])),
        tooltip=["patch:Q", alt.Tooltip("start:T", format="%a %H:%M"), alt.Tooltip("end:T", format="%a %H:%M")])
    labels = alt.Chart(patches).mark_text(fontSize=9, color="white").encode(
        x=alt.X("mid:T", scale=alt.Scale(domain=domain)), y=alt.Y("lane:O"), text="patch:Q"
    ).transform_calculate(mid="(toDate(datum.start) + toDate(datum.end)) / 2")
    return (bars + labels).properties(height=120)


def heatmap(attention, patches):
    data = attention.copy()
    data.columns = range(1, data.shape[1] + 1)
    data = data.assign(forecast_time=data.index.strftime("%H:%M")).melt("forecast_time", var_name="patch", value_name="weight")
    data = data.merge(patches[["patch", "start", "end"]], on="patch")
    return alt.Chart(data).mark_rect().encode(
        x=alt.X("patch:O", title="Patch of the past week (1 is the oldest, 27 ends at the 08:00 cutoff)", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("forecast_time:O", title="Forecast time tomorrow", axis=alt.Axis(values=[f"{h:02d}:00" for h in range(0, 24, 3)])),
        color=alt.Color("weight:Q", title="Attention", scale=alt.Scale(scheme="blues")),
        tooltip=["forecast_time:O", "patch:O", alt.Tooltip("start:T", title="from", format="%a %H:%M"),
                 alt.Tooltip("end:T", title="to", format="%a %H:%M"), alt.Tooltip("weight:Q", format=".1%")],
    ).properties(height=380)


def attention_bars(patches, weights, domain):
    bars = patches.assign(weight=weights)
    x = alt.X("mid_start:T", title=None, axis=x_axis, scale=alt.Scale(domain=domain))
    return alt.Chart(bars).mark_rect(color=colours["line"], opacity=0.85).encode(
        x=x, x2="mid_end:T", y=alt.Y("weight:Q", title="Attention", axis=alt.Axis(format="%", minExtent=y_width)),
        tooltip=["patch:Q", alt.Tooltip("start:T", title="from", format="%a %H:%M"),
                 alt.Tooltip("end:T", title="to", format="%a %H:%M"), alt.Tooltip("weight:Q", format=".1%")],
    ).properties(height=150)


def output_chart(day_forecast):
    data = day_forecast.rename_axis("time").reset_index()
    band = alt.Chart(data).mark_area(opacity=0.3, color=colours["line"]).encode(
        x=alt.X("time:T", title=None, axis=alt.Axis(format="%H:%M", tickCount=12, labelOverlap=True)),
        y=alt.Y("patchtst_low:Q", title="Demand (MW)",
        scale=alt.Scale(zero=False)), y2="patchtst_high:Q")
    lines = data.melt("time", ["patchtst", "actual"], var_name="series", value_name="demand")
    lines["series"] = lines["series"].map({"patchtst": "PatchTST forecast", "actual": "Actual"})
    line = alt.Chart(lines).mark_line(strokeWidth=2).encode(
        x="time:T", y="demand:Q",
        color=alt.Color("series:N", title=None, scale=alt.Scale(domain=["PatchTST forecast", "Actual"],
                                                              range=[colours["line"], colours["actual"]]), legend=alt.Legend(orient="top")),
        tooltip=[alt.Tooltip("time:T", format="%H:%M"), "series:N", alt.Tooltip("demand:Q", format=",.0f")])
    return (band + line).properties(height=300)


st.title("Inside The Transformer")
st.caption("A step by step walk through what PatchTST, the transformer implemented and trained from scratch for this project, does to "
           "forecast one day, using the real inputs and attention of the trained model on a test day. Pick a day, then "
           "step through it.")

loaded = load()
if loaded is None:
    st.info("No walkthrough data yet. Run src/walkthrough.py after src/train_patchtst.py.")
    st.stop()
inputs, attention, forecasts = loaded

days = list(inputs.index)
default = next((i for i, d in enumerate(days) if d.date() == pd.Timestamp("2026-01-05").date()), 0)
day = st.selectbox("Test day", days, index=default, format_func=lambda d: d.strftime("%A %d %B %Y"))
row = inputs.loc[day]
week = week_frame(row)
n_patches = attention.shape[1]
patches = patch_frame(row, n_patches)
cutoff = week["time"].iloc[-1] + pd.Timedelta(minutes=30)
domain = [week["time"].iloc[0].isoformat(), cutoff.isoformat()]
day_attention = attention.loc[day:day + pd.Timedelta(days=1) - pd.Timedelta(minutes=30)]
day_forecast = forecasts.loc[day:day + pd.Timedelta(days=1) - pd.Timedelta(minutes=30)]

st.session_state.setdefault("walk_step", 1)
left, right, _ = st.columns([1, 1, 4])
if left.button("Next step", disabled=st.session_state["walk_step"] >= steps):
    st.session_state["walk_step"] += 1
if right.button("Start again", disabled=st.session_state["walk_step"] == 1):
    st.session_state["walk_step"] = 1
shown = st.session_state["walk_step"]
st.progress(shown / steps, text=f"Step {shown} of {steps}")

st.subheader(step_titles[0])
st.altair_chart(week_chart(week, "demand", "Demand (MW)", cutoff=cutoff, day=day), width="stretch")
st.caption(f"PatchTST reads the {len(week)} half hours of demand up to 08:00 the day before (one week). The 16 hours "
           "between the cutoff and midnight aren't known yet when the forecast is made, so the shaded day is forecast "
           "from this week plus tomorrow's weather and calendar.")

if shown >= 2:
    st.subheader(step_titles[1])
    a, b = st.columns(2)
    a.metric("Week average (subtracted)", f"{row['mean']:,.0f} MW")
    b.metric("Week spread (divided by)", f"{row['std']:,.0f} MW")
    st.altair_chart(week_chart(week, "scaled", "Rescaled demand", domain=domain), width="stretch")
    st.caption("RevIN subtracts the week's average and divides by its spread, so the model learns the shape of demand rather "
               "than its level, and copes when demand is higher in winter or drifts between years. The same numbers are "
               "used to scale the forecast back to MW at the end. Without it, coverage of the 80% range fell to 75.9% in "
               "the ablations.")

if shown >= 3:
    st.subheader(step_titles[2])
    patch_hours = int(row["patch_len"]) // 2
    stride_hours = int(row["stride"]) // 2
    st.altair_chart(week_chart(week, "scaled", "Rescaled demand", domain=domain).properties(height=200), width="stretch")
    st.altair_chart(lanes_chart(patches, domain), width="stretch")
    st.caption(f"The week is cut into {n_patches} patches of {patch_hours} hours, each starting {stride_hours} hours after "
               "the last, so neighbouring patches overlap by half (shown in two rows). Each patch becomes one token, like a "
               "word in a sentence, and the encoder lets every patch compare itself with every other patch of the week.")

if shown >= 4:
    st.subheader(step_titles[3])
    st.altair_chart(heatmap(day_attention, patches), width="stretch")
    times = list(day_attention.index.strftime("%H:%M"))
    choice = st.select_slider("Forecast time to look at", times, value="18:00" if "18:00" in times else times[0])
    weights = day_attention.iloc[times.index(choice)].to_numpy()
    st.altair_chart(week_chart(week, "scaled", "Rescaled demand", domain=domain).properties(height=200), width="stretch")
    st.altair_chart(attention_bars(patches, weights, domain), width="stretch")
    top = patches.assign(weight=weights).nlargest(3, "weight")
    looked = "; ".join(f"{r.start:%a %H:%M} to {r.end:%a %H:%M} ({r.weight:.0%})" for r in top.itertuples())
    st.caption(f"Each of tomorrow's half hours builds a query from its time of day, weather and calendar, and uses cross "
               f"attention to choose which patches of the week to read. At {choice} tomorrow it looks most at: {looked}. "
               "Each bar is drawn over the middle of its patch. Attention shows where the model looks, not proof of what "
               "drives the forecast; the influence map on the Test Year page measures that directly.")

if shown >= 5:
    st.subheader(step_titles[4])
    actual, middle = day_forecast["actual"], day_forecast["patchtst"]
    day_mape = ((middle - actual).abs() / actual).mean() * 100
    inside = ((actual >= day_forecast["patchtst_low"]) & (actual <= day_forecast["patchtst_high"])).mean() * 100
    a, b = st.columns(2)
    a.metric("Error on this day (MAPE)", f"{day_mape:.2f}%")
    b.metric("Half hours inside the 80% range", f"{inside:.0f}%")
    st.altair_chart(output_chart(day_forecast), width="stretch")
    st.caption("The head combines each half hour's summary of the week with its own weather and calendar to give a median "
               "and a 10% to 90% range, which are scaled back to MW using the week's average and spread from step 2. The "
               "range is then widened by the conformal adjustment from the calibration year, so about 80% of actual "
               "values fall inside it over the test year.")