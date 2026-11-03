"""Streamlit entry point, sets up the page and the navigation between the live and research pages."""
import sys
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

st.set_page_config(page_title="GB Power Market Forecasting and Trading", layout="wide")
pages = st.navigation({
    "Demand Forecasting": [
        st.Page("monitoring.py", title="Live", default=True),
        st.Page("transformer.py", title="Test Year"),
    ],
    "Prices and Trading (LightGBM, Test Year)": [
        st.Page("price_forecast.py", title="Price Forecast"),
        st.Page("price_spikes.py", title="Price Spike Classifier"),
        st.Page("battery.py", title="Battery Backtest"),
    ],
})
pages.run()