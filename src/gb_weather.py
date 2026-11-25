"""Weather for the whole of Great Britain, not just London: one weather point in each of the 11 regions of Great
Britain (the 9 English regions, Wales and Scotland), averaged with weights that follow what each variable does to
national demand.

- Temperature, wind speed, humidity and cloud cover are weighted by each region's population, since heating, cooling
  and lighting follow people.
- Sunshine (shortwave radiation) is weighted by each region's installed solar capacity, since sunshine lowers the
  demand the grid sees through rooftop and other embedded solar panels, and that depends on where the panels are.

The weights are fixed, from figures published before the test year began (September 2025), so no later information
leaks in: ONS mid-2023 population estimates (published July and October 2024) and DESNZ installed solar capacity at
the end of 2023. The source spreadsheets are kept in reference/, and running this file recomputes
reference/weather_weights.csv from them:
    python src/gb_weather.py
Reading the spreadsheets needs openpyxl (in requirements-research.txt); using the saved weights doesn't."""
from pathlib import Path

import numpy as np
import pandas as pd

reference = Path("reference")
weights_path = reference / "weather_weights.csv"
uk_population_file = reference / "ons_population_mid2023.xlsx"
england_wales_population_file = reference / "ons_population_england_wales_mid2023.xlsx"
solar_file = reference / "desnz_regional_capacity_2003_2025.xlsx"
solar_year = 2023
# Every weather request names this model instead of leaving the choice to Open-Meteo's "best match", which switches
# source in March 2022 (ECMWF before, the Met Office's UKV model after). The Met Office model's archive starts on
# 1 March 2022, it is what best match uses for Great Britain today (so training and live weather are made the same way),
# and its earlier runs, needed to rescore on day-ahead weather, are archived from August 2024.
weather_model = "ukmo_seamless"

# region, ONS area code, the region's name in the DESNZ spreadsheet, and the city used as its weather point
regions = pd.DataFrame([
    ("North East", "E12000001", "North East", "Newcastle", 54.9783, -1.6178),
    ("North West", "E12000002", "North West", "Manchester", 53.4808, -2.2426),
    ("Yorkshire and the Humber", "E12000003", "Yorkshire and Humber", "Leeds", 53.8008, -1.5491),
    ("East Midlands", "E12000004", "East Midlands", "Nottingham", 52.9548, -1.1581),
    ("West Midlands", "E12000005", "West Midlands", "Birmingham", 52.4862, -1.8904),
    ("East of England", "E12000006", "East of England", "Cambridge", 52.2053, 0.1218),
    ("London", "E12000007", "London", "London", 51.5072, -0.1276),
    ("South East", "E12000008", "South East", "Reading", 51.4543, -0.9781),
    ("South West", "E12000009", "South West", "Bristol", 51.4545, -2.5879),
    ("Wales", "W92000004", "Wales", "Cardiff", 51.4816, -3.1791),
    ("Scotland", "S92000003", "Scotland", "Glasgow", 55.8642, -4.2518),
], columns=["region", "ons_code", "desnz_name", "city", "latitude", "longitude"])

by_population = ["temperature_2m", "wind_speed_10m", "relative_humidity_2m", "cloud_cover"]
by_solar = ["shortwave_radiation"]
variables = by_population + by_solar  # every weather variable downloaded for each region


def population():
    """Mid-2023 population of each region: the English regions and Wales from the England and Wales tables, Scotland
    from the UK tables (the England and Wales tables don't cover it)."""
    frames = [pd.read_excel(path, sheet_name="MYE2 - Persons", header=7, usecols=["Code", "All ages"])
              for path in (england_wales_population_file, uk_population_file)]
    counts = pd.concat(frames).drop_duplicates("Code").set_index("Code")["All ages"]
    return regions["ons_code"].map(counts)


def solar_capacity():
    """Installed solar capacity (MW) of each region at the end of `solar_year`."""
    table = pd.read_excel(solar_file, sheet_name=f"MW{solar_year}", header=6).set_index("Region")
    return regions["desnz_name"].map(table["Solar PV"])


def compute_weights():
    weights = regions[["region", "city", "latitude", "longitude"]].copy()
    weights["population"] = population().to_numpy()
    weights["solar_mw"] = solar_capacity().to_numpy()
    if weights[["population", "solar_mw"]].isna().any().any():
        missing = weights.loc[weights[["population", "solar_mw"]].isna().any(axis=1), "region"].tolist()
        raise SystemExit(f"no figures found for {missing}, check the spreadsheets in {reference}/")
    weights["population_weight"] = weights["population"] / weights["population"].sum()
    weights["solar_weight"] = weights["solar_mw"] / weights["solar_mw"].sum()
    return weights


def load_weights():
    return pd.read_csv(weights_path)


def gb_average(points, weights=None):
    """One GB-wide weather series from the weather at every region's point. `points` has a time column, a region column
    and the weather columns; the result has one row per time, with the time as its index. Every time must have every
    region, so a missing region shows up as an error instead of quietly shifting the average."""
    weights = load_weights() if weights is None else weights
    wide = points.pivot(index="time", columns="region")
    regions_seen = set(wide.columns.get_level_values("region"))
    if regions_seen != set(weights["region"]):
        raise ValueError(f"weather points don't match the weights: missing {sorted(set(weights['region']) - regions_seen)}")
    incomplete = wide.isna().any(axis=1)
    if incomplete.any():
        raise ValueError(f"{incomplete.sum()} times are missing weather for at least one region, "
                         f"first at {wide.index[incomplete][0]}")
    weight = weights.set_index("region")
    out = {}
    for column in by_population + by_solar:
        kind = "population_weight" if column in by_population else "solar_weight"
        values = wide[column][weight.index]
        out[column] = values.to_numpy() @ weight[kind].to_numpy()
    return pd.DataFrame(out, index=wide.index).sort_index()


if __name__ == "__main__":
    weights = compute_weights()
    weights.to_csv(weights_path, index=False)
    shown = pd.DataFrame({"region": weights["region"], "city": weights["city"],
                          "population": weights["population"].map("{:,.0f}".format),
                          "solar MW": weights["solar_mw"].map("{:,.0f}".format),
                          "population %": (weights["population_weight"] * 100).round(1),
                          "solar %": (weights["solar_weight"] * 100).round(1)})
    print(shown.to_string(index=False))
    print(f"\npopulation {weights['population'].sum():,.0f}, solar {weights['solar_mw'].sum():,.0f} MW, "
          f"weights add up to {weights['population_weight'].sum():.3f} and {weights['solar_weight'].sum():.3f}")
    assert np.isclose(weights["population_weight"].sum(), 1) and np.isclose(weights["solar_weight"].sum(), 1)
    print(f"saved {weights_path}")