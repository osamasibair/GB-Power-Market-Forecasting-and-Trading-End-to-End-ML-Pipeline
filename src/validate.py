"""Checking raw data quality before it reaches the model."""

from pathlib import Path

import pandas as pd

from gb_weather import regions as weather_points

raw = Path("data/raw")

demand_min, demand_max = 10000, 60000
temp_min, temp_max = -20, 45

def check_demand(df):
    problems = []
    counts = df.groupby("settlementDate").size()
    odd = counts[~counts.isin([46, 48, 50])]
    if len(odd) > 0:
        problems.append(f"{len(odd)} dates with unexpected period counts: {odd.head().to_dict()}")
    dates = pd.to_datetime(df["settlementDate"])
    expected = pd.date_range(dates.min(), dates.max(), freq="D")
    missing = len(expected) - dates.nunique()
    if missing > 0:
        problems.append(f"{missing} dates missing entirely from the series")
    dupes = df.duplicated(subset=["settlementDate", "settlementPeriod"]).sum()
    if dupes > 0:
        problems.append(f"{dupes} duplicate settlement date/period rows")
    nulls = df["initialDemandOutturn"].isna().sum()
    if nulls > 0:
        problems.append(f"{nulls} missing demand values")
    outside = df[(df["initialDemandOutturn"] < demand_min) | (df["initialDemandOutturn"] > demand_max)]
    if len(outside) > 0:
        problems.append(f"{len(outside)} demand values outside {demand_min}-{demand_max} MW")
    return problems

def check_weather(df):
    """Checks the weather at every region's point: all regions present, each with the same unbroken hourly series."""
    problems = []
    expected = set(weather_points["region"])
    missing = expected - set(df["region"])
    if missing:
        problems.append(f"no weather for {', '.join(sorted(missing))}")
    lengths = df.groupby("region").size()
    if lengths.nunique() > 1:
        problems.append(f"regions have different numbers of hours: {lengths.min()} to {lengths.max()}")
    for region, part in df.groupby("region"):
        gaps = pd.to_datetime(part["time"]).diff().dropna()
        irregular = (gaps != pd.Timedelta(hours=1)).sum()
        if irregular > 0:
            problems.append(f"{region}: {irregular} gaps or overlaps in hourly sequence")
    nulls = df[["temperature_2m", "wind_speed_10m", "shortwave_radiation"]].isna().sum()
    for column, count in nulls[nulls > 0].items():
        regions = ", ".join(sorted(df.loc[df[column].isna(), "region"].unique()))
        problems.append(f"{count} missing {column} values ({regions})")
    outside = df[(df["temperature_2m"] < temp_min) | (df["temperature_2m"] > temp_max)]
    if len(outside) > 0:
        n = len(outside)
        problems.append(f"{n} temperature{'s' if n != 1 else ''} outside {temp_min}-{temp_max} C")
    return problems

def check_coverage(demand, weather):
    problems = []
    d_start = pd.to_datetime(demand["startTime"]).min()
    d_end = pd.to_datetime(demand["startTime"]).max()
    w_start = pd.to_datetime(weather["time"]).min()
    w_end = pd.to_datetime(weather["time"]).max()
    if abs((d_start - w_start).days) > 1:
        problems.append(f"start dates differ: demand {d_start.date()}, weather {w_start.date()}")
    if abs((d_end - w_end).days) > 1:
        problems.append(f"end dates differ: demand {d_end.date()}, weather {w_end.date()}")
    return problems

def check_prices(prices):
    problems = []
    times = pd.DatetimeIndex(pd.to_datetime(prices["startTime"]))
    full = pd.date_range(times.min(), times.max(), freq="30min")
    missing = full.difference(times)
    if len(missing):
        problems.append(f"{len(missing)} missing half-hours: {list(missing.strftime('%Y-%m-%d %H:%M'))}")
    dupes = times.duplicated().sum()
    if dupes:
        problems.append(f"{dupes} duplicate timestamps")
    nulls = prices[["price", "volume"]].isna().sum().sum()
    if nulls:
        problems.append(f"{nulls} null values")
    zero_volume = (prices["volume"] == 0).sum()
    if zero_volume:
        problems.append(f"{zero_volume} half-hours with zero volume")
    return problems

def check_generation(generation):
    problems = []
    dupes = generation.duplicated(["startTime", "psrType"]).sum()
    if dupes:
        problems.append(f"{dupes} duplicate rows (republished data)")
    latest = generation.sort_values("publishTime").drop_duplicates(["startTime", "psrType"], keep="last")
    times = pd.DatetimeIndex(pd.to_datetime(latest["startTime"].unique()))
    full = pd.date_range(times.min(), times.max(), freq="30min")
    missing = full.difference(times)
    if len(missing):
        runs = missing.to_series().diff().ne(pd.Timedelta(minutes=30)).cumsum()
        problems.append(f"{len(missing)} missing half-hours, longest gap {runs.value_counts().max()} half-hours")
    negative = (latest["quantity"] < 0).sum()
    if negative:
        problems.append(f"{negative} negative generation values")
    return problems

if __name__ == "__main__":
    demand = pd.read_csv(raw / "demand.csv")
    weather = pd.read_csv(raw / "weather.csv")
    print(f"demand: {len(demand)} rows, weather: {len(weather)} rows")
    problems = check_demand(demand) + check_weather(weather)
    if problems:
        for p in problems:
            print(f"  ! {p}")
    else:
        print("  all checks passed")

    prices = pd.read_csv(raw / "prices.csv")
    print(f"prices: {len(prices)} rows")
    price_problems = check_prices(prices)
    for p in price_problems:
        print(f"  ! {p}")
    if not price_problems:
        print("  all price checks passed")
    print(prices["price"].describe().round(2))
    print(f"negative prices: {(prices['price'] < 0).sum()} half-hours")

    generation = pd.read_csv(raw / "generation.csv")
    print(f"generation: {len(generation)} rows")
    gen_problems = check_generation(generation)
    for p in gen_problems:
        print(f"  ! {p}")
    if not gen_problems:
        print("  all generation checks passed")
    

        

