"""Ridge regression baseline on exactly the features LightGBM uses. Comparing the two shows how much of
LightGBM's accuracy comes from the features and how much from LightGBM itself learning non linear effects, such as
temperature mattering differently at different times of day.

A linear model can't learn that 6pm differs from 3am from a number, so the half hour of the day and the day of the week
are one hot encoded, and the day of the year is turned into a sine and cosine so 31 December sits next to 1 January. The
other features are standardised, so the penalty treats them alike. Ridge rather than plain linear regression, because
the demand lags move together, and the penalty keeps their coefficients stable.

The penalty strength is chosen on the calibration year, like every other choice: models fitted on data before it are
scored on it, and the best one is kept. The model is then refitted on everything before the test year. Its 80% range is
the forecast plus or minus the 80th percentile of its absolute errors on the calibration year, the simplest conformal
interval. Results are added as columns to transformer_forecasts and transformer_calibration, so run
src/train_patchtst.py first (it rewrites those tables), then this.
    python src/ridge.py"""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from train import features, target

categories = ["period", "dayofweek"]
numeric = [f for f in features if f not in categories + ["dayofyear"]] + ["year_sin", "year_cos"]
alphas = np.logspace(-2, 3, 11)


def design(df):
    """The LightGBM features, with the day of the year as a position on a circle."""
    angle = 2 * np.pi * df["dayofyear"] / 365.25
    return df[features].assign(year_sin=np.sin(angle), year_cos=np.cos(angle))


def make_model(alpha):
    encode = ColumnTransformer([("categories", OneHotEncoder(handle_unknown="ignore"), categories),
                                ("numeric", StandardScaler(), numeric)])
    return make_pipeline(encode, Ridge(alpha=alpha))


def mape(actual, predicted):
    return float(np.mean(np.abs(predicted - actual) / actual) * 100)


def choose_alpha(fit_part, calib):
    """The penalty with the lowest calibration-year error, for models fitted only on data before the calibration year."""
    scores = {float(alpha): mape(calib[target].to_numpy(), make_model(alpha).fit(design(fit_part), fit_part[target])
                                 .predict(design(calib))) for alpha in alphas}
    return min(scores, key=scores.get), scores


def ridge_forecasts(df, calib, test, calibration_start, split_date, coverage=0.8):
    """Calibration year forecasts from the model fitted before it, test forecasts from the model fitted on everything
    before the test year, each with an 80% range from the calibration year errors."""
    fit_part = df[df.index < calibration_start]
    alpha, scores = choose_alpha(fit_part, calib)
    early = make_model(alpha).fit(design(fit_part), fit_part[target]).predict(design(calib))
    width = float(np.quantile(np.abs(calib[target].to_numpy() - early), coverage))
    train = df[df.index < split_date]
    point = make_model(alpha).fit(design(train), train[target]).predict(design(test))
    return {
        "alpha": alpha,
        "scores": scores,
        "calibration": {"ridge": early, "ridge_low": early - width, "ridge_high": early + width},
        "forecasts": {"ridge": point, "ridge_low": point - width, "ridge_high": point + width},
    }


if __name__ == "__main__":
    from chronos2 import add_columns, scored
    from db import load_table, save_table
    from forecast_windows import day_windows
    from quantiles import calibration_start
    from tracking import log_run
    from train import split_date
    from train_patchtst import rows

    df = load_table("features")
    calib = df.iloc[rows(day_windows(df.index, calibration_start, split_date))]
    test = df.iloc[rows(day_windows(df.index, split_date))]
    result = ridge_forecasts(df, calib, test, calibration_start, split_date)
    print("calibration-year MAPE by penalty: " + ", ".join(f"{a:g}: {s:.2f}%" for a, s in result["scores"].items()))
    print(f"chosen penalty {result['alpha']:g}")

    forecasts_table = add_columns(load_table("transformer_forecasts"), result["forecasts"])
    calibration_table = add_columns(load_table("transformer_calibration"), result["calibration"])
    save_table(forecasts_table, "transformer_forecasts")
    save_table(calibration_table, "transformer_calibration")
    print("added ridge to transformer_forecasts and transformer_calibration")

    everything = scored(forecasts_table, ["baseline", "ridge", "lightgbm"])
    print(f"\n{'model':10} {'MAE':>7} {'RMSE':>7} {'MAPE':>7} {'in range':>9} {'width':>8}")
    for name, m in everything.items():
        interval = f"{m['coverage']:8.1f}% {m['width']:6.0f} MW" if "coverage" in m else f"{'-':>9} {'-':>8}"
        print(f"{name:10} {m['mae']:7.0f} {m['rmse']:7.0f} {m['mape']:6.2f}% {interval}")

    log_run(experiment="demand-ridge", params={"alpha": result["alpha"], "features": ",".join(features)},
            metrics={f"ridge_{k}": v for k, v in everything["ridge"].items()}, files=[])