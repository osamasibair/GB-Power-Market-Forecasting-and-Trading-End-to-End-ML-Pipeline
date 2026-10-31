"""Blend the LightGBM and PatchTST forecasts, choosing the blend weight on the calibration year and scoring it on the test year."""
import argparse
import itertools

import numpy as np
import pandas as pd

from db import load_table, save_table
from evaluate import metrics
from tracking import log_run

weights = np.round(np.arange(0, 1.0001, 0.05), 2)


def conformal_adjustment(actual, low, high, coverage=0.8):
    return float(np.quantile(np.maximum(low - actual, actual - high), coverage))


def mix(frame, shares):
    """The weighted average of several models middle forecasts and of their ranges, for shares {model: weight}."""
    return pd.DataFrame({
        f"ensemble{part}": sum(weight * frame[f"{model}{part}"] for model, weight in shares.items())
        for part in ("", "_low", "_high")
    }, index=frame.index)


def blend(frame, weight):
    """The weighted average of LightGBM and PatchTST, with `weight` the PatchTST share."""
    return mix(frame, {"patchtst": weight, "lightgbm": 1 - weight})


def score(calib, test, shares):
    """The error on the calibration year (used to choose the shares) and the error, coverage and width on the test year,
    with the range recalibrated on the calibration year."""
    c, t = mix(calib, shares), mix(test, shares)
    adjustment = conformal_adjustment(calib["actual"], c["ensemble_low"], c["ensemble_high"])
    low, high = t["ensemble_low"] - adjustment, t["ensemble_high"] + adjustment
    return {
        "calibration_mape": metrics(calib["actual"], c["ensemble"])["mape"],
        "test_mape": metrics(test["actual"], t["ensemble"])["mape"],
        "test_mae": metrics(test["actual"], t["ensemble"])["mae"],
        "adjustment": adjustment,
        "test_coverage": ((test["actual"] >= low) & (test["actual"] <= high)).mean() * 100,
        "test_width": (high - low).mean(),
    }


def weight_table(calib, test):
    """For every PatchTST share: the calibration year error and the test year error, coverage and width."""
    return pd.DataFrame([{"weight": weight, **score(calib, test, {"patchtst": weight, "lightgbm": 1 - weight})}
                         for weight in weights])


def three_way_table(calib, test, third):
    """Every blend of LightGBM, PatchTST and `third` with weights in steps of 0.05 that add up to 1."""
    rows = []
    for patchtst in weights:
        for other in weights:
            if patchtst + other > 1.0001:
                continue
            shares = {"lightgbm": round(1 - patchtst - other, 2), "patchtst": patchtst, third: other}
            rows.append({"lightgbm_weight": shares["lightgbm"], "patchtst_weight": patchtst, "third_weight": other,
                         **score(calib, test, shares)})
    return pd.DataFrame(rows)


def all_blends(calib, test, models, steps=20):
    """Every blend of `models` with weights in steps of 1/`steps` that add up to 1, each scored like the others."""
    rows = []
    for parts in itertools.product(range(steps + 1), repeat=len(models) - 1):
        if sum(parts) > steps:
            continue
        shares = dict(zip(models, [part / steps for part in parts] + [(steps - sum(parts)) / steps]))
        rows.append({**shares, **score(calib, test, {model: w for model, w in shares.items() if w > 0})})
    return pd.DataFrame(rows).sort_values("calibration_mape", ignore_index=True)


def best_weight(table):
    return float(table.loc[table["calibration_mape"].idxmin(), "weight"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--third", default="chronos2_finetuned", help="the third model to try in a three way blend")
    parser.add_argument("--search-all", action="store_true",
                        help="also try every blend of every model with calibration year forecasts, to check nothing beats the three way blend")
    args = parser.parse_args()
    calib, test = load_table("transformer_calibration"), load_table("transformer_forecasts")
    table = weight_table(calib, test)
    chosen = best_weight(table)
    by_weight = table.set_index("weight")

    print(f"{'PatchTST share':>15} {'calibration MAPE':>17} {'test MAPE':>10} {'in range':>9} {'width':>9}")
    for weight, row in by_weight.iterrows():
        marker = "  <- chosen on the calibration year" if weight == chosen else ""
        print(f"{weight:15.2f} {row['calibration_mape']:16.2f}% {row['test_mape']:9.2f}% {row['test_coverage']:8.1f}% "
              f"{row['test_width']:6.0f} MW{marker}")

    lightgbm, patchtst, ensemble = by_weight.loc[0.0], by_weight.loc[1.0], by_weight.loc[chosen]
    print(f"\nchosen blend: {1 - chosen:.0%} LightGBM + {chosen:.0%} PatchTST")
    print(f"test MAPE: LightGBM {lightgbm['test_mape']:.2f}%, PatchTST {patchtst['test_mape']:.2f}%, blend {ensemble['test_mape']:.2f}%")

    save_table(table, "ensemble_weights")
    print("saved ensemble_weights for the dashboard")

    log_run(
        experiment="demand-ensemble",
        params={"patchtst_weight": chosen, "weights_tried": len(weights)},
        metrics={"ensemble_test_mape": ensemble["test_mape"], "lightgbm_test_mape": lightgbm["test_mape"],
                 "patchtst_test_mape": patchtst["test_mape"], "ensemble_test_coverage": ensemble["test_coverage"],
                 "ensemble_test_width": ensemble["test_width"]},
        files=[],
    )

    if args.third in calib and args.third in test:
        three = three_way_table(calib, test, args.third)
        best = three.loc[three["calibration_mape"].idxmin()]
        alone = three[(three["third_weight"] == 1.0)].iloc[0]
        print(f"\nthree way blend with {args.third}, weights chosen on the calibration year "
              f"(calibration MAPE {best['calibration_mape']:.2f}%):")
        print(f"  {best['lightgbm_weight']:.0%} LightGBM + {best['patchtst_weight']:.0%} PatchTST + {best['third_weight']:.0%} {args.third}")
        print(f"  test MAPE {best['test_mape']:.2f}% (two way blend {ensemble['test_mape']:.2f}%, {args.third} alone "
              f"{alone['test_mape']:.2f}%), in range {best['test_coverage']:.1f}%, width {best['test_width']:.0f} MW")
        print(f"  best possible on the test year: {three['test_mape'].min():.2f}% (for reference only, not a fair choice)")
        log_run(
            experiment="demand-ensemble",
            params={"third": args.third, "lightgbm_weight": best["lightgbm_weight"],
                    "patchtst_weight": best["patchtst_weight"], "third_weight": best["third_weight"]},
            metrics={"three_way_test_mape": best["test_mape"], "three_way_test_coverage": best["test_coverage"],
                     "three_way_test_width": best["test_width"], "two_way_test_mape": ensemble["test_mape"]},
            files=[],
        )
    else:
        print(f"\n{args.third} not in the tables yet, run src/chronos2.py for the three way blend")


    if args.search_all:
        models = [m for m in ("lightgbm", "patchtst", "chronos2_finetuned", "chronos2_covariates", "chronos2")
                  if m in calib and m in test]
        search = all_blends(calib, test, models)
        print(f"\nevery blend of {', '.join(models)} in steps of 5% ({len(search):,} blends), best five on the calibration year:")
        print(search.head(5)[models + ["calibration_mape", "test_mape", "test_coverage"]].round(3).to_string(index=False))
        save_table(search, "ensemble_search")
        print("saved ensemble_search")