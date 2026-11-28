import numpy as np

from final_model import combine, pack, seed_runs, unpack


def test_forecasts_survive_being_stored_as_text():
    array = np.random.default_rng(0).normal(30000, 2000, (500, 3))
    assert np.array_equal(unpack(pack(array)), array)


def test_every_seed_trains_on_both_periods_and_the_runs_come_back_in_seed_order():
    seen = []

    def train_and_forecast(seed, part):
        seen.append((seed, part))
        return np.full((4, 3), seed + (0.5 if part == "test" else 0.0))

    calibration, test = seed_runs(train_and_forecast, [42, 1, 2])
    assert sorted(seen) == sorted((s, p) for s in (42, 1, 2) for p in ("calibration", "test"))
    assert [run[0, 0] for run in calibration] == [42, 1, 2] and [run[0, 0] for run in test] == [42.5, 1.5, 2.5]


def test_the_seeds_are_averaged_and_the_range_is_calibrated_to_80_percent_and_never_crosses_the_median():
    rng = np.random.default_rng(1)
    actual = rng.normal(30000, 2000, (2, 2000))
    medians = [a + rng.normal(0, 800, a.shape) for a in actual for _ in range(3)]
    runs = [np.stack([m - 500, m, m + 500], axis=1) for m in medians]
    result = combine("m_tuned", runs[:3], runs[3:], actual[0], actual[1])
    calib = result["calibration"]
    assert np.allclose(calib["m_tuned"], np.mean([r[:, 1] for r in runs[:3]], axis=0))
    inside = (actual[0] >= calib["m_tuned_low"]) & (actual[0] <= calib["m_tuned_high"])
    assert 0.8 <= inside.mean() < 0.82  # 80% after calibration; keeping the median inside can only widen it
    for part in ("calibration", "forecasts"):
        r = result[part]
        assert (r["m_tuned_low"] <= r["m_tuned"]).all() and (r["m_tuned"] <= r["m_tuned_high"]).all()
    assert len(result["seed_mapes"]) == 3
