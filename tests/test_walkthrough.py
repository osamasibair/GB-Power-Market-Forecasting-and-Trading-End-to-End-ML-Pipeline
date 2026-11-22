import numpy as np
import pytest
from test_patchtst import make_features

from forecast_windows import day_windows, gap_steps, history_steps, horizon
from patchtst import PatchTSTForecaster
from walkthrough import walkthrough


def test_the_walkthrough_gives_each_days_inputs_and_attention_that_adds_up_to_one():
    df = make_features(days=40)
    starts = day_windows(df.index)
    model = PatchTSTForecaster(max_epochs=1, d_model=16, hidden=32).fit(df, starts[:-3])
    inputs, attention = walkthrough(model, df, starts[-3:])
    patches = (history_steps - model.net.patch_len) // model.net.stride + 1

    assert len(inputs) == 3 and len(attention) == 3 * horizon
    assert attention.shape[1] == patches
    assert np.allclose(attention.sum(axis=1), 1, atol=1e-5)
    assert (inputs["history_start"] == df.index[starts[-3:] - gap_steps - history_steps]).all()
    week = inputs.filter(regex=r"^d\d{3}$").to_numpy()
    assert np.allclose(week[0], df["demand"].to_numpy()[starts[-3] - gap_steps - history_steps:starts[-3] - gap_steps])
    assert np.allclose(inputs["mean"], week.mean(axis=1), rtol=1e-5)
    assert model.net.keep_attention is False


def test_the_walkthrough_needs_the_cross_attention_head():
    df = make_features(days=30)
    starts = day_windows(df.index)
    model = PatchTSTForecaster(max_epochs=1, d_model=16, hidden=32, pooling="mean").fit(df, starts[:-2])
    with pytest.raises(SystemExit):
        walkthrough(model, df, starts[-2:])