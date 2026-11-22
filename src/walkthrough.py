"""Save what PatchTST does inside for every test day, for the dashboard's "Inside the Transformer" page: the week of
demand it reads, how RevIN rescales that week, and how much each of tomorrow's 48 half hours attends to each of the 27
patches of the week through the cross attention head. The deployed dashboard has no PyTorch, so this is worked out once
here and saved to two tables:
- transformer_inputs: one row per test day, with the week of demand it read and the RevIN mean, spread and weights
- transformer_attention: one row per forecast half hour, with its attention over the 27 patches (adding up to 1)
Run after src/train_patchtst.py, which saves the model this reads:
    python src/walkthrough.py"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from forecast_windows import gap_steps, history_steps, horizon

model_path = Path("models") / "patchtst.pt"


def walkthrough(model, df, starts):
    """The inputs and the cross attention of `model` for the forecast days starting at `starts`."""
    net = model.net
    if net.pooling != "attention":
        raise SystemExit("the walkthrough needs the cross attention head, train PatchTST with pooling attention")
    history, future, _ = model.tensors(df, starts)
    net.keep_attention = True
    weights, means, stds = [], [], []
    with torch.no_grad():
        for h, f in zip(history.split(256), future.split(256)):
            net(h, f)
            weights.append(net.attention)  # averaged over the heads, shape (days, 48, patches)
            _, mean, std = net.normalise(h)
            means.append(mean[:, 0])
            stds.append(std[:, 0])
    net.keep_attention = False
    weights = torch.cat(weights).numpy()

    day_start = df.index[starts]
    inputs = pd.DataFrame(history.numpy(), index=pd.Index(day_start, name="timestamp"),
                          columns=[f"d{i:03d}" for i in range(history_steps)])
    inputs.insert(0, "history_start", df.index[starts - gap_steps - history_steps])
    inputs.insert(1, "mean", torch.cat(means).numpy())
    inputs.insert(2, "std", torch.cat(stds).numpy())
    inputs.insert(3, "revin_weight", float(net.revin.weight.detach()))
    inputs.insert(4, "revin_bias", float(net.revin.bias.detach()))
    inputs.insert(5, "patch_len", net.patch_len)
    inputs.insert(6, "stride", net.stride)

    times = df.index[(np.asarray(starts)[:, None] + np.arange(horizon)).ravel()]
    attention = pd.DataFrame(weights.reshape(-1, weights.shape[-1]), index=pd.Index(times, name="timestamp"),
                             columns=[f"p{i:02d}" for i in range(weights.shape[-1])])
    return inputs, attention


if __name__ == "__main__":
    from db import load_table, save_table
    from forecast_windows import day_windows
    from patchtst import PatchTSTForecaster
    from train import split_date

    if not model_path.exists():
        raise SystemExit(f"{model_path} not found, run src/train_patchtst.py first")
    model, _ = PatchTSTForecaster.load(model_path)
    df = load_table("features")
    starts = day_windows(df.index, split_date)
    inputs, attention = walkthrough(model, df, starts)
    save_table(inputs, "transformer_inputs")
    save_table(attention, "transformer_attention")
    print(f"saved transformer_inputs ({len(inputs)} days) and transformer_attention ({len(attention)} half hours, "
          f"{attention.shape[1]} patches)")