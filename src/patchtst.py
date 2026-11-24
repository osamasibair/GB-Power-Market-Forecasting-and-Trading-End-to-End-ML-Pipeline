"""PatchTST, a transformer that forecasts the 48 half hours of the next day from the past week of demand, with a 10-90% range."""
import math
import time

import numpy as np
import torch
from torch import nn

from forecast_windows import (
    arrays,
    covariate_columns,
    covariate_matrix,
    gap_steps,
    history_steps,
    horizon,
)

quantiles = (0.1, 0.5, 0.9)
poolings = ("flatten", "mean", "attention")


lag_steps = (96, 336)  # the same half hour 2 days and 7 days earlier, both inside the history window


def same_time(history):
    """For each of tomorrow's 48 half hours, the demand at the same half hour 2 days and 7 days earlier, taken from the
    history window, shape (batch, 48, 2). Same time yesterday isn't used, history stops at 08:00 the day before."""
    end = history_steps + gap_steps
    return torch.stack([history[:, end - lag:end - lag + horizon] for lag in lag_steps], dim=-1)


class RevIN(nn.Module):
    """Reversible instance normalisation: scale each week of history to mean 0 and spread 1, then undo it on the forecast,
    so the model learns the shape of demand and copes when the overall level drifts between seasons and years."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def normalise(self, x):
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True, unbiased=False) + 1e-5
        return (x - mean) / std * self.weight + self.bias, mean, std

    def denormalise(self, y, mean, std):
        return (y - self.bias) / (self.weight + 1e-8) * std[..., None] + mean[..., None]


class PatchTST(nn.Module):
    """The transformer reads the past week as 27 overlapping patches. `pooling` decides how those 27 patch summaries
    become forecasts for each of tomorrow's 48 half hours:
    - "flatten": join them into one long vector and pass it through a large linear layer, as in the PatchTST paper
    - "mean": average them into one summary of the week
    - "attention": each forecast half hour builds a query from its time of day, weather and calendar, and uses
      cross attention to pick the patches it needs, so every half hour gets its own summary of the week"""

    def __init__(self, n_covariates, patch_len=24, stride=12, d_model=64, heads=4, layers=2, dropout=0.1, hidden=128,
                 pooling="attention", lags=0, revin=1):
        super().__init__()
        if pooling not in poolings:
            raise ValueError(f"pooling must be one of {poolings}, not {pooling!r}")
        if lags not in (0, 1):
            raise ValueError(f"lags must be 0 or 1, not {lags!r}")
        if revin not in (0, 1):
            raise ValueError(f"revin must be 0 or 1, not {revin!r}")
        self.lags, self.use_revin = lags, revin
        self.patch_len, self.stride, self.pooling = patch_len, stride, pooling
        n_patches = (history_steps - patch_len) // stride + 1
        self.revin = RevIN()
        self.embed = nn.Linear(patch_len, d_model)
        self.position = nn.Parameter(torch.randn(1, n_patches, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, heads, 4 * d_model, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        if pooling == "flatten":
            self.summary = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(n_patches * d_model, hidden), nn.GELU())
        else:
            self.summary = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, hidden), nn.GELU())
        self.covariate = nn.Sequential(nn.Linear(n_covariates + lags * len(lag_steps), hidden // 4), nn.GELU())
        self.step = nn.Parameter(torch.zeros(horizon, hidden // 4))
        self.head = nn.Sequential(nn.Linear(hidden + hidden // 4, hidden), nn.GELU(), nn.Linear(hidden, len(quantiles)))
        if pooling == "attention":
            self.norm = nn.LayerNorm(d_model)
            self.query = nn.Linear(hidden // 4, d_model)
            self.cross = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        if not revin:
            #without revin every week is scaled by the same mean and spread, those of the training data
            self.register_buffer("scale", torch.tensor([0.0, 1.0]))
        self.keep_attention, self.attention = False, None

    def normalise(self, history):
        if self.use_revin:
            return self.revin.normalise(history)
        mean, std = self.scale[0].expand(len(history), 1), self.scale[1].expand(len(history), 1)
        return (history - mean) / std, mean, std

    def denormalise(self, y, mean, std):
        if self.use_revin:
            return self.revin.denormalise(y, mean, std)
        return y * std[..., None] + mean[..., None]

    def forward(self, history, future):
        x, mean, std = self.normalise(history)
        patches = x.unfold(1, self.patch_len, self.stride)
        encoded = self.encoder(self.embed(patches) + self.position)
        if self.lags:
            future = torch.cat([future, same_time(x)], dim=-1)
        ahead = self.covariate(future) + self.step
        if self.pooling == "attention":
            encoded = self.norm(encoded)
            attended, self.attention = self.cross(self.query(ahead), encoded, encoded, need_weights=self.keep_attention)
            week = self.summary(attended)
        else:
            pooled = encoded if self.pooling == "flatten" else encoded.mean(dim=1)
            week = self.summary(pooled)[:, None, :].expand(-1, horizon, -1)
        out = self.head(torch.cat([week, ahead], dim=-1))
        median = out[..., 1]
        low = median - nn.functional.softplus(out[..., 0])
        high = median + nn.functional.softplus(out[..., 2])
        return self.denormalise(torch.stack([low, median, high], dim=-1), mean, std)


def pinball_loss(predicted, actual, scale):
    error = (actual[..., None] - predicted) / scale[..., None]
    q = torch.tensor(quantiles, dtype=predicted.dtype)
    return torch.maximum(q * error, (q - 1) * error).mean()


defaults = {"patch_len": 24, "stride": 12, "d_model": 64, "heads": 4, "layers": 2, "dropout": 0.1, "hidden": 128,
            "pooling": "attention", "lags": 0, "revin": 1, "weather": 1, "solar": 1, "christmas": 1, "learning_rate": 1e-3, "weight_decay": 1e-2, "batch_size": 128, "max_epochs": 40,
            "patience": 6, "warmup": 0.05, "train_stride": 2, "seed": 42}
architecture = ["patch_len", "stride", "d_model", "heads", "layers", "dropout", "hidden", "pooling", "lags", "revin"]


class PatchTSTForecaster:
    def __init__(self, **settings):
        unknown = set(settings) - set(defaults)
        if unknown:
            raise ValueError(f"unknown settings: {sorted(unknown)}")
        self.settings = {**defaults, **settings}

    def tensors(self, df, starts):
        history, future, actual = arrays(df, starts)
        future = (future[..., self.columns()] - self.cov_mean) / self.cov_std
        return torch.from_numpy(history), torch.from_numpy(future), torch.from_numpy(actual)

    def columns(self):
        return covariate_columns(bool(self.settings["weather"]), bool(self.settings["solar"]), bool(self.settings["christmas"]))

    def loss(self, history, future, actual):
        scale = history.std(dim=1, unbiased=False) + 1e-5
        return pinball_loss(self.net(history, future), actual, scale[:, None])

    def fit(self, df, starts, valid=None, epochs=None, verbose=False, callback=None):
        """Train on the windows at `starts`. With `valid`, stop when the loss on those windows stops improving;
        with `epochs`, train for exactly that many (used to refit on all the data once the best length is known), 
        `callback(epoch, loss)` is called after each epoch's validation loss, which lets Optuna stop poor trials early."""
        torch.manual_seed(self.settings["seed"])
        cov = covariate_matrix(df)[starts[:, None] + np.arange(horizon)][..., self.columns()].reshape(-1, len(self.columns()))
        self.cov_mean, self.cov_std = cov.mean(axis=0), cov.std(axis=0) + 1e-6
        history, future, actual = self.tensors(df, starts)
        if valid is not None:
            valid_tensors = self.tensors(df, valid)

        self.net = PatchTST(future.shape[-1], **{name: self.settings[name] for name in architecture})
        if not self.settings["revin"]:
            self.net.scale.copy_(torch.stack([history.mean(), history.std()]))
        s = self.settings
        epochs = epochs or s["max_epochs"]
        steps = epochs * math.ceil(len(history) / s["batch_size"])
        warmup = max(1, int(steps * s["warmup"]))
        optimiser = torch.optim.AdamW(self.net.parameters(), lr=s["learning_rate"], weight_decay=s["weight_decay"])
        schedule = torch.optim.lr_scheduler.LambdaLR(
            optimiser, lambda step: min(1, (step + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(step, steps) / steps)))

        self.valid_losses, best, best_state, waited = [], float("inf"), None, 0
        generator = torch.Generator().manual_seed(s["seed"])
        for epoch in range(1, epochs + 1):
            self.net.train()
            started = time.perf_counter()
            for batch in torch.randperm(len(history), generator=generator).split(s["batch_size"]):
                optimiser.zero_grad()
                loss = self.loss(history[batch], future[batch], actual[batch])
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                optimiser.step()
                schedule.step()
            if valid is None:
                if verbose:
                    print(f"  epoch {epoch}/{epochs} ({time.perf_counter() - started:.0f}s)")
                continue
            self.net.eval()
            with torch.no_grad():
                valid_loss = self.loss(*valid_tensors).item()
            self.valid_losses.append(valid_loss)
            if callback is not None:
                callback(epoch, valid_loss)
            if verbose:
                print(f"  epoch {epoch}: calibration loss {valid_loss:.4f} ({time.perf_counter() - started:.0f}s)")
            if valid_loss < best:
                best, waited = valid_loss, 0
                best_state = {name: value.clone() for name, value in self.net.state_dict().items()}
            else:
                waited += 1
                if waited >= s["patience"]:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.best_epochs = int(np.argmin(self.valid_losses)) + 1 if self.valid_losses else epochs
        self.net.eval()
        return self

    def predict(self, df, starts):
        """Forecasts in MW for each window, shape (windows, 48, 3): the 10%, 50% and 90% quantiles."""
        history, future, _ = self.tensors(df, starts)
        with torch.no_grad():
            return torch.cat([self.net(h, f) for h, f in zip(history.split(1024), future.split(1024))]).numpy()

    def save(self, path, **extra):
        torch.save({"settings": self.settings, "state": self.net.state_dict(), "cov_mean": self.cov_mean.tolist(),
                    "cov_std": self.cov_std.tolist(), "best_epochs": self.best_epochs, **extra}, path)

    @classmethod
    def load(cls, path):
        saved = torch.load(path)
        model = cls(**{"pooling": "flatten", "lags": 0, "revin": 1, "weather": 1, "solar": 1, "christmas": 1, **saved["settings"]})
        model.cov_mean = np.array(saved["cov_mean"], dtype=np.float32)
        model.cov_std = np.array(saved["cov_std"], dtype=np.float32)
        model.best_epochs = saved["best_epochs"]
        model.net = PatchTST(len(model.cov_mean), **{name: model.settings[name] for name in architecture})
        model.net.load_state_dict(saved["state"])
        model.net.eval()
        return model, saved


def chronos_forecast(df, starts, pipeline, context=2048):
    """Zero shot forecasts from a pretrained Chronos model given only past demand, cut off at the same time as the others."""
    demand = df["demand"].to_numpy(dtype=np.float32)
    out = []
    for chunk in np.array_split(starts, max(1, len(starts) // 64)):
        inputs = [torch.from_numpy(demand[max(0, s - gap_steps - context):s - gap_steps]) for s in chunk]
        predicted, _ = pipeline.predict_quantiles(inputs, prediction_length=gap_steps + horizon, quantile_levels=list(quantiles))
        out.append(predicted[:, gap_steps:, :].numpy())
    return np.concatenate(out)