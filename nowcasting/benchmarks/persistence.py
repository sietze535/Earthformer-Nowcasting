"""Persistence nowcasting baseline: last radar frame held fixed."""

from __future__ import annotations

import numpy as np

try:
    import torch
except ImportError:
    torch = None

MEAN_LOG = 0.03019706713265408
STD_LOG = 0.5392297631902654


def to_mmh_numpy(img: np.ndarray, mean: float = MEAN_LOG, std: float = STD_LOG) -> np.ndarray:
    return np.clip(10 ** (img * std + mean), 0.0, 160.0).astype(np.float32, copy=False)


def run_persistence_mmh(context_mmh: np.ndarray, n_leadtimes: int = 18) -> np.ndarray:
    """Repeat last context frame. ``context_mmh`` is (T, H, W)."""
    last = context_mmh[-1]
    return np.repeat(last[np.newaxis, ...], n_leadtimes, axis=0).astype(np.float32, copy=False)


class PersistenceModel:
    """Repeat the last context radar frame for all forecast lead times."""

    def __init__(self, config):
        self.future_timesteps = int(config.model.get("future_timesteps", 18))
        transforms = getattr(config.dataloader, "var_info", {}).get("transforms", {})
        radar = transforms.get("radar", {}).get("default_rainrate", {})
        self.mean = float(radar.get("mean", MEAN_LOG))
        self.std = float(radar.get("std", STD_LOG))
        self.inputs_are_normalized = bool(config.model.get("inputs_are_normalized", True))

    def to_mmh(self, img, norm_method="zscore", mean=None, std=None):
        mean = self.mean if mean is None else mean
        std = self.std if std is None else std
        if norm_method != "zscore":
            raise ValueError("only zscore supported")
        out = 10 ** (img * std + mean)
        if torch is not None and isinstance(img, torch.Tensor):
            return torch.clamp(out, 0.0, 160.0)
        return np.clip(np.asarray(out), 0.0, 160.0)

    def __call__(self, x, y):
        if torch is not None and isinstance(x, torch.Tensor):
            x_np = x.detach().cpu().numpy()
        else:
            x_np = np.asarray(x)
        if torch is not None and isinstance(y, torch.Tensor):
            y_np = y.detach().cpu().numpy()
        else:
            y_np = np.asarray(y)

        if self.inputs_are_normalized:
            x_mmh = to_mmh_numpy(x_np, self.mean, self.std)
            y_mmh = to_mmh_numpy(y_np, self.mean, self.std)
        else:
            x_mmh, y_mmh = x_np.astype(np.float32), y_np.astype(np.float32)

        last = x_mmh[:, 0:1, -1:, :, :]
        y_hat = np.repeat(last, self.future_timesteps, axis=2)
        return y_mmh[:, 0:1, ...], y_hat.astype(np.float32, copy=False)
