# following https://pysteps.readthedocs.io/en/stable/auto_examples/plot_steps_nowcast.html
"""PySTEPS / S-PROG wrappers for thesis classical baselines."""

from __future__ import annotations

from datetime import timedelta

import dask
import numpy as np

try:
    import torch
except ImportError:
    torch = None

from pysteps import nowcasts
from pysteps.motion.lucaskanade import dense_lucaskanade
from pysteps.utils import transformation

MEAN_LOG = 0.03019706713265408
STD_LOG = 0.5392297631902654


def to_mmh_numpy(img: np.ndarray, mean: float = MEAN_LOG, std: float = STD_LOG) -> np.ndarray:
    return np.clip(10 ** (img.astype(np.float64) * std + mean), 0.0, 160.0).astype(np.float32)


def run_sprog_mmh(
    context_mmh: np.ndarray,
    n_leadtimes: int = 18,
    *,
    n_cascade_levels: int = 6,
    ar_order: int = 2,
    precip_thr: float = -10.0,
    zerovalue: float = -15.0,
    decomp_method: str = "fft",
    bandpass_filter: str = "gaussian",
    probmatching_method=None,
) -> np.ndarray:
    """Deterministic S-PROG on one sample. ``context_mmh`` is (T, H, W) in mm/h."""
    import contextlib
    import io
    import os

    R, _ = transformation.dB_transform(context_mmh, threshold=0.1, zerovalue=zerovalue)
    R[~np.isfinite(R)] = zerovalue
    if (R == zerovalue).all():
        return np.zeros((n_leadtimes,) + context_mmh.shape[1:], dtype=np.float32)

    try:
        # PySTEPS prints cascade diagnostics to stdout; silence for bulk eval.
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            V = dense_lucaskanade(R, verbose=False)
            R_f = nowcasts.get_method("sprog")(
                R,
                V,
                n_leadtimes,
                precip_thr=precip_thr,
                n_cascade_levels=n_cascade_levels,
                ar_order=ar_order,
                decomp_method=decomp_method,
                bandpass_filter_method=bandpass_filter,
                probmatching_method=probmatching_method,
                measure_time=False,
            )
    except (ValueError, RuntimeError, IndexError, np.linalg.LinAlgError):
        return np.zeros((n_leadtimes,) + context_mmh.shape[1:], dtype=np.float32)

    R_f = transformation.dB_transform(R_f, threshold=precip_thr, inverse=True)[0]
    R_f = np.nan_to_num(R_f, nan=0.0)
    return R_f.astype(np.float32, copy=False)


class PySTEPSModel:
    """Callable wrapper used by ``nowcasting.scripts.eval`` (mm/h I/O for thesis)."""

    def __init__(self, config):
        self.method_name = str(config.model.get("method", "sprog")).lower()
        self.nowcast_method = nowcasts.get_method(self.method_name)
        self.future_timesteps = int(config.model.get("future_timesteps", 18))
        self.km_per_pixel = float(config.model.get("km_per_pixel", 1.0))
        self.interval = timedelta(minutes=5)

        self.ensemble_size = int(config.model.get("ensemble_size", 1))
        self.n_cascade_levels = int(config.model.get("n_cascade_levels", 6))
        self.noise_method = config.model.get("noise_method", "nonparametric")
        self.vel_pert_method = config.model.get("vel_pert_method", "bps")
        self.mask_method = config.model.get("mask_method", "incremental")

        self.ar_order = int(config.model.get("ar_order", 2))
        self.decomp_method = config.model.get("decomp_method", "fft")
        self.bandpass_filter = config.model.get("bandpass_filter", "gaussian")
        self.probmeth = config.model.get("probmatching_method", None)

        self.transform_to_rainrate = config.model.get("transform_to_rainrate", None)
        self.transform_from_rainrate = config.model.get("transform_from_rainrate", None)
        self.inputs_are_normalized = bool(config.model.get("inputs_are_normalized", True))
        self.verbose = bool(config.model.get("verbose", False))
        self.parallel = bool(config.model.get("parallel", False))

        transforms = getattr(config.dataloader, "var_info", {}).get("transforms", {})
        radar = transforms.get("radar", {}).get("default_rainrate", {})
        self.mean = float(radar.get("mean", MEAN_LOG))
        self.std = float(radar.get("std", STD_LOG))

    def to_mmh(self, img, norm_method="zscore", mean=None, std=None):
        mean = self.mean if mean is None else mean
        std = self.std if std is None else std
        if norm_method != "zscore":
            raise ValueError("only zscore supported")
        out = 10 ** (img * std + mean)
        if torch is not None and isinstance(img, torch.Tensor):
            return torch.clamp(out, 0.0, 160.0)
        return np.clip(np.asarray(out), 0.0, 160.0)

    def zero_prediction(self, R: np.ndarray, zerovalue: float) -> np.ndarray:
        spatial_shape = R.shape[1:]
        if self.method_name == "sprog":
            out_shape = (self.future_timesteps,) + spatial_shape + (1,)
        else:
            out_shape = (self.future_timesteps,) + spatial_shape + (self.ensemble_size,)
        return np.full(out_shape, zerovalue, dtype=R.dtype)

    def predict_sample(self, x: np.ndarray, threshold: float = -10.0, zerovalue: float = -15.0) -> np.ndarray:
        """``x`` is (T, H, W) in mm/h."""
        if self.transform_to_rainrate is not None:
            R = self.transform_to_rainrate(x)
        else:
            R = x

        if self.method_name == "sprog":
            arr = run_sprog_mmh(
                R,
                n_leadtimes=self.future_timesteps,
                n_cascade_levels=self.n_cascade_levels,
                ar_order=self.ar_order,
                precip_thr=threshold,
                zerovalue=zerovalue,
                decomp_method=self.decomp_method,
                bandpass_filter=self.bandpass_filter,
                probmatching_method=self.probmeth,
            )
            return arr[..., np.newaxis]

        (R, _) = transformation.dB_transform(R, threshold=0.1, zerovalue=zerovalue)
        if self.verbose:
            print("R stats:", R.shape, np.min(R), np.max(R), np.isnan(R).sum())
        R[~np.isfinite(R)] = zerovalue
        if (R == zerovalue).all():
            R_f = self.zero_prediction(R, zerovalue)
        else:
            V = dense_lucaskanade(R, verbose=False)
            try:
                arr = self.nowcast_method(
                    R,
                    V,
                    self.future_timesteps,
                    n_ens_members=self.ensemble_size,
                    n_cascade_levels=self.n_cascade_levels,
                    precip_thr=threshold,
                    kmperpixel=self.km_per_pixel,
                    timestep=self.interval.total_seconds() / 60,
                    noise_method=self.noise_method,
                    vel_pert_method=self.vel_pert_method,
                    mask_method=self.mask_method,
                    probmatching_method=self.probmeth,
                    num_workers=self.future_timesteps,
                )
                R_f = arr.transpose(1, 2, 3, 0)
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
                zero_error = (
                    str(e).endswith("contains non-finite values")
                    or str(e).startswith("zero-size array to reduction operation")
                    or str(e).endswith("nonstationary AR(p) process")
                    or str(e).endswith("Singular matrix")
                )
                if zero_error:
                    R_f = self.zero_prediction(R, zerovalue)
                else:
                    raise
            R_f = transformation.dB_transform(R_f, threshold=threshold, inverse=True)[0]
            R_f[np.isnan(R_f)] = 0

        if self.transform_from_rainrate is not None:
            R_f = self.transform_from_rainrate(R_f)
        return R_f

    def __call__(self, x, y, parallel: bool | None = None):
        if parallel is None:
            parallel = self.parallel

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

        x_in = np.transpose(x_mmh, (0, 2, 3, 4, 1))
        pred = self.predict_sample
        if parallel:
            pred = dask.delayed(pred)
        y_hat = [pred(x_in[i, :, :, :, 0]) for i in range(x_in.shape[0])]
        if parallel:
            y_hat = dask.compute(y_hat, scheduler="threads", num_workers=len(y_hat))[0]
        y_hat = np.stack(y_hat, axis=0)
        y_hat = np.transpose(y_hat, (0, 4, 1, 2, 3))
        if y_hat.shape[1] > 1:
            y_hat = np.mean(y_hat, axis=1, keepdims=True)
        return y_mmh[:, 0:1, ...], y_hat.astype(np.float32, copy=False)
