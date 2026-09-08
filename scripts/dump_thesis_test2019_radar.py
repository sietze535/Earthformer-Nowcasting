#!/usr/bin/env python3
"""Dump 2019 rainy test radar patches for classical baselines (z-score → HDF5).

Uses the same val dataloader as EarthFormer thesis evals (test_2019.npz).
Streams into a resizable HDF5 to avoid holding all samples in RAM.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from nowcasting.scripts.eval import setup_dataloader  # noqa: E402

LEADTIMES_MIN = [5, 10, 15, 30, 60, 90]
LEAD_IDX = [m // 5 - 1 for m in LEADTIMES_MIN]
MEAN_LOG = 0.03019706713265408
STD_LOG = 0.5392297631902654


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT
        / "config/earthformer/thesis_scaling_v1_eval2019/sprog_baseline.yaml",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "results/evaluations/_thesis_baseline_dump.h5",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    loader = setup_dataloader(**cfg.dataloader)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        args.out.unlink()

    n = 0
    with h5py.File(args.out, "w") as f:
        ctx_ds = tgt_ds = None
        for batch in tqdm(loader, desc="dump test_2019"):
            if len(batch) == 3:
                x, y, _ = batch
            else:
                x, y = batch
            x_np = x[:, 0].detach().cpu().numpy()
            y_np = y[:, 0].detach().cpu().numpy()
            ctx = x_np[:, -4:].astype(np.float16)
            tgt = y_np[:, LEAD_IDX].astype(np.float16)
            b, _, h, w = ctx.shape
            if ctx_ds is None:
                ctx_ds = f.create_dataset(
                    "context",
                    shape=(0, 4, h, w),
                    maxshape=(None, 4, h, w),
                    chunks=(8, 4, h, w),
                    dtype="float16",
                    compression="gzip",
                    compression_opts=1,
                )
                tgt_ds = f.create_dataset(
                    "target",
                    shape=(0, len(LEAD_IDX), h, w),
                    maxshape=(None, len(LEAD_IDX), h, w),
                    chunks=(8, len(LEAD_IDX), h, w),
                    dtype="float16",
                    compression="gzip",
                    compression_opts=1,
                )
            ctx_ds.resize(n + b, axis=0)
            tgt_ds.resize(n + b, axis=0)
            ctx_ds[n : n + b] = ctx
            tgt_ds[n : n + b] = tgt
            n += b

        f.attrs["n_samples"] = n
        f.attrs["lead_idx"] = np.asarray(LEAD_IDX, dtype=np.int32)
        f.attrs["leadtimes_min"] = np.asarray(LEADTIMES_MIN, dtype=np.int32)
        f.attrs["config_path"] = str(args.config)
        f.attrs["space"] = "zscore_log10_mmh"
        f.attrs["mean_log"] = MEAN_LOG
        f.attrs["std_log"] = STD_LOG
        f.attrs["forecast_len"] = 18

    print(
        json.dumps(
            {
                "n_samples": n,
                "lead_idx": LEAD_IDX,
                "out": str(args.out),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
