#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import zarr


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute joint normalization stats for TMP+DPT channels.")
    p.add_argument("--zarr-path", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--sample-step", type=int, default=12, help="Temporal stride for stats estimate (default: hourly from 5-min data).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = zarr.open_group(args.zarr_path, mode="r")
    grp = root["harmonie"]
    t = grp["TMP_GDS0_HTGL"]
    td = grp["DPT_GDS0_HTGL"]
    mask = grp["time_mask"][:]
    idx = np.where(mask)[0][:: max(args.sample_step, 1)]

    if idx.size == 0:
        raise RuntimeError("No valid timesteps found in harmonie/time_mask")

    # Streaming stats to avoid huge temporary allocations.
    n = 0
    mean = 0.0
    m2 = 0.0

    def update_stats(arr: np.ndarray) -> tuple[int, float, float]:
        nonlocal n, mean, m2
        flat = arr.astype(np.float64, copy=False).ravel()
        # Batch update using aggregate moments.
        k = flat.size
        if k == 0:
            return n, mean, m2
        b_mean = float(flat.mean())
        b_m2 = float(((flat - b_mean) ** 2).sum())
        if n == 0:
            n = k
            mean = b_mean
            m2 = b_m2
            return n, mean, m2
        delta = b_mean - mean
        new_n = n + k
        mean = mean + delta * (k / new_n)
        m2 = m2 + b_m2 + (delta * delta) * (n * k / new_n)
        n = new_n
        return n, mean, m2

    chunk = 64
    for s in range(0, idx.size, chunk):
        e = min(s + chunk, idx.size)
        sub_idx = idx[s:e]
        update_stats(t[sub_idx, :, :])
        update_stats(td[sub_idx, :, :])

    if n < 2:
        raise RuntimeError("Insufficient samples for normalization stats")
    mu = float(mean)
    sigma = float(np.sqrt(m2 / n))

    payload = {
        "joint_temp_dew_mean": mu,
        "joint_temp_dew_std": sigma,
        "n_timesteps_used": int(idx.size),
        "sample_step": int(args.sample_step),
    }
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
