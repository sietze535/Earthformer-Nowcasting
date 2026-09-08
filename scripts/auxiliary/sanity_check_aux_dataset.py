#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

import numpy as np
import zarr


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Read-only sanity checks for aux-augmented zarr.")
    p.add_argument("--zarr-path", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--sample-steps", type=int, default=256)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = zarr.open_group(args.zarr_path, mode="r")
    radar = root["radar"]
    harmonie = root["harmonie"]

    n = radar["rtcor"].shape[0]
    idx = np.linspace(0, n - 1, num=min(args.sample_steps, n), dtype=int)

    t = harmonie["TMP_GDS0_HTGL"][idx, :, :].astype(np.float32)
    td = harmonie["DPT_GDS0_HTGL"][idx, :, :].astype(np.float32)
    m = harmonie["time_mask"][idx]
    station_count = harmonie["station_count"][idx]
    age_t = harmonie["TMP_GDS0_HTGL_age_minutes"][idx]
    age_td = harmonie["DPT_GDS0_HTGL_age_minutes"][idx]

    result = {
        "n_total_steps": int(n),
        "n_sampled_steps": int(idx.size),
        "harmonie_time_mask_fraction_sampled": float(m.mean()),
        "tmp_nan_count": int(np.isnan(t).sum()),
        "dpt_nan_count": int(np.isnan(td).sum()),
        "tmp_min": float(np.nanmin(t)),
        "tmp_max": float(np.nanmax(t)),
        "dpt_min": float(np.nanmin(td)),
        "dpt_max": float(np.nanmax(td)),
        "station_count_min": int(station_count.min()),
        "station_count_max": int(station_count.max()),
        "age_tmp_max": int(age_t.max()),
        "age_dpt_max": int(age_td.max()),
    }

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
