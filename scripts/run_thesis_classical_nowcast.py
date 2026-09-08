#!/usr/bin/env python3
"""Run S-PROG or persistence on a dumped test set (no torch; venv_pysteps).

Reads context z-score from HDF5 dump, converts to mm/h, writes predictions at
thesis lead indices to a shard HDF5 (mm/h float16).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from nowcasting.benchmarks.persistence import run_persistence_mmh, to_mmh_numpy  # noqa: E402
from nowcasting.benchmarks.pysteps import run_sprog_mmh  # noqa: E402

MEAN_LOG = 0.03019706713265408
STD_LOG = 0.5392297631902654


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--method", choices=("sprog", "persistence"), required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--nshards", type=int, default=1)
    args = parser.parse_args()

    with h5py.File(args.dump, "r") as g:
        n = int(g.attrs["n_samples"])
        lead_idx = [int(i) for i in np.asarray(g.attrs["lead_idx"]).tolist()]
        mean = float(g.attrs.get("mean_log", MEAN_LOG))
        std = float(g.attrs.get("std_log", STD_LOG))
        forecast_len = int(g.attrs.get("forecast_len", 18))
        ctx = g["context"]
        h, w = int(ctx.shape[-2]), int(ctx.shape[-1])

        per = (n + args.nshards - 1) // args.nshards
        i0 = args.shard * per
        i1 = min(n, (args.shard + 1) * per)
        if i0 >= i1:
            print(json.dumps({"shard": args.shard, "empty": True}))
            return

        n_leads = len(lead_idx)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        if args.out.exists():
            args.out.unlink()

        with h5py.File(args.out, "w") as out:
            pred = out.create_dataset(
                "pred",
                shape=(i1 - i0, n_leads, h, w),
                chunks=(8, n_leads, h, w),
                dtype="float16",
                compression="gzip",
                compression_opts=1,
            )
            out.attrs["method"] = args.method
            out.attrs["shard"] = args.shard
            out.attrs["nshards"] = args.nshards
            out.attrs["i0"] = i0
            out.attrs["i1"] = i1
            out.attrs["lead_idx"] = np.asarray(lead_idx, dtype=np.int32)
            out.attrs["space"] = "mmh"

            for local_i, global_i in enumerate(
                tqdm(range(i0, i1), desc=f"{args.method} shard{args.shard}")
            ):
                c_z = np.asarray(ctx[global_i], dtype=np.float32)
                c_mmh = to_mmh_numpy(c_z, mean=mean, std=std)
                if args.method == "sprog":
                    full = run_sprog_mmh(c_mmh, n_leadtimes=forecast_len)
                else:
                    full = run_persistence_mmh(c_mmh, n_leadtimes=forecast_len)
                pred[local_i] = full[lead_idx].astype(np.float16)

    print(
        json.dumps(
            {
                "method": args.method,
                "shard": args.shard,
                "i0": i0,
                "i1": i1,
                "out": str(args.out),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
