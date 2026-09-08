#!/usr/bin/env python3
"""Score classical baseline predictions with PrecipitationEvaluator (.venv + torch)."""

from __future__ import annotations

import argparse
import json
import sys
from glob import glob
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from nowcasting.benchmarks.persistence import to_mmh_numpy  # noqa: E402
from nowcasting.validation.validation import PrecipitationEvaluator  # noqa: E402

MEAN_LOG = 0.03019706713265408
STD_LOG = 0.5392297631902654
LEADTIMES = [5, 10, 15, 30, 60, 90]
THRESHOLDS = [0, 0.5, 1, 2, 5, 10, 20, 30]
SCALES = [1, 2, 4, 8, 16, 32]


def _expand_leads(arr6: np.ndarray, lead_idx: list[int], forecast_len: int = 18) -> np.ndarray:
    """(B, n_leads, H, W) → (B, 1, forecast_len, H, W) placed at lead_idx."""
    b, _, h, w = arr6.shape
    out = np.zeros((b, 1, forecast_len, h, w), dtype=np.float32)
    for j, ti in enumerate(lead_idx):
        out[:, 0, ti] = arr6[:, j]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument(
        "--pred-glob",
        type=str,
        required=True,
        help="Glob of shard HDF5 files, e.g. results/evaluations/_preds_sprog/shard_*.h5",
    )
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--save-dir", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--eval-subdir", type=str, default="evaluations")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    with h5py.File(args.dump, "r") as dump:
        n = int(dump.attrs["n_samples"])
        lead_idx = [int(i) for i in np.asarray(dump.attrs["lead_idx"]).tolist()]
        mean = float(dump.attrs.get("mean_log", MEAN_LOG))
        std = float(dump.attrs.get("std_log", STD_LOG))
        forecast_len = int(dump.attrs.get("forecast_len", 18))
        tgt = dump["target"]

        shard_paths = [Path(p) for p in sorted(glob(args.pred_glob))]
        if not shard_paths:
            raise FileNotFoundError(f"No pred shards matched: {args.pred_glob}")

        shards_meta = []
        for p in shard_paths:
            with h5py.File(p, "r") as g:
                shards_meta.append((int(g.attrs["i0"]), int(g.attrs["i1"]), p))
        shards_meta.sort(key=lambda t: t[0])
        if shards_meta[0][0] != 0 or shards_meta[-1][1] != n:
            raise RuntimeError(
                f"Shard coverage incomplete: {shards_meta[0]} .. {shards_meta[-1]} vs n={n}"
            )

        # Keep shard files open for reading
        opened = []
        for i0, i1, p in shards_meta:
            fh = h5py.File(p, "r")
            opened.append((i0, i1, fh, fh["pred"]))

        def pred_slice(a: int, b: int) -> np.ndarray:
            parts = []
            for s0, s1, _fh, arr in opened:
                lo = max(a, s0)
                hi = min(b, s1)
                if lo < hi:
                    parts.append(np.asarray(arr[lo - s0 : hi - s0], dtype=np.float32))
            return np.concatenate(parts, axis=0)

        evaluator = PrecipitationEvaluator(
            nowcast_method=args.model_name,
            save_dir=str(args.save_dir),
            eval_subdir=args.eval_subdir,
            thresholds=THRESHOLDS,
            leadtimes=LEADTIMES,
            scales=SCALES,
            crop_size=None,
            save_frequency=10_000,
            compute_crps=False,
            compute_fss=True,
            compute_dry_to_wet=False,
        )

        try:
            for start in tqdm(range(0, n, args.batch_size), desc=f"score {args.model_name}"):
                end = min(n, start + args.batch_size)
                tgt_z = np.asarray(tgt[start:end], dtype=np.float32)
                tgt_mmh = to_mmh_numpy(tgt_z, mean=mean, std=std)
                pred_mmh = pred_slice(start, end)
                y = _expand_leads(tgt_mmh, lead_idx, forecast_len)
                y_hat = _expand_leads(pred_mmh, lead_idx, forecast_len)
                evaluator.process_batch(y, y_hat)
        finally:
            for _i0, _i1, fh, _arr in opened:
                fh.close()

        evaluator.metadata["n_samples"] = n
        evaluator.save_results("final_results")
        scores = evaluator.get_final_scores()
        print(
            json.dumps(
                {
                    "model_name": args.model_name,
                    "n_samples": n,
                    "n_shards": len(shards_meta),
                    "continuous": len(scores.get("continuous", [])),
                    "categorical": len(scores.get("categorical", [])),
                    "fss": len(scores.get("fss", [])),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
