#!/usr/bin/env python3
"""Generate deterministic nested index files for thesis scaling experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zarr
from omegaconf import OmegaConf

from nowcasting.data.utils.generate_samples import get_sample_dfs


def _create_mask(root: zarr.Group, split_info: dict, num_timesteps: int) -> np.ndarray:
    global_mask = np.ones(num_timesteps, dtype=bool)
    for group_name in split_info.get("apply_missing_masks", []):
        if group_name not in root or "time_mask" not in root[group_name]:
            continue
        factor = root[group_name].attrs["interval_minutes"] // 5
        group_mask = root[group_name]["time_mask"][:]
        global_mask &= np.repeat(group_mask, factor)

    clutter_threshold = split_info.get("clutter_threshold")
    if (
        clutter_threshold is not None
        and "radar" in root
        and "clutter_score" in root["radar"]
    ):
        global_mask &= root["radar"]["clutter_score"][:] < clutter_threshold
    return global_mask


def _apply_threshold(df: pd.DataFrame, threshold: float | None) -> pd.DataFrame:
    if threshold is None:
        return df.copy()
    threshold_raw = threshold * (255 / 50)
    return df[df["weight"] > threshold_raw].reset_index(drop=True)


def _split_with_temporal_rules(
    sample_array: np.ndarray,
    timestamps: pd.DatetimeIndex,
    kernel: tuple[int, int, int],
    stride: tuple[int, int, int],
    methods: dict,
    time_mask: np.ndarray,
) -> dict[str, pd.DataFrame]:
    split_rules = {
        "test": {"year": [2019]},
        "val": {"year": [2018]},
        "train": {"year": list(range(2008, 2018))},
    }
    return get_sample_dfs(
        sample_array=sample_array,
        kernel=kernel,
        stride=stride,
        timestamps=timestamps,
        split_rules=split_rules,
        methods=methods,
        time_mask=time_mask,
    )


def _assign_bins(weight_values: np.ndarray) -> np.ndarray:
    """Map sample weights to 4 bins based on mm/h thresholds."""
    mmh = weight_values.astype(np.float64) / 5.1
    bins = np.zeros_like(mmh, dtype=np.uint8)
    bins[(mmh >= 0.5) & (mmh < 2.0)] = 1
    bins[(mmh >= 2.0) & (mmh < 10.0)] = 2
    bins[mmh >= 10.0] = 3
    return bins


def _save_index_npz(
    out_path: Path,
    df: pd.DataFrame,
    timestamps: pd.DatetimeIndex,
    bin_ids: np.ndarray | None,
    split: str,
    fraction: str,
) -> None:
    t_idx = df["t_idx"].to_numpy(dtype=np.uint32, copy=False)
    h_idx = df["h_idx"].to_numpy(dtype=np.uint8, copy=False)
    w_idx = df["w_idx"].to_numpy(dtype=np.uint8, copy=False)
    ts_ns = timestamps[t_idx].view("int64")
    payload = {
        "t_idx": t_idx,
        "h_idx": h_idx,
        "w_idx": w_idx,
        "timestamp_ns": ts_ns.astype(np.int64, copy=False),
    }
    if bin_ids is not None:
        payload["bin_id"] = bin_ids.astype(np.uint8, copy=False)
    np.savez_compressed(out_path, **payload)
    print(f"[saved] {split} {fraction}: {out_path} ({len(df)} rows)")


def _plot_train_histograms(
    out_path: Path,
    mmh_100: np.ndarray,
    mmh_5: np.ndarray,
    bins: np.ndarray,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(
        mmh_100,
        bins=bins,
        density=True,
        alpha=0.45,
        label="S100 train",
        color="tab:blue",
    )
    ax.hist(
        mmh_5,
        bins=bins,
        density=True,
        alpha=0.45,
        label="S5 train",
        color="tab:orange",
    )
    ax.set_xlabel("Max intensity (mm/h)")
    ax.set_ylabel("Density")
    ax.set_title("Train intensity histogram overlap (S5 vs S100)")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"[saved] histogram: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="config/earthformer/thesis_scaling_v1/radar_aux__100pct.yaml",
        help="Reference config used to read dataloader parameters.",
    )
    parser.add_argument(
        "--out_dir",
        default="indices",
        help="Directory for index artifacts.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    data_cfg = cfg.dataloader
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    root = zarr.open_group(data_cfg.paths[0], mode="r")
    sample_array = root[data_cfg.var_info.sample_var][:]

    start_time = root["radar"].attrs["start_time"]
    end_time = root["radar"].attrs["end_time"]
    timestamps = pd.date_range(start_time, end_time, freq="5min")

    kernel = (
        int(data_cfg.context_len + data_cfg.forecast_len),
        int(data_cfg.img_size[0]),
        int(data_cfg.img_size[1]),
    )
    stride = tuple(int(x) for x in data_cfg.stride)
    methods = OmegaConf.to_container(data_cfg.sample_info.methods, resolve=True)
    split_info = OmegaConf.to_container(data_cfg.split_info, resolve=True)
    threshold = (
        float(data_cfg.sample_info.threshold)
        if "threshold" in data_cfg.sample_info
        else None
    )

    time_mask = _create_mask(root, split_info=split_info, num_timesteps=len(sample_array))
    dfs = _split_with_temporal_rules(
        sample_array=sample_array,
        timestamps=timestamps,
        kernel=kernel,
        stride=stride,
        methods=methods,
        time_mask=time_mask,
    )
    train_df = _apply_threshold(dfs["train"], threshold=threshold)
    val_df = _apply_threshold(dfs["val"], threshold=threshold)
    test_df = _apply_threshold(dfs["test"], threshold=threshold)
    print(
        f"[counts] train={len(train_df)} val={len(val_df)} test={len(test_df)} "
        "(after threshold)"
    )

    rng = np.random.default_rng(args.seed)
    bin_id_all = _assign_bins(train_df["weight"].to_numpy())

    # Build deterministic shuffled order within each bin.
    bin_indices = {}
    for bid in range(4):
        idx = np.flatnonzero(bin_id_all == bid)
        rng.shuffle(idx)
        bin_indices[bid] = idx

    fractions = [5, 10, 25, 50, 100]
    summary = {
        "seed": args.seed,
        "split_years": {"train": "2008-2017", "val": "2018", "test": "2019"},
        "counts_after_threshold": {
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "train_subsets": {},
    }

    for frac in fractions:
        selected = []
        for bid in range(4):
            ordered = bin_indices[bid]
            n_take = int(np.floor(len(ordered) * (frac / 100.0)))
            if frac == 100:
                n_take = len(ordered)
            selected.append(ordered[:n_take])
        sel_idx = np.concatenate(selected) if selected else np.array([], dtype=np.int64)
        # Keep deterministic sample order for reproducibility.
        sel_idx = np.sort(sel_idx)
        subset = train_df.iloc[sel_idx].reset_index(drop=True)
        subset_bins = bin_id_all[sel_idx]
        _save_index_npz(
            out_path=out_dir / f"train_{frac:03d}pct.npz",
            df=subset,
            timestamps=timestamps,
            bin_ids=subset_bins,
            split="train",
            fraction=f"{frac}%",
        )
        summary["train_subsets"][f"{frac:03d}pct"] = {
            "n": int(len(subset)),
            "bin_counts": {
                str(b): int((subset_bins == b).sum()) for b in range(4)
            },
        }

    _save_index_npz(
        out_path=out_dir / "val_2018.npz",
        df=val_df,
        timestamps=timestamps,
        bin_ids=None,
        split="val",
        fraction="2018",
    )
    _save_index_npz(
        out_path=out_dir / "test_2019.npz",
        df=test_df,
        timestamps=timestamps,
        bin_ids=None,
        split="test",
        fraction="2019",
    )

    mmh_100 = train_df["weight"].to_numpy(dtype=np.float64) / 5.1
    subset_5 = np.load(out_dir / "train_005pct.npz")
    mmh_5 = train_df.merge(
        pd.DataFrame(
            {
                "t_idx": subset_5["t_idx"],
                "h_idx": subset_5["h_idx"],
                "w_idx": subset_5["w_idx"],
            }
        ),
        on=["t_idx", "h_idx", "w_idx"],
        how="inner",
    )["weight"].to_numpy(dtype=np.float64) / 5.1
    subset_5.close()

    histogram_bins = np.array([0.0, 0.5, 2.0, 10.0, 20.0, 40.0, 80.0, 160.0], dtype=float)
    _plot_train_histograms(
        out_path=out_dir / "hist_train_005_vs_100.png",
        mmh_100=mmh_100,
        mmh_5=mmh_5,
        bins=histogram_bins,
    )

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[saved] summary: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
