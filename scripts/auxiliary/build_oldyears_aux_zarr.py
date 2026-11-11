#!/usr/bin/env python3
"""
Build old-years air temperature and dewpoint channels for Earthformer training.

Pipeline overview
-----------------
1. Open the existing old-years radar zarr **read-only** and rebuild its 5-min timeline
   from ``radar`` group attrs (``start_time``, ``end_time``, ``interval_minutes``).
2. Read hourly KNMI station NetCDF files (``T`` / ``TD``) and interpolate them onto the
   radar grid with inverse-distance weighting (IDW), matching pkools' formula.
3. Forward-fill each hourly field onto consecutive 5-min radar steps for up to
   ``--max-fill-minutes`` (default 60 → the observation hour plus the next 12 steps).
4. Write temperature / dewpoint into a **new** versioned output zarr under ``harmonie/``,
   using HARMONIE-compatible array names so downstream loaders can treat old-years
   aux the same way as recent-years HARMONIE aux.

Source zarr is never modified; output is opened in append mode so a partial run can
resume if arrays already exist.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import xarray as xr
import zarr


# Filename pattern: hourly-observations-validated-YYYYMMDD-HH.nc
TS_RE = re.compile(r"hourly-observations-validated-(\d{8})-(\d{2})\.nc$")


@dataclass(frozen=True)
class GridSpec:
    """Radar / aux grid extent. Must match the old-years radar domain exactly."""

    nx: int = 704
    ny: int = 768
    lon_min: float = 0.0
    lon_max: float = 10.8564
    lat_min: float = 48.8953
    lat_max: float = 55.9736


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build old-years aux zarr (temp + dewpoint)")
    p.add_argument("--source-zarr", required=True, help="Read-only source old-years zarr")
    p.add_argument("--output-zarr", required=True, help="New output zarr path to create/update")
    p.add_argument("--hourly-dir", required=True, help="Root directory with hourly validated .nc files")
    p.add_argument("--report-dir", required=True, help="Directory for manifests/reports")
    p.add_argument(
        "--min-stations",
        type=int,
        default=8,
        help="Skip an hour if fewer than this many stations have finite T or TD",
    )
    p.add_argument(
        "--idw-power",
        type=float,
        default=3.0,
        help="IDW distance exponent (pkools default is 3.0)",
    )
    p.add_argument(
        "--max-fill-minutes",
        type=int,
        default=60,
        help="How far past each hourly stamp to reuse the same grid (5-min steps)",
    )
    p.add_argument("--limit-files", type=int, default=0, help="Debug: process only first N files")
    return p.parse_args()


def _list_hourly_files(hourly_dir: Path) -> List[Path]:
    # Expected layout: hourly_dir/<subdir>/*.nc
    files = sorted(hourly_dir.glob("*/*.nc"))
    return files


def _timestamp_from_filename(path: Path) -> pd.Timestamp | None:
    m = TS_RE.search(path.name)
    if not m:
        return None
    day, hour = m.group(1), m.group(2)
    # Files are tagged by hour, with timestamp semantics already in file metadata.
    return pd.Timestamp(f"{day} {hour}:00:00")


def _build_timeline(radar_group: zarr.Group) -> pd.DatetimeIndex:
    """Rebuild the full 5-min index from radar attrs so aux aligns with radar time."""
    start = pd.Timestamp(radar_group.attrs["start_time"])
    end = pd.Timestamp(radar_group.attrs["end_time"])
    step = int(radar_group.attrs["interval_minutes"])
    return pd.date_range(start=start, end=end, freq=f"{step}min")


def _pkools_idw_weights(
    station_lons: np.ndarray,
    station_lats: np.ndarray,
    grid_lons_flat: np.ndarray,
    grid_lats_flat: np.ndarray,
    power: float,
) -> np.ndarray:
    """
    Inverse-distance weights of shape (n_stations, n_grid_cells).

    Matches pkools' ``precompute_idw_weights``:
      dist = hypot(dx, dy)
      w    = 1 / max(dist, 1e-12)^power
      then normalize so each grid cell sums to 1 over stations.
    """
    dx = station_lons[:, None] - grid_lons_flat[None, :]
    dy = station_lats[:, None] - grid_lats_flat[None, :]
    dist = np.hypot(dx, dy)
    w = 1.0 / np.maximum(dist, 1e-12) ** power
    w /= w.sum(axis=0, keepdims=True)
    return w.astype(np.float32, copy=False)


def _grid_lon_lat(spec: GridSpec) -> Tuple[np.ndarray, np.ndarray]:
    """
    Flattened lon/lat of every grid cell.

    Latitude runs north→south (lat_max → lat_min) so row 0 matches the radar
    convention used elsewhere in this repo.
    """
    lon_grid, lat_grid = np.meshgrid(
        np.linspace(spec.lon_min, spec.lon_max, spec.nx, dtype=np.float32),
        np.linspace(spec.lat_max, spec.lat_min, spec.ny, dtype=np.float32),
    )
    return lon_grid.ravel(), lat_grid.ravel()


def _read_hourly_file(path: Path) -> Dict[str, np.ndarray]:
    """Load station lon/lat plus temperature (T) and dewpoint (TD) from one NetCDF."""
    ds = xr.open_dataset(path)
    try:
        lons = ds["lon"].to_numpy().astype(np.float32)
        lats = ds["lat"].to_numpy().astype(np.float32)
        t = ds["T"].to_numpy().astype(np.float32).reshape(-1)
        td = ds["TD"].to_numpy().astype(np.float32).reshape(-1)
    finally:
        ds.close()
    return {"lon": lons, "lat": lats, "t": t, "td": td}


def _to_5min_indices(ts_hourly: pd.Timestamp, timeline: pd.DatetimeIndex, max_fill_minutes: int) -> np.ndarray:
    """
    Map one hourly observation onto the radar timeline.

    Returns contiguous indices ``[base, base+1, …]`` covering the observation
    timestamp itself plus forward-fill steps of 5 minutes up to ``max_fill_minutes``.
    Empty array if the hourly stamp is outside the radar timeline.
    """
    if ts_hourly not in timeline:
        return np.array([], dtype=np.int64)
    base_idx = timeline.get_loc(ts_hourly)
    max_steps = max_fill_minutes // 5
    return np.arange(base_idx, min(base_idx + max_steps + 1, len(timeline)), dtype=np.int64)


def _ensure_output_groups(out_root: zarr.Group, src_radar: zarr.Group, timeline_len: int, spec: GridSpec) -> zarr.Group:
    """
    Create (or reopen) the output zarr layout.

    - ``radar/``: thin metadata + ``time_mask`` copy so split / alignment code that
      expects a radar group still works on the aux-only store.
    - ``harmonie/``: aux fields named like HARMONIE products:
        TMP_GDS0_HTGL / DPT_GDS0_HTGL          → (time, ny, nx) float16 grids
        *_age_minutes                          → minutes since last real obs (0, 5, …)
        station_count                          → stations used for that hour
        time_mask                              → True where aux was written
    """
    if "radar" not in out_root:
        # Keep a minimal radar group metadata mirror for downstream split logic.
        radar_out = out_root.create_group("radar")
        for k, v in dict(src_radar.attrs).items():
            radar_out.attrs[k] = v
        radar_out.create_array("time_mask", shape=(timeline_len,), chunks=(timeline_len,), dtype=bool)
        radar_out["time_mask"][:] = src_radar["time_mask"][:]
    harmonie = out_root.require_group("harmonie")
    harmonie.attrs["interval_minutes"] = 5
    if "time_mask" not in harmonie:
        harmonie.create_array(
            "time_mask",
            shape=(timeline_len,),
            chunks=(timeline_len,),
            dtype=bool,
            fill_value=False,
        )
    # Chunk along time only so each write touches one (or a few) full spatial frames.
    arr_shape = (timeline_len, spec.ny, spec.nx)
    arr_chunks = (1, spec.ny, spec.nx)
    for name in ("TMP_GDS0_HTGL", "DPT_GDS0_HTGL"):
        if name not in harmonie:
            # Avoid eager multi-terabyte prefill; rely on fill_value for untouched chunks.
            harmonie.create_array(
                name,
                shape=arr_shape,
                chunks=arr_chunks,
                dtype=np.float16,
                fill_value=np.float16(0.0),
            )
    for name in ("TMP_GDS0_HTGL_age_minutes", "DPT_GDS0_HTGL_age_minutes"):
        if name not in harmonie:
            # 65535 = sentinel for "never written" (uint16 max).
            harmonie.create_array(
                name,
                shape=(timeline_len,),
                chunks=(timeline_len,),
                dtype=np.uint16,
                fill_value=np.uint16(65535),
            )
    if "station_count" not in harmonie:
        harmonie.create_array(
            "station_count",
            shape=(timeline_len,),
            chunks=(timeline_len,),
            dtype=np.uint16,
            fill_value=np.uint16(0),
        )
    return harmonie


def _write_manifest(source_zarr: Path, hourly_dir: Path, output_zarr: Path, report_dir: Path, files: List[Path]) -> None:
    """Record input paths / file range for reproducibility of this build run."""
    report_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source_zarr": str(source_zarr),
        "hourly_dir": str(hourly_dir),
        "output_zarr": str(output_zarr),
        "hourly_file_count": len(files),
        "hourly_first": str(files[0]) if files else None,
        "hourly_last": str(files[-1]) if files else None,
        "created_at_utc": pd.Timestamp.now("UTC").isoformat(),
    }
    (report_dir / "source_manifest.json").write_text(json.dumps(manifest, indent=2))


def _validate_station_domain(st_lons: np.ndarray, st_lats: np.ndarray, spec: GridSpec) -> Dict[str, float]:
    """Count how many stations fall inside the radar lon/lat box (sanity check)."""
    inside = (
        (st_lons >= spec.lon_min)
        & (st_lons <= spec.lon_max)
        & (st_lats >= spec.lat_min)
        & (st_lats <= spec.lat_max)
    )
    return {
        "stations_total": int(st_lons.shape[0]),
        "stations_inside": int(inside.sum()),
        "stations_outside": int((~inside).sum()),
    }


def main() -> None:
    args = parse_args()
    source_zarr = Path(args.source_zarr)
    output_zarr = Path(args.output_zarr)
    hourly_dir = Path(args.hourly_dir)
    report_dir = Path(args.report_dir)
    spec = GridSpec()

    files = _list_hourly_files(hourly_dir)
    if args.limit_files > 0:
        files = files[: args.limit_files]
    if not files:
        raise RuntimeError(f"No hourly files found in {hourly_dir}")

    _write_manifest(source_zarr, hourly_dir, output_zarr, report_dir, files)

    # Timeline comes from radar attrs; aux is written into a separate output store.
    src_root = zarr.open_group(str(source_zarr), mode="r")
    src_radar = src_root["radar"]
    timeline = _build_timeline(src_radar)
    out_root = zarr.open_group(str(output_zarr), mode="a")
    harmonie = _ensure_output_groups(out_root, src_radar, len(timeline), spec)

    grid_lons_flat, grid_lats_flat = _grid_lon_lat(spec)

    processed_hourly = 0
    written_5min = 0
    parity_stats: List[float] = []
    station_domain_summary: Dict[str, float] | None = None

    for fp in files:
        payload = _read_hourly_file(fp)
        st_lons = payload["lon"]
        st_lats = payload["lat"]
        t_vals = payload["t"]
        td_vals = payload["td"]

        # Keep a station if either T or TD is finite; missing channel is filled below.
        valid_temp = np.isfinite(t_vals)
        valid_dew = np.isfinite(td_vals)
        valid_both = valid_temp | valid_dew
        if valid_both.sum() < args.min_stations:
            continue

        # Domain check once (first accepted hour) — enough for a build-time sanity report.
        if station_domain_summary is None:
            station_domain_summary = _validate_station_domain(st_lons[valid_both], st_lats[valid_both], spec)

        w = _pkools_idw_weights(st_lons[valid_both], st_lats[valid_both], grid_lons_flat, grid_lats_flat, args.idw_power)
        # pkools parity check on one frame: same formula done twice should be numerically near-identical.
        w_ref = _pkools_idw_weights(st_lons[valid_both], st_lats[valid_both], grid_lons_flat, grid_lats_flat, args.idw_power)
        parity_stats.append(float(np.max(np.abs(w - w_ref))))

        # Stations with NaN on one channel get the median of the valid stations for that channel.
        t_valid = np.where(np.isfinite(t_vals[valid_both]), t_vals[valid_both], np.nanmedian(t_vals[valid_both]))
        td_valid = np.where(np.isfinite(td_vals[valid_both]), td_vals[valid_both], np.nanmedian(td_vals[valid_both]))
        # (n_stations,) @ (n_stations, n_cells) → flat grid, then reshape to (ny, nx).
        t_grid = (t_valid @ w).reshape(spec.ny, spec.nx).astype(np.float16)
        td_grid = (td_valid @ w).reshape(spec.ny, spec.nx).astype(np.float16)

        ts = _timestamp_from_filename(fp)
        if ts is None:
            continue
        indices = _to_5min_indices(ts, timeline, args.max_fill_minutes)
        if indices.size == 0:
            continue

        # Zarr v3: assigning a 2D (ny, nx) RHS to arr[integer_array, :, :] is not supported the same
        # as NumPy; the write path expects a (n_time, ny, nx) block. Indices here are always a
        # contiguous np.arange, so use a time slice and broadcast the grid to (n, ny, nx).
        i0 = int(indices[0])
        i1 = int(indices[-1]) + 1
        n = i1 - i0
        if n != int(indices.size):
            raise RuntimeError("expected contiguous 5-min indices from hourly timestamp")
        block_t = np.broadcast_to(t_grid[np.newaxis, :, :], (n, spec.ny, spec.nx)).copy()
        block_td = np.broadcast_to(td_grid[np.newaxis, :, :], (n, spec.ny, spec.nx)).copy()
        harmonie["TMP_GDS0_HTGL"][i0:i1, :, :] = block_t
        harmonie["DPT_GDS0_HTGL"][i0:i1, :, :] = block_td
        # age=0 at the observation hour, then 5, 10, … for forward-filled steps.
        harmonie["TMP_GDS0_HTGL_age_minutes"][i0:i1] = np.arange(0, 5 * n, 5, dtype=np.uint16)
        harmonie["DPT_GDS0_HTGL_age_minutes"][i0:i1] = np.arange(0, 5 * n, 5, dtype=np.uint16)
        harmonie["station_count"][i0:i1] = np.uint16(valid_both.sum())
        harmonie["time_mask"][i0:i1] = True

        processed_hourly += 1
        written_5min += int(indices.size)
        if processed_hourly % 200 == 0:
            print(f"[progress] hourly={processed_hourly} written_5min={written_5min}", flush=True)

    if station_domain_summary is None:
        station_domain_summary = {"stations_total": 0, "stations_inside": 0, "stations_outside": 0}

    summary = {
        "processed_hourly_files": processed_hourly,
        "written_5min_steps": written_5min,
        "time_mask_true": int(harmonie["time_mask"][:].sum()),
        "time_mask_fraction": float(harmonie["time_mask"][:].mean()),
        "pkools_parity_max_absdiff_weights": float(max(parity_stats) if parity_stats else math.nan),
        "pkools_parity_mean_absdiff_weights": float(np.mean(parity_stats) if parity_stats else math.nan),
        "station_domain": station_domain_summary,
        "max_fill_minutes": args.max_fill_minutes,
        "min_stations": args.min_stations,
        "idw_power": args.idw_power,
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "build_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
