import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import zarr


DEFAULT_SOURCES = [
    # Set via --sources. Example: $THESIS_DATA_ROOT/radar_{year}.zarr for 2008-2019.
]


@dataclass
class SourceInfo:
    path: Path
    year: int
    n_time: int
    rtcor_shape: tuple[int, int, int]
    rtcor_chunks: tuple[int, int, int]
    rtcor_dtype: np.dtype
    mask_shape: tuple[int]
    mask_dtype: np.dtype
    max_shape: tuple[int, int, int]
    max_dtype: np.dtype
    attrs: dict[str, Any]


def parse_year(path: Path) -> int:
    name = path.name
    token = name.replace("radar_", "").replace(".zarr", "")
    return int(token)


def inspect_source(path: Path) -> SourceInfo:
    root = zarr.open_group(str(path), mode="r")
    radar = root["radar"]
    rtcor = radar["rtcor"]
    time_mask = radar["time_mask"]
    max_grid = radar["max_intensity_grid"]
    return SourceInfo(
        path=path,
        year=parse_year(path),
        n_time=int(rtcor.shape[0]),
        rtcor_shape=tuple(rtcor.shape),
        rtcor_chunks=tuple(rtcor.chunks),
        rtcor_dtype=rtcor.dtype,
        mask_shape=tuple(time_mask.shape),
        mask_dtype=time_mask.dtype,
        max_shape=tuple(max_grid.shape),
        max_dtype=max_grid.dtype,
        attrs=dict(radar.attrs),
    )


def load_state(state_path: Path) -> dict[str, Any]:
    if not state_path.exists():
        return {"completed_years": [], "writes": []}
    return json.loads(state_path.read_text())


def write_state(state_path: Path, state: dict[str, Any]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True))


def combine(args: argparse.Namespace) -> None:
    source_paths = [Path(p) for p in args.sources]
    source_infos = [inspect_source(p) for p in source_paths]
    source_infos.sort(key=lambda s: s.year)
    years = [s.year for s in source_infos]
    if years != list(range(min(years), max(years) + 1)):
        raise ValueError(f"Non-contiguous year list: {years}")

    base = source_infos[0]
    for src in source_infos[1:]:
        if src.rtcor_shape[1:] != base.rtcor_shape[1:]:
            raise ValueError(f"Spatial mismatch: {src.path} vs {base.path}")
        if src.rtcor_dtype != base.rtcor_dtype:
            raise ValueError(f"rtcor dtype mismatch: {src.path} vs {base.path}")
        if src.max_shape[1:] != base.max_shape[1:]:
            raise ValueError(f"max_intensity_grid mismatch: {src.path} vs {base.path}")
        if src.max_dtype != base.max_dtype:
            raise ValueError(f"max grid dtype mismatch: {src.path} vs {base.path}")
        if src.mask_dtype != base.mask_dtype:
            raise ValueError(f"time_mask dtype mismatch: {src.path} vs {base.path}")

    total_time = int(sum(s.n_time for s in source_infos))
    out_path = Path(args.output)
    state_path = Path(args.state_path)
    root = zarr.open_group(str(out_path), mode="a")
    radar = root.require_group("radar")

    rt_shape = (total_time, base.rtcor_shape[1], base.rtcor_shape[2])
    mg_shape = (total_time, base.max_shape[1], base.max_shape[2])
    tm_shape = (total_time,)
    rt_chunks = (min(args.chunk_frames, total_time), base.rtcor_shape[1], base.rtcor_shape[2])
    mg_chunks = (min(args.chunk_frames, total_time), base.max_shape[1], base.max_shape[2])
    tm_chunks = (min(max(4096, args.chunk_frames * 16), total_time),)

    if "rtcor" not in radar:
        radar.create_array("rtcor", shape=rt_shape, chunks=rt_chunks, dtype=base.rtcor_dtype)
    if "time_mask" not in radar:
        radar.create_array("time_mask", shape=tm_shape, chunks=tm_chunks, dtype=base.mask_dtype)
    if "max_intensity_grid" not in radar:
        radar.create_array("max_intensity_grid", shape=mg_shape, chunks=mg_chunks, dtype=base.max_dtype)

    radar_out_rt = radar["rtcor"]
    radar_out_tm = radar["time_mask"]
    radar_out_mg = radar["max_intensity_grid"]
    if tuple(radar_out_rt.shape) != rt_shape or tuple(radar_out_mg.shape) != mg_shape or tuple(radar_out_tm.shape) != tm_shape:
        raise ValueError("Output arrays exist with unexpected shapes. Use a clean output path.")

    state = load_state(state_path)
    completed_years = set(int(y) for y in state.get("completed_years", []))
    writes = list(state.get("writes", []))

    offset = 0
    for src in source_infos:
        src_end = offset + src.n_time
        if src.year in completed_years:
            print(f"[skip] year={src.year} already marked complete")
            offset = src_end
            continue

        print(f"[copy] year={src.year} source={src.path} time={src.n_time} offset={offset}:{src_end}")
        src_radar = zarr.open_group(str(src.path), mode="r")["radar"]
        src_rt = src_radar["rtcor"]
        src_tm = src_radar["time_mask"]
        src_mg = src_radar["max_intensity_grid"]

        step = args.chunk_frames
        for i in range(0, src.n_time, step):
            j = min(i + step, src.n_time)
            radar_out_rt[offset + i : offset + j] = src_rt[i:j]
            radar_out_mg[offset + i : offset + j] = src_mg[i:j]
            radar_out_tm[offset + i : offset + j] = src_tm[i:j]

        completed_years.add(src.year)
        writes.append(
            {
                "year": src.year,
                "start": offset,
                "end": src_end,
                "source": str(src.path),
                "valid_time_mask": int(np.count_nonzero(src_tm[:])),
            }
        )
        state["completed_years"] = sorted(completed_years)
        state["writes"] = writes
        write_state(state_path, state)
        offset = src_end

    # Copy/refresh attrs from first and last source.
    start_time = source_infos[0].attrs.get("start_time")
    end_time = source_infos[-1].attrs.get("end_time")
    radar.attrs.update(base.attrs)
    if start_time is not None:
        radar.attrs["start_time"] = start_time
    if end_time is not None:
        radar.attrs["end_time"] = end_time
    radar.attrs["combined_from_paths"] = [str(s.path) for s in source_infos]
    radar.attrs["combined_years"] = years
    radar.attrs["combined_total_time"] = total_time

    print(f"[done] output={out_path} total_time={total_time}")


def validate(args: argparse.Namespace) -> None:
    source_paths = [Path(p) for p in args.sources]
    source_infos = [inspect_source(p) for p in source_paths]
    source_infos.sort(key=lambda s: s.year)

    out = zarr.open_group(args.output, mode="r")
    radar = out["radar"]
    out_rt = radar["rtcor"]
    out_tm = radar["time_mask"]
    out_mg = radar["max_intensity_grid"]

    expected_total = sum(s.n_time for s in source_infos)
    if out_rt.shape[0] != expected_total:
        raise ValueError(f"Total time mismatch: output={out_rt.shape[0]} expected={expected_total}")
    if out_tm.shape[0] != expected_total or out_mg.shape[0] != expected_total:
        raise ValueError("time axis mismatch across output arrays")

    # Per-year mask totals and random slice equivalence.
    rng = np.random.default_rng(args.seed)
    offset = 0
    report = {"per_year": [], "sample_checks": []}
    for src in source_infos:
        src_radar = zarr.open_group(str(src.path), mode="r")["radar"]
        src_tm = src_radar["time_mask"][:]
        src_valid = int(np.count_nonzero(src_tm))
        out_valid = int(np.count_nonzero(out_tm[offset : offset + src.n_time]))
        if src_valid != out_valid:
            raise ValueError(f"time_mask valid mismatch year={src.year}: src={src_valid} out={out_valid}")
        report["per_year"].append({"year": src.year, "valid_src": src_valid, "valid_out": out_valid})

        for _ in range(args.samples_per_year):
            idx = int(rng.integers(0, src.n_time))
            out_idx = offset + idx
            src_rt = src_radar["rtcor"][idx]
            out_rt_one = out_rt[out_idx]
            src_mg = src_radar["max_intensity_grid"][idx]
            out_mg_one = out_mg[out_idx]
            src_tm_one = bool(src_radar["time_mask"][idx])
            out_tm_one = bool(out_tm[out_idx])
            if not np.array_equal(src_rt, out_rt_one):
                raise ValueError(f"rtcor mismatch year={src.year} idx={idx}")
            if not np.array_equal(src_mg, out_mg_one):
                raise ValueError(f"max_intensity_grid mismatch year={src.year} idx={idx}")
            if src_tm_one != out_tm_one:
                raise ValueError(f"time_mask mismatch year={src.year} idx={idx}")
            report["sample_checks"].append({"year": src.year, "local_idx": idx, "global_idx": out_idx})

        offset += src.n_time

    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"[ok] validation passed report={report_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Combine old per-year radar Zarr stores.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_combine = sub.add_parser("combine", help="Combine source stores into one output.")
    p_combine.add_argument("--sources", nargs="+", required=True)
    p_combine.add_argument(
        "--output",
        required=True,
        help="Output combined Zarr path.",
    )
    p_combine.add_argument(
        "--state-path",
        default="data_cache/combine_state.json",
    )
    p_combine.add_argument("--chunk-frames", type=int, default=128)
    p_combine.set_defaults(func=combine)

    p_validate = sub.add_parser("validate", help="Run structural and sample validation checks.")
    p_validate.add_argument("--sources", nargs="+", required=True)
    p_validate.add_argument(
        "--output",
        required=True,
        help="Combined Zarr path to validate.",
    )
    p_validate.add_argument("--samples-per-year", type=int, default=5)
    p_validate.add_argument("--seed", type=int, default=1234)
    p_validate.add_argument(
        "--report-path",
        default="data_cache/validation_report.json",
    )
    p_validate.set_defaults(func=validate)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
