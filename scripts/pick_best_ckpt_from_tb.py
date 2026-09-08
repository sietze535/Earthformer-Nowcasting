#!/usr/bin/env python3
"""Pick evaluation checkpoint from TensorBoard val/loss (nearest step on disk)."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def _parse_step_from_ckpt(path: Path) -> int | None:
    m = re.search(r"step=(\d+)\.ckpt$", path.name)
    if m:
        return int(m.group(1))
    return None


def pick_best_ckpt(version_dir: Path, prefer_best_ckpt: bool = True) -> dict:
    version_dir = Path(version_dir)
    ckpt_dir = version_dir / "checkpoints"
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"Missing checkpoints dir: {ckpt_dir}")

    ea = EventAccumulator(str(version_dir), size_guidance={"scalars": 0})
    ea.Reload()
    tag = "val/loss"
    if tag not in ea.Tags().get("scalars", []):
        raise KeyError(f"No {tag} in {version_dir}")
    events = ea.Scalars(tag)
    best = min(events, key=lambda e: e.value)
    best_step = int(best.step)
    best_val = float(best.value)

    periodic = []
    for p in ckpt_dir.glob("epoch=*-step=*.ckpt"):
        step = _parse_step_from_ckpt(p)
        if step is not None:
            periodic.append((step, p))
    periodic.sort(key=lambda x: x[0])

    best_ckpt = ckpt_dir / "best.ckpt"
    chosen_path: Path | None = None
    chosen_step = best_step
    source = "nearest_periodic"

    if prefer_best_ckpt and best_ckpt.is_file():
        chosen_path = best_ckpt
        chosen_step = best_step
        source = "best.ckpt"
    elif periodic:
        nearest_step, nearest_path = min(periodic, key=lambda x: abs(x[0] - best_step))
        chosen_path = nearest_path
        chosen_step = nearest_step
        source = "nearest_periodic"

    if chosen_path is None:
        last_candidates = sorted(ckpt_dir.glob("last*.ckpt"))
        if not last_candidates:
            raise FileNotFoundError(f"No usable checkpoint in {ckpt_dir}")
        chosen_path = last_candidates[-1]
        source = "last.ckpt_fallback"

    return {
        "version_dir": str(version_dir),
        "best_val_loss": best_val,
        "best_val_loss_step": best_step,
        "checkpoint_path": str(chosen_path.resolve()),
        "checkpoint_step": chosen_step,
        "selection": source,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("version_dir", type=Path)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    result = pick_best_ckpt(args.version_dir)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with args.json_out.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    print(result["checkpoint_path"])
    print(f"# step={result['checkpoint_step']} val_loss={result['best_val_loss']:.6f} @tb_step={result['best_val_loss_step']} ({result['selection']})", file=sys.stderr)


if __name__ == "__main__":
    main()
