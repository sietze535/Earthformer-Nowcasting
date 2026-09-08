#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

DATASET = "hourly-in-situ-meteorological-observations-validated"
BASE_FILES = "https://api.dataplatform.knmi.nl/open-data/v1/datasets/{}/versions/1.0/files"
URL_TMPL = "https://api.dataplatform.knmi.nl/open-data/v1/datasets/{}/versions/1.0/files/{}/url"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download full KNMI hourly validated files for 2008-2019.")
    p.add_argument("--out-root", required=True, help="Output folder, e.g. .../raw_hourly_validated_full")
    p.add_argument("--start-year", type=int, default=2008)
    p.add_argument("--end-year", type=int, default=2019)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--max-files", type=int, default=0, help="Debug limit; 0 means all")
    p.add_argument("--log-json", required=True, help="Summary/progress json path")
    return p.parse_args()


def api_list(headers: dict[str, str], **params) -> dict:
    r = requests.get(BASE_FILES.format(DATASET), headers=headers, params=params, timeout=60)
    r.raise_for_status()
    return r.json()


def list_all_files(
    headers: dict[str, str], start_year: int, end_year: int, max_keys: int = 500
) -> list[dict]:
    files: list[dict] = []
    token = None
    begin_filename = f"hourly-observations-validated-{start_year}0101-00.nc"
    while True:
        params = {"maxKeys": max_keys}
        if token:
            params["nextPageToken"] = token
        else:
            params["begin"] = begin_filename
        page = api_list(headers, **params)
        page_files = page.get("files", [])
        stop = False
        for f in page_files:
            fn = f.get("filename", "")
            if not fn.startswith("hourly-observations-validated-"):
                continue
            try:
                year = int(fn[30:34])
            except Exception:
                continue
            if year > end_year:
                stop = True
                break
            files.append(f)
        token = page.get("nextPageToken")
        if stop or not token:
            break
    return files


def get_temp_url(headers: dict[str, str], filename: str) -> str:
    r = requests.get(URL_TMPL.format(DATASET, filename), headers=headers, timeout=60)
    r.raise_for_status()
    return r.json()["temporaryDownloadUrl"]


def download_one(headers: dict[str, str], out_path: Path, timeout: int, item: tuple[str, str]) -> tuple[str, str, int]:
    month, filename = item
    month_dir = out_path / month
    month_dir.mkdir(parents=True, exist_ok=True)
    dst = month_dir / filename
    if dst.exists() and dst.stat().st_size > 0:
        return month, "skipped", int(dst.stat().st_size)

    tmp = dst.with_suffix(dst.suffix + ".part")
    for attempt in range(1, 4):
        try:
            url = get_temp_url(headers, filename)
            with requests.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                with tmp.open("wb") as w:
                    for chunk in r.iter_content(chunk_size=1024 * 512):
                        if chunk:
                            w.write(chunk)
            tmp.replace(dst)
            return month, "downloaded", int(dst.stat().st_size)
        except Exception:
            if attempt == 3:
                if tmp.exists():
                    tmp.unlink(missing_ok=True)
                return month, "error", 0
            time.sleep(1.5 * attempt)
    return month, "error", 0


def main() -> None:
    args = parse_args()
    key = os.environ.get("KNMI_API_KEY")
    if not key:
        raise RuntimeError("KNMI_API_KEY not set")
    headers = {"Authorization": key}

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    log_json = Path(args.log_json)
    log_json.parent.mkdir(parents=True, exist_ok=True)

    monthly_listing: dict[str, int] = {}
    items: list[tuple[str, str]] = []
    all_files = list_all_files(headers, args.start_year, args.end_year, max_keys=500)
    for f in all_files:
        fn = f.get("filename", "")
        if not fn.startswith("hourly-observations-validated-"):
            continue
        try:
            # "hourly-observations-validated-" is 30 chars.
            year = int(fn[30:34])
            month = fn[30:36]
        except Exception:
            continue
        if year < args.start_year or year > args.end_year:
            continue
        monthly_listing[month] = monthly_listing.get(month, 0) + 1
        items.append((month, fn))

    if args.max_files > 0:
        items = items[: args.max_files]

    summary = {
        "out_root": str(out_root),
        "start_year": args.start_year,
        "end_year": args.end_year,
        "workers": args.workers,
        "listed_files_total": len(items),
        "monthly_listing_counts": monthly_listing,
        "downloaded": 0,
        "skipped": 0,
        "errors": 0,
        "bytes_total": 0,
        "updated_at_utc": None,
    }
    log_json.write_text(json.dumps(summary, indent=2))

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(download_one, headers, out_root, args.timeout, item) for item in items]
        for i, fut in enumerate(as_completed(futs), start=1):
            month, status, nbytes = fut.result()
            if status == "downloaded":
                summary["downloaded"] += 1
            elif status == "skipped":
                summary["skipped"] += 1
            else:
                summary["errors"] += 1
            summary["bytes_total"] += int(nbytes)
            if i % 500 == 0:
                summary["updated_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                log_json.write_text(json.dumps(summary, indent=2))
                print(
                    f"[progress] done={i}/{len(items)} downloaded={summary['downloaded']} "
                    f"skipped={summary['skipped']} errors={summary['errors']}",
                    flush=True,
                )

    summary["updated_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    log_json.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
