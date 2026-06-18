#!/usr/bin/env python3
"""
Create GitHub Release shards for the generated dataset.

This script copies small metadata files and splits large binary files into
fixed-size shard files. The output directory can be uploaded as GitHub Release
assets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path


DATASET_FILES = [
    "X.npy",
    "y.npy",
    "snr_db.npy",
    "channel_profile.npy",
    "mode_names.json",
    "channel_profiles.json",
    "meta.json",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def split_file(path: Path, out_dir: Path, shard_size: int) -> list[dict[str, object]]:
    shards = []
    total_size = path.stat().st_size
    total_parts = int(math.ceil(total_size / shard_size))
    with path.open("rb") as src:
        for part_index in range(total_parts):
            shard_name = f"{path.name}.part{part_index:03d}"
            shard_path = out_dir / shard_name
            remaining = shard_size
            h = hashlib.sha256()
            written = 0
            with shard_path.open("wb") as dst:
                while remaining > 0:
                    chunk = src.read(min(8 * 1024 * 1024, remaining))
                    if not chunk:
                        break
                    dst.write(chunk)
                    h.update(chunk)
                    written += len(chunk)
                    remaining -= len(chunk)
            shards.append(
                {
                    "name": shard_name,
                    "bytes": written,
                    "sha256": h.hexdigest(),
                    "part_index": part_index,
                }
            )
            print(f"Wrote {shard_path} ({written:,} bytes)")
    return shards


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--shard-size-mb", type=int, default=512)
    args = parser.parse_args()

    dataset_root = args.dataset_root
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_size = args.shard_size_mb * 1024 * 1024

    manifest = {
        "dataset_root": str(dataset_root),
        "shard_size_bytes": shard_size,
        "files": [],
    }

    for name in DATASET_FILES:
        src = dataset_root / name
        if not src.exists():
            raise FileNotFoundError(src)

        entry = {
            "name": name,
            "bytes": src.stat().st_size,
            "sha256": sha256_file(src),
            "sharded": src.stat().st_size > shard_size,
            "shards": [],
        }

        if entry["sharded"]:
            entry["shards"] = split_file(src, out_dir, shard_size)
        else:
            dst = out_dir / name
            shutil.copy2(src, dst)
            entry["shards"] = [
                {
                    "name": name,
                    "bytes": dst.stat().st_size,
                    "sha256": sha256_file(dst),
                    "part_index": 0,
                }
            ]
            print(f"Copied {dst} ({dst.stat().st_size:,} bytes)")

        manifest["files"].append(entry)

    manifest_path = out_dir / "release_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Release manifest: {manifest_path}")


if __name__ == "__main__":
    main()
