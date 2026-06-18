#!/usr/bin/env python3
"""
Restore dataset files from GitHub Release shards.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def restore_file(file_entry: dict, shard_dir: Path, out_root: Path) -> None:
    out_path = out_root / file_entry["name"]
    if file_entry["sharded"]:
        with out_path.open("wb") as dst:
            for shard in sorted(file_entry["shards"], key=lambda x: int(x["part_index"])):
                shard_path = shard_dir / shard["name"]
                if not shard_path.exists():
                    raise FileNotFoundError(shard_path)
                got = sha256_file(shard_path)
                if got != shard["sha256"]:
                    raise RuntimeError(f"Checksum mismatch for {shard_path}")
                with shard_path.open("rb") as src:
                    shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
    else:
        shard = file_entry["shards"][0]
        src_path = shard_dir / shard["name"]
        if not src_path.exists():
            raise FileNotFoundError(src_path)
        shutil.copy2(src_path, out_path)

    got = sha256_file(out_path)
    if got != file_entry["sha256"]:
        raise RuntimeError(f"Restored checksum mismatch for {out_path}")
    print(f"Restored {out_path} ({out_path.stat().st_size:,} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()

    manifest_path = args.shard_dir / "release_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    args.out_root.mkdir(parents=True, exist_ok=True)
    for file_entry in manifest["files"]:
        restore_file(file_entry, args.shard_dir, args.out_root)

    print("Dataset restored successfully.")


if __name__ == "__main__":
    main()
