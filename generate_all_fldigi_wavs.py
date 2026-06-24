"""
Record every modem name exposed by the current Fldigi installation.

This script discovers modes with fldigi.modem.get_names(), converts each modem
name into a safe folder/file name, and records one WAV per modem.

Typical use:
    python generate_all_fldigi_wavs.py --device-id 7 --out-root "E:\\raw_wav_all"

Use --dry-run first to see exactly which modes will be recorded.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path

from generate_fldigi_wavs import (
    DEFAULT_FLDIGI_XMLRPC_URL,
    connect_fldigi,
    list_audio_devices,
    record_one_mode,
)


def safe_folder_name(modem_name: str) -> str:
    name = re.sub(r"[^0-9A-Za-z]+", "_", modem_name).strip("_")
    return name or "mode"


def unique_folder_name(folder: str, used: set[str]) -> str:
    if folder not in used:
        used.add(folder)
        return folder

    index = 2
    while f"{folder}_{index}" in used:
        index += 1
    folder = f"{folder}_{index}"
    used.add(folder)
    return folder


def get_fldigi_modem_names(fldigi) -> list[str]:
    names = list(fldigi.modem.get_names())
    return [str(name).strip() for name in names if str(name).strip()]


def build_mode_table(modem_names: list[str]) -> list[dict[str, str]]:
    used: set[str] = set()
    modes = []
    for modem_name in modem_names:
        folder = unique_folder_name(safe_folder_name(modem_name), used)
        modes.append({"folder": folder, "fldigi_modem": modem_name})
    return modes


def apply_filters(
    modes: list[dict[str, str]],
    include_regex: str | None,
    exclude_regex: str | None,
    start_index: int,
    end_index: int | None,
) -> list[dict[str, str]]:
    filtered = modes

    if include_regex:
        include = re.compile(include_regex, re.IGNORECASE)
        filtered = [
            mode
            for mode in filtered
            if include.search(mode["fldigi_modem"]) or include.search(mode["folder"])
        ]

    if exclude_regex:
        exclude = re.compile(exclude_regex, re.IGNORECASE)
        filtered = [
            mode
            for mode in filtered
            if not exclude.search(mode["fldigi_modem"]) and not exclude.search(mode["folder"])
        ]

    if start_index < 0:
        raise ValueError("--start-index must be >= 0")

    return filtered[start_index:end_index]


def wav_path_for_mode(out_root: Path, mode: dict[str, str]) -> Path:
    folder = mode["folder"]
    return out_root / folder / f"{folder}.wav"


def save_mode_manifest(out_root: Path, modes: list[dict[str, str]], args: argparse.Namespace) -> Path:
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "xmlrpc_url": args.xmlrpc_url,
        "out_root": str(out_root),
        "device_id": args.device_id,
        "record_fs": args.record_fs,
        "duration_sec": args.duration_sec,
        "carrier_hz": args.carrier_hz,
        "payload_chars": args.payload_chars,
        "skip_existing": args.skip_existing,
        "include_regex": args.include_regex,
        "exclude_regex": args.exclude_regex,
        "start_index": args.start_index,
        "end_index": args.end_index,
        "mode_count": len(modes),
        "modes": modes,
    }
    manifest_path = out_root / "all_fldigi_modems_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Record all Fldigi modem names as WAV files.")
    parser.add_argument("--xmlrpc-url", default=DEFAULT_FLDIGI_XMLRPC_URL, help="Fldigi XML-RPC URL.")
    parser.add_argument("--device-id", type=int, default=7, help="Recording device id, e.g. VB-CABLE Output.")
    parser.add_argument("--record-fs", type=int, default=8000, help="Recording sample rate.")
    parser.add_argument("--duration-sec", type=int, default=180, help="Recording duration per mode.")
    parser.add_argument("--carrier-hz", type=float, default=1500.0, help="Fldigi audio carrier frequency.")
    parser.add_argument("--payload-chars", type=int, default=30000, help="Random text payload length.")
    parser.add_argument("--out-root", default=r"E:\raw_wav_all", help="Output root folder for all WAV files.")
    parser.add_argument("--include-regex", default=None, help="Only record modem/folder names matching this regex.")
    parser.add_argument("--exclude-regex", default=None, help="Skip modem/folder names matching this regex.")
    parser.add_argument("--start-index", type=int, default=0, help="Resume from this filtered mode index.")
    parser.add_argument("--end-index", type=int, default=None, help="Stop before this filtered mode index.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip modes whose WAV file already exists.")
    parser.add_argument("--dry-run", action="store_true", help="Only print the mode list; do not record.")
    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit.")
    args = parser.parse_args()

    if args.list_devices:
        list_audio_devices()
        return

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    fldigi = connect_fldigi(args.xmlrpc_url)
    modem_names = get_fldigi_modem_names(fldigi)
    all_modes = build_mode_table(modem_names)
    selected_modes = apply_filters(
        modes=all_modes,
        include_regex=args.include_regex,
        exclude_regex=args.exclude_regex,
        start_index=args.start_index,
        end_index=args.end_index,
    )

    manifest_path = save_mode_manifest(out_root, selected_modes, args)

    print("\nFldigi modems selected for recording:")
    for index, mode in enumerate(selected_modes):
        print(f"{index:04d}  {mode['folder']}  <=  {mode['fldigi_modem']}")

    print(f"\nSelected mode count: {len(selected_modes)}")
    print(f"Manifest: {manifest_path}")
    print(f"Estimated recording time: {len(selected_modes) * args.duration_sec / 3600.0:.2f} hours")

    if args.dry_run:
        print("\nDry run only. No WAV files recorded.")
        return

    results: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []

    for index, mode in enumerate(selected_modes, start=1):
        out_path = wav_path_for_mode(out_root, mode)
        if args.skip_existing and out_path.exists():
            print(f"\nSkipping existing WAV [{index}/{len(selected_modes)}]: {out_path}")
            continue

        try:
            print(f"\nRecording [{index}/{len(selected_modes)}]: {mode['fldigi_modem']}")
            result = record_one_mode(
                fldigi=fldigi,
                mode=mode,
                out_root=out_root,
                device_id=args.device_id,
                record_fs=args.record_fs,
                duration_sec=args.duration_sec,
                carrier_hz=args.carrier_hz,
                payload_chars=args.payload_chars,
                backup_existing=not args.skip_existing,
            )
            results.append(result)
        except Exception as exc:
            failures.append(
                {
                    "folder": mode["folder"],
                    "fldigi_modem": mode["fldigi_modem"],
                    "error": str(exc),
                }
            )
            print(f"Failed: {mode['fldigi_modem']} -> {exc}")
            try:
                fldigi.main.rx()
            except Exception:
                pass
            time.sleep(2)

        progress_path = out_root / "all_fldigi_recording_progress.json"
        with progress_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                    "recorded_count": len(results),
                    "failure_count": len(failures),
                    "recorded": results,
                    "failures": failures,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    print("\nAll selected Fldigi modes processed.")
    print(f"Recorded: {len(results)}")
    print(f"Failed:   {len(failures)}")
    print(f"Output:   {out_root}")


if __name__ == "__main__":
    main()
