#!/usr/bin/env python3
"""
Validate Fldigi raw WAV recordings before building the IQ dataset.

The script estimates the baseband occupied bandwidth after removing the
configured Fldigi audio carrier. Modes with explicit bandwidth in their label,
such as Olivia_4_250 and Contestia_8_500, are checked against that expected
bandwidth.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.io import wavfile


EXPECTED_OBW_HZ = {
    "Olivia_4_250": 250.0,
    "Olivia_8_500": 500.0,
    "Contestia_4_250": 250.0,
    "Contestia_8_500": 500.0,
}


def read_wav(path: Path) -> tuple[int, np.ndarray]:
    fs, data = wavfile.read(path)
    data = np.asarray(data)
    if data.ndim == 2:
        data = data.astype(np.float32).mean(axis=1)
    if np.issubdtype(data.dtype, np.integer):
        info = np.iinfo(data.dtype)
        scale = float(max(abs(info.min), info.max))
        data = data.astype(np.float32) / scale
    else:
        data = data.astype(np.float32)
    data = data - float(np.mean(data))
    return int(fs), data


def find_wavs(raw_root: Path) -> list[tuple[str, Path]]:
    wavs: list[tuple[str, Path]] = []
    for folder in sorted(raw_root.iterdir()):
        if not folder.is_dir():
            continue
        exact = folder / f"{folder.name}.wav"
        if exact.exists():
            wavs.append((folder.name, exact))
            continue
        candidates = sorted(folder.glob("*.wav"))
        if candidates:
            wavs.append((folder.name, candidates[0]))
    return wavs


def trim_active(x: np.ndarray, fs: int) -> np.ndarray:
    frame = max(16, int(round(0.25 * fs)))
    kernel = np.ones(frame, dtype=np.float32) / float(frame)
    energy = signal.fftconvolve(x * x, kernel, mode="same")
    rms = np.sqrt(np.maximum(energy, 0.0) + 1e-12)
    threshold = max(1e-5, 0.03 * float(np.percentile(rms, 95)))
    active = np.flatnonzero(rms > threshold)
    if active.size == 0:
        return x
    pad = int(round(0.25 * fs))
    start = max(0, int(active[0]) - pad)
    stop = min(len(x), int(active[-1]) + pad)
    return x[start:stop]


def estimate_obw(audio: np.ndarray, fs: int, carrier_hz: float, max_seconds: float) -> dict[str, float]:
    active = trim_active(audio, fs)
    segment = active[: min(len(active), int(round(max_seconds * fs)))]
    analytic = signal.hilbert(segment)
    n = np.arange(len(analytic), dtype=np.float64)
    baseband = analytic * np.exp(-1j * 2.0 * np.pi * carrier_hz * n / fs)
    nperseg = min(4096, max(256, len(baseband)))
    freqs, psd = signal.welch(
        baseband,
        fs=fs,
        nperseg=nperseg,
        noverlap=nperseg // 2,
        return_onesided=False,
        scaling="density",
    )
    freqs = np.fft.fftshift(freqs)
    psd = np.fft.fftshift(psd)
    mask = np.abs(freqs) <= 1400.0
    f = freqs[mask]
    p = psd[mask]
    order = np.argsort(f)
    f = f[order]
    p = p[order]
    cumulative = np.cumsum(p)
    cumulative = cumulative / cumulative[-1]
    return {
        "duration_active_s": float(len(active) / fs),
        "rms": float(np.sqrt(np.mean(active * active))),
        "peak": float(np.max(np.abs(active))),
        "peak_hz": float(f[np.argmax(p)]),
        "centroid_hz": float(np.sum(f * p) / np.sum(p)),
        "obw_95_hz": float(np.interp(0.975, cumulative, f) - np.interp(0.025, cumulative, f)),
        "obw_99_hz": float(np.interp(0.995, cumulative, f) - np.interp(0.005, cumulative, f)),
    }


def classify_status(mode: str, stats: dict[str, float]) -> tuple[str, str]:
    if stats["rms"] < 1e-4 or stats["peak"] < 1e-4:
        return "FAIL_SILENT", "recording is near silent"

    expected = EXPECTED_OBW_HZ.get(mode)
    if expected is None:
        return "INFO", "no explicit bandwidth check configured"

    obw = stats["obw_99_hz"]
    lower = 0.70 * expected
    upper = 1.35 * expected
    if lower <= obw <= upper:
        return "PASS", f"expected about {expected:.0f} Hz"
    return "FAIL_BANDWIDTH", f"expected about {expected:.0f} Hz, measured {obw:.1f} Hz"


def write_reports(rows: list[dict[str, object]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "raw_wav_validation.csv"
    md_path = out_dir / "raw_wav_validation.md"
    fields = [
        "mode",
        "status",
        "reason",
        "expected_obw_hz",
        "obw_99_hz",
        "obw_95_hz",
        "peak_hz",
        "centroid_hz",
        "rms",
        "peak",
        "duration_active_s",
        "fs",
        "wav_path",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    lines = [
        "# Raw WAV Validation Report",
        "",
        "| Mode | Status | Expected OBW | OBW99 | OBW95 | Peak Freq. | Reason |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        expected = row["expected_obw_hz"]
        expected_text = "--" if expected == "" else f"{float(expected):.0f} Hz"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["mode"]),
                    str(row["status"]),
                    expected_text,
                    f"{float(row['obw_99_hz']):.1f} Hz",
                    f"{float(row['obw_95_hz']):.1f} Hz",
                    f"{float(row['peak_hz']):.1f} Hz",
                    str(row["reason"]),
                ]
            )
            + " |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"CSV report: {csv_path}")
    print(f"Markdown report: {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path(r"E:\仿真\raw_wav"))
    parser.add_argument("--out-dir", type=Path, default=Path("diagnostics/raw_wav_validation"))
    parser.add_argument("--carrier-hz", type=float, default=1500.0)
    parser.add_argument("--max-seconds", type=float, default=18.0)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for mode, wav_path in find_wavs(args.raw_root):
        fs, audio = read_wav(wav_path)
        stats = estimate_obw(audio, fs, args.carrier_hz, args.max_seconds)
        status, reason = classify_status(mode, stats)
        row = {
            "mode": mode,
            "status": status,
            "reason": reason,
            "expected_obw_hz": EXPECTED_OBW_HZ.get(mode, ""),
            "fs": fs,
            "wav_path": str(wav_path),
            **stats,
        }
        rows.append(row)
        print(
            f"{mode:18s} {status:14s} "
            f"OBW99={stats['obw_99_hz']:7.1f} Hz "
            f"OBW95={stats['obw_95_hz']:7.1f} Hz "
            f"peak={stats['peak_hz']:7.1f} Hz"
        )

    write_reports(rows, args.out_dir)
    failures = [row for row in rows if str(row["status"]).startswith("FAIL")]
    if failures:
        print("\nFailures:")
        for row in failures:
            print(f"  {row['mode']}: {row['reason']}")
    else:
        print("\nNo validation failures found.")


if __name__ == "__main__":
    main()
