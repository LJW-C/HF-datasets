#!/usr/bin/env python3
"""
Visual diagnostics for confusing HF digital-mode classes.

The script reads the raw Fldigi WAV files and the generated IQ dataset, then
creates paper/debug figures for checking whether confusing classes are
visually separable.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io import wavfile


DEFAULT_MODES = [
    "Contestia_4_250",
    "Contestia_8_500",
    "Olivia_4_250",
    "Olivia_8_500",
    "DominoEX_8",
    "MFSK16",
    "RTTY",
]


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_wav(path: Path) -> tuple[int, np.ndarray]:
    fs, data = wavfile.read(path)
    data = np.asarray(data)
    if data.ndim == 2:
        data = data.mean(axis=1)
    if np.issubdtype(data.dtype, np.integer):
        max_abs = max(abs(np.iinfo(data.dtype).min), np.iinfo(data.dtype).max)
        data = data.astype(np.float32) / float(max_abs)
    else:
        data = data.astype(np.float32)
    data = data - float(np.mean(data))
    return int(fs), data


def trim_active(x: np.ndarray, fs: int, frame_sec: float = 0.25) -> np.ndarray:
    frame = max(16, int(round(frame_sec * fs)))
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


def to_baseband(x: np.ndarray, fs: int, carrier_hz: float) -> np.ndarray:
    analytic = signal.hilbert(x)
    n = np.arange(len(analytic), dtype=np.float64)
    return analytic * np.exp(-1j * 2.0 * np.pi * carrier_hz * n / fs)


def power_spectrum_db(x: np.ndarray, fs: int, nperseg: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    nperseg = min(nperseg, max(256, len(x)))
    freqs, psd = signal.welch(
        x,
        fs=fs,
        nperseg=nperseg,
        noverlap=nperseg // 2,
        return_onesided=False,
        scaling="density",
    )
    freqs = np.fft.fftshift(freqs)
    psd = np.fft.fftshift(psd)
    psd_db = 10.0 * np.log10(np.maximum(psd, 1e-20))
    return freqs, psd_db


def occupied_bandwidth(freqs: np.ndarray, psd_db: np.ndarray, span_hz: float = 1400.0) -> dict[str, float]:
    mask = np.abs(freqs) <= span_hz
    f = freqs[mask]
    p = np.power(10.0, psd_db[mask] / 10.0)
    if f.size == 0 or float(np.sum(p)) <= 0.0:
        return {"peak_hz": 0.0, "centroid_hz": 0.0, "obw_99_hz": 0.0, "obw_95_hz": 0.0}
    order = np.argsort(f)
    f = f[order]
    p = p[order]
    cumulative = np.cumsum(p)
    cumulative = cumulative / cumulative[-1]
    f005 = float(np.interp(0.005, cumulative, f))
    f995 = float(np.interp(0.995, cumulative, f))
    f025 = float(np.interp(0.025, cumulative, f))
    f975 = float(np.interp(0.975, cumulative, f))
    return {
        "peak_hz": float(f[np.argmax(p)]),
        "centroid_hz": float(np.sum(f * p) / np.sum(p)),
        "obw_99_hz": f995 - f005,
        "obw_95_hz": f975 - f025,
    }


def find_wav(raw_root: Path, mode: str) -> Path | None:
    folder = raw_root / mode
    if folder.exists():
        wavs = sorted(folder.glob("*.wav"))
        if wavs:
            return wavs[0]
    direct = raw_root / f"{mode}.wav"
    if direct.exists():
        return direct
    return None


def plot_raw_wav_diagnostics(
    raw_root: Path,
    modes: list[str],
    out_dir: Path,
    carrier_hz: float,
    max_seconds: float,
) -> list[dict[str, float | str]]:
    rows = []
    fig, axes = plt.subplots(
        len(modes),
        2,
        figsize=(11.0, 2.25 * len(modes)),
        dpi=180,
        constrained_layout=True,
    )
    if len(modes) == 1:
        axes = np.asarray([axes])

    for row_index, mode in enumerate(modes):
        wav_path = find_wav(raw_root, mode)
        ax_psd, ax_spec = axes[row_index]
        if wav_path is None:
            ax_psd.text(0.5, 0.5, f"{mode}\nmissing WAV", ha="center", va="center")
            ax_spec.axis("off")
            continue

        fs, audio = read_wav(wav_path)
        active = trim_active(audio, fs)
        max_len = min(len(active), int(round(max_seconds * fs)))
        segment = active[:max_len]
        baseband = to_baseband(segment, fs, carrier_hz)
        freqs, psd_db = power_spectrum_db(baseband, fs)
        stats = occupied_bandwidth(freqs, psd_db)
        rows.append(
            {
                "mode": mode,
                "wav_path": str(wav_path),
                "fs": fs,
                "duration_active_s": len(active) / fs,
                "rms": float(np.sqrt(np.mean(active * active))),
                "peak": float(np.max(np.abs(active))),
                **stats,
            }
        )

        mask = np.abs(freqs) <= 1000.0
        ax_psd.plot(freqs[mask], psd_db[mask], color="#2f6f9f", linewidth=1.0)
        ax_psd.set_title(f"{mode} raw WAV baseband spectrum", fontsize=9)
        ax_psd.set_xlabel("Frequency after 1500 Hz downmix (Hz)")
        ax_psd.set_ylabel("PSD (dB)")
        ax_psd.grid(True, linestyle="--", alpha=0.3)
        ax_psd.text(
            0.02,
            0.95,
            f"OBW99={stats['obw_99_hz']:.0f} Hz\npeak={stats['peak_hz']:.0f} Hz",
            transform=ax_psd.transAxes,
            ha="left",
            va="top",
            fontsize=7,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )

        nperseg = min(512, max(128, len(baseband) // 12))
        f, t, sxx = signal.spectrogram(
            baseband,
            fs=fs,
            nperseg=nperseg,
            noverlap=nperseg // 2,
            return_onesided=False,
            scaling="density",
            mode="magnitude",
        )
        f = np.fft.fftshift(f)
        sxx = np.fft.fftshift(sxx, axes=0)
        fmask = np.abs(f) <= 1000.0
        image = 20.0 * np.log10(np.maximum(sxx[fmask, :], 1e-8))
        ax_spec.pcolormesh(t, f[fmask], image, shading="auto", cmap="magma")
        ax_spec.set_title(f"{mode} raw WAV spectrogram", fontsize=9)
        ax_spec.set_xlabel("Time (s)")
        ax_spec.set_ylabel("Frequency (Hz)")

    fig.savefig(out_dir / "raw_wav_problem_modes.png", bbox_inches="tight")
    plt.close(fig)
    return rows


def plot_raw_spectrum_overlay(
    raw_root: Path,
    modes: list[str],
    out_dir: Path,
    carrier_hz: float,
    max_seconds: float,
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.7), dpi=200)
    for mode in modes:
        wav_path = find_wav(raw_root, mode)
        if wav_path is None:
            continue
        fs, audio = read_wav(wav_path)
        active = trim_active(audio, fs)
        segment = active[: min(len(active), int(round(max_seconds * fs)))]
        baseband = to_baseband(segment, fs, carrier_hz)
        freqs, psd_db = power_spectrum_db(baseband, fs)
        mask = np.abs(freqs) <= 800.0
        normalized = psd_db[mask] - float(np.max(psd_db[mask]))
        ax.plot(freqs[mask], normalized, linewidth=1.25, label=mode)

    for edge in [-250, -125, 125, 250]:
        ax.axvline(edge, color="0.4", linestyle="--", linewidth=0.8, alpha=0.45)
    ax.text(-125, -4, "250 Hz\nnominal", ha="center", va="top", fontsize=8)
    ax.text(250, -4, "500 Hz\nnominal edge", ha="left", va="top", fontsize=8)
    ax.set_title("Raw WAV baseband spectrum overlay")
    ax.set_xlabel("Frequency after 1500 Hz downmix (Hz)")
    ax.set_ylabel("Normalized PSD (dB)")
    ax.set_xlim(-650, 650)
    ax.set_ylim(-75, 3)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(fontsize=7, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "raw_wav_spectrum_overlay.png", bbox_inches="tight")
    plt.close(fig)


def choose_iq_index(
    y: np.ndarray,
    snr_db: np.ndarray | None,
    channel_profile: np.ndarray | None,
    label: int,
    snr_preference: int,
    profile_preference: int | None,
) -> int | None:
    mask = y == label
    if snr_db is not None:
        snr_mask = mask & (snr_db == snr_preference)
        if np.any(snr_mask):
            mask = snr_mask
    if channel_profile is not None and profile_preference is not None:
        profile_mask = mask & (channel_profile == profile_preference)
        if np.any(profile_mask):
            mask = profile_mask
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return None
    return int(indices[0])


def plot_iq_diagnostics(
    dataset_root: Path,
    modes: list[str],
    out_dir: Path,
    snr_preference: int,
    profile_name: str,
) -> None:
    x = np.load(dataset_root / "X.npy", mmap_mode="r")
    y = np.load(dataset_root / "y.npy", mmap_mode="r")
    snr_db = np.load(dataset_root / "snr_db.npy", mmap_mode="r") if (dataset_root / "snr_db.npy").exists() else None
    channel_profile = (
        np.load(dataset_root / "channel_profile.npy", mmap_mode="r")
        if (dataset_root / "channel_profile.npy").exists()
        else None
    )
    mode_names = read_json(dataset_root / "mode_names.json")
    channel_profiles = read_json(dataset_root / "channel_profiles.json") if (dataset_root / "channel_profiles.json").exists() else []
    meta = read_json(dataset_root / "meta.json") if (dataset_root / "meta.json").exists() else {}
    fs = int(meta.get("target_fs", meta.get("fs", 6000)))
    profile_preference = channel_profiles.index(profile_name) if profile_name in channel_profiles else None

    selected = []
    for mode in modes:
        if mode not in mode_names:
            continue
        idx = choose_iq_index(
            y,
            snr_db,
            channel_profile,
            mode_names.index(mode),
            snr_preference,
            profile_preference,
        )
        if idx is not None:
            selected.append((mode, idx))

    fig, axes = plt.subplots(
        len(selected),
        4,
        figsize=(16.0, 2.35 * len(selected)),
        dpi=180,
        constrained_layout=True,
    )
    if len(selected) == 1:
        axes = np.asarray([axes])

    for row_index, (mode, idx) in enumerate(selected):
        sample = np.asarray(x[idx], dtype=np.complex64)
        t_ms = np.arange(len(sample)) / fs * 1000.0
        title_suffix = f"idx={idx}"
        if snr_db is not None:
            title_suffix += f", SNR={int(snr_db[idx])} dB"
        if channel_profile is not None and channel_profiles:
            title_suffix += f", {channel_profiles[int(channel_profile[idx])]}"

        ax_iq, ax_const, ax_psd, ax_spec = axes[row_index]
        ax_iq.plot(t_ms, sample.real, linewidth=0.8, label="I")
        ax_iq.plot(t_ms, sample.imag, linewidth=0.8, label="Q")
        ax_iq.set_title(f"{mode} I/Q waveform ({title_suffix})", fontsize=8)
        ax_iq.set_xlabel("Time (ms)")
        ax_iq.set_ylabel("Amplitude")
        ax_iq.grid(True, linestyle="--", alpha=0.3)
        ax_iq.legend(fontsize=6, frameon=False, loc="upper right")

        ax_const.plot(sample.real, sample.imag, ".", markersize=1.4, alpha=0.45)
        ax_const.set_title(f"{mode} constellation/trajectory", fontsize=8)
        ax_const.set_xlabel("I")
        ax_const.set_ylabel("Q")
        ax_const.grid(True, linestyle="--", alpha=0.25)
        lim = max(1e-3, float(np.percentile(np.abs(np.r_[sample.real, sample.imag]), 99.5)))
        ax_const.set_xlim(-1.2 * lim, 1.2 * lim)
        ax_const.set_ylim(-1.2 * lim, 1.2 * lim)

        freqs, psd_db = power_spectrum_db(sample, fs, nperseg=min(1024, len(sample)))
        mask = np.abs(freqs) <= 1500.0
        ax_psd.plot(freqs[mask], psd_db[mask], color="#2f6f9f", linewidth=1.0)
        ax_psd.set_title(f"{mode} IQ spectrum", fontsize=8)
        ax_psd.set_xlabel("Frequency (Hz)")
        ax_psd.set_ylabel("PSD (dB)")
        ax_psd.grid(True, linestyle="--", alpha=0.3)

        f, tt, sxx = signal.spectrogram(
            sample,
            fs=fs,
            nperseg=256,
            noverlap=192,
            return_onesided=False,
            scaling="density",
            mode="magnitude",
        )
        f = np.fft.fftshift(f)
        sxx = np.fft.fftshift(sxx, axes=0)
        fmask = np.abs(f) <= 1500.0
        ax_spec.pcolormesh(tt * 1000.0, f[fmask], 20 * np.log10(np.maximum(sxx[fmask], 1e-8)), shading="auto", cmap="magma")
        ax_spec.set_title(f"{mode} IQ spectrogram", fontsize=8)
        ax_spec.set_xlabel("Time (ms)")
        ax_spec.set_ylabel("Frequency (Hz)")

    fig.savefig(out_dir / "iq_problem_mode_samples.png", bbox_inches="tight")
    plt.close(fig)


def write_stats_csv(rows: list[dict[str, float | str]], out_path: Path) -> None:
    if not rows:
        return
    fieldnames = [
        "mode",
        "fs",
        "duration_active_s",
        "rms",
        "peak",
        "peak_hz",
        "centroid_hz",
        "obw_95_hz",
        "obw_99_hz",
        "wav_path",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path(r"E:\仿真\panoradio_like_ccir520_npy"))
    parser.add_argument("--raw-root", type=Path, default=Path(r"E:\仿真\raw_wav"))
    parser.add_argument("--out-dir", type=Path, default=Path("diagnostics/mode_visuals"))
    parser.add_argument("--modes", default=",".join(DEFAULT_MODES))
    parser.add_argument("--carrier-hz", type=float, default=1500.0)
    parser.add_argument("--max-seconds", type=float, default=18.0)
    parser.add_argument("--snr", type=int, default=25)
    parser.add_argument("--profile", default="good")
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    modes = [item.strip() for item in args.modes.split(",") if item.strip()]

    raw_rows = plot_raw_wav_diagnostics(args.raw_root, modes, out_dir, args.carrier_hz, args.max_seconds)
    plot_raw_spectrum_overlay(args.raw_root, modes, out_dir, args.carrier_hz, args.max_seconds)
    write_stats_csv(raw_rows, out_dir / "raw_wav_spectral_stats.csv")
    plot_iq_diagnostics(args.dataset_root, modes, out_dir, args.snr, args.profile)

    print(f"Wrote diagnostics to {out_dir}")
    print(f"Raw WAV figure: {out_dir / 'raw_wav_problem_modes.png'}")
    print(f"Raw spectrum overlay: {out_dir / 'raw_wav_spectrum_overlay.png'}")
    print(f"IQ sample figure: {out_dir / 'iq_problem_mode_samples.png'}")
    print(f"Raw spectral stats: {out_dir / 'raw_wav_spectral_stats.csv'}")


if __name__ == "__main__":
    main()
