"""
Visualize Panoradio-like IQ dataset samples.

Each output PNG contains:
  1. I/Q time-domain waveform
  2. IQ constellation / trajectory
  3. FFT magnitude spectrum
  4. Time-frequency spectrogram

Example:
    python visualize_iq_samples.py --dataset-root E:\\仿真\\panoradio_like_ccir520_npy --indices 0,10,100
    python visualize_iq_samples.py --dataset-root E:\\仿真\\panoradio_like_ccir520_npy --random 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


def load_json_list(path: Path) -> list[str] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_indices(value: str) -> list[int]:
    indices: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            pieces = [int(x) if x else None for x in part.split(":")]
            start = pieces[0] if len(pieces) > 0 and pieces[0] is not None else 0
            stop = pieces[1] if len(pieces) > 1 and pieces[1] is not None else start + 1
            step = pieces[2] if len(pieces) > 2 and pieces[2] is not None else 1
            indices.extend(range(start, stop, step))
        else:
            indices.append(int(part))
    return indices


def make_sample_title(
    index: int,
    y: np.ndarray | None,
    snr_db: np.ndarray | None,
    channel_profile: np.ndarray | None,
    mode_names: list[str] | None,
    channel_profiles: list[str] | None,
) -> str:
    parts = [f"sample {index}"]
    if y is not None:
        label = int(y[index])
        mode = mode_names[label] if mode_names and 0 <= label < len(mode_names) else f"class {label}"
        parts.append(mode)
    if snr_db is not None:
        parts.append(f"SNR {int(snr_db[index])} dB")
    if channel_profile is not None:
        ch = int(channel_profile[index])
        ch_name = channel_profiles[ch] if channel_profiles and 0 <= ch < len(channel_profiles) else f"profile {ch}"
        parts.append(ch_name)
    return " | ".join(parts)


def plot_iq_sample(
    x: np.ndarray,
    fs: int,
    title: str,
    out_path: Path,
    fft_size: int,
    spectrogram_nperseg: int,
) -> None:
    x = np.asarray(x, dtype=np.complex64)
    t_ms = np.arange(len(x)) / fs * 1000.0

    window = np.hanning(len(x))
    spectrum = np.fft.fftshift(np.fft.fft(x * window, n=fft_size))
    freqs = np.fft.fftshift(np.fft.fftfreq(fft_size, d=1.0 / fs))
    spectrum_db = 20.0 * np.log10(np.abs(spectrum) + 1e-12)
    spectrum_db -= np.max(spectrum_db)

    nperseg = min(spectrogram_nperseg, len(x))
    noverlap = int(nperseg * 0.75)
    stft_freqs, stft_times, zxx = signal.stft(
        x,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=max(fft_size, nperseg),
        boundary=None,
        padded=False,
        return_onesided=False,
    )
    order = np.argsort(stft_freqs)
    stft_freqs = stft_freqs[order]
    zxx = zxx[order, :]
    spectrogram_db = 20.0 * np.log10(np.abs(zxx) + 1e-12)
    spectrogram_db -= np.max(spectrogram_db)

    fig, axes = plt.subplots(2, 2, figsize=(13.8, 9.0), constrained_layout=True)
    axes = axes.ravel()
    fig.suptitle(title, fontsize=13, fontweight="bold")

    ax = axes[0]
    ax.plot(t_ms, x.real, color="#1F6F8B", linewidth=1.0, label="I")
    ax.plot(t_ms, x.imag, color="#B85C38", linewidth=1.0, label="Q", alpha=0.9)
    ax.set_title("I/Q Waveform")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8.5, frameon=False)

    ax = axes[1]
    ax.plot(x.real, x.imag, color="#8A9A5B", alpha=0.35, linewidth=0.7)
    ax.scatter(x.real, x.imag, s=7, color="#1F6F8B", alpha=0.72, edgecolors="none")
    lim = float(np.max(np.abs([x.real, x.imag]))) * 1.08
    lim = max(lim, 1e-3)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Constellation")
    ax.set_xlabel("I")
    ax.set_ylabel("Q")
    ax.grid(True, alpha=0.25)

    ax = axes[2]
    ax.plot(freqs, spectrum_db, color="#1F6F8B", linewidth=1.1)
    ax.set_title("Spectrum")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB, normalized)")
    ax.set_ylim(-90, 3)
    ax.grid(True, alpha=0.25)

    ax = axes[3]
    mesh = ax.pcolormesh(
        stft_times * 1000.0,
        stft_freqs,
        np.clip(spectrogram_db, -90, 0),
        shading="auto",
        cmap="viridis",
        vmin=-90,
        vmax=0,
    )
    ax.set_title("Time-Frequency")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Frequency (Hz)")
    fig.colorbar(mesh, ax=ax, label="dB")

    for ax in axes:
        ax.tick_params(labelsize=8.5)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def build_index_list(args: argparse.Namespace, total: int) -> list[int]:
    indices: list[int] = []
    if args.indices:
        indices.extend(parse_indices(args.indices))
    if args.random:
        rng = np.random.default_rng(args.seed)
        count = min(args.random, total)
        indices.extend(rng.choice(total, size=count, replace=False).tolist())
    if args.all:
        indices.extend(range(total))
    if not indices:
        indices = list(range(min(args.first, total)))

    clean = []
    seen = set()
    for index in indices:
        if index < 0 or index >= total:
            raise IndexError(f"Sample index {index} is out of range 0..{total - 1}")
        if index not in seen:
            clean.append(index)
            seen.add(index)
    return clean


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize complex IQ samples as IQ, spectrum, and spectrogram plots.")
    parser.add_argument("--dataset-root", required=True, help="Folder containing X.npy and optional label files.")
    parser.add_argument("--out-dir", default=None, help="Output image folder. Default: dataset_root/visualizations")
    parser.add_argument("--fs", type=int, default=6000, help="Sample rate of IQ vectors.")
    parser.add_argument("--indices", default="", help="Comma-separated indices or slices, e.g. 0,5,10:20:2")
    parser.add_argument("--random", type=int, default=0, help="Randomly visualize this many samples.")
    parser.add_argument("--all", action="store_true", help="Visualize every sample in X.npy.")
    parser.add_argument("--first", type=int, default=12, help="Visualize first N samples when no index/random is given.")
    parser.add_argument("--seed", type=int, default=20260608, help="Random seed for --random.")
    parser.add_argument("--fft-size", type=int, default=2048, help="FFT size for spectrum and STFT.")
    parser.add_argument("--spectrogram-nperseg", type=int, default=256, help="STFT window length.")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    out_dir = Path(args.out_dir) if args.out_dir else dataset_root / "visualizations"

    x_path = dataset_root / "X.npy"
    if not x_path.exists():
        raise FileNotFoundError(f"Cannot find {x_path}")

    X = np.load(x_path, mmap_mode="r")
    y = np.load(dataset_root / "y.npy", mmap_mode="r") if (dataset_root / "y.npy").exists() else None
    snr_db = np.load(dataset_root / "snr_db.npy", mmap_mode="r") if (dataset_root / "snr_db.npy").exists() else None
    channel_profile = (
        np.load(dataset_root / "channel_profile.npy", mmap_mode="r")
        if (dataset_root / "channel_profile.npy").exists()
        else None
    )
    mode_names = load_json_list(dataset_root / "mode_names.json")
    channel_profiles = load_json_list(dataset_root / "channel_profiles.json")

    indices = build_index_list(args, len(X))
    print(f"Visualizing {len(indices)} sample(s) from {x_path}")

    for index in indices:
        title = make_sample_title(index, y, snr_db, channel_profile, mode_names, channel_profiles)
        safe_title = f"sample_{index:06d}.png"
        out_path = out_dir / safe_title
        plot_iq_sample(X[index], args.fs, title, out_path, args.fft_size, args.spectrogram_nperseg)
        print(out_path)

    print("Done.")


if __name__ == "__main__":
    main()
