"""
Build a Panoradio-like RF classification dataset from Fldigi-recorded WAV files.

Input layout:
    E:/仿真/raw_wav/BPSK31/BPSK31.wav
    E:/仿真/raw_wav/BPSK63/BPSK63.wav
    ...

Output layout:
    dataset_root/
      X.npy              complex64, shape [N, 2048]
      y.npy              int64, mode index
      snr_db.npy          int16, SNR label
      mode_names.json
      meta.json

The WAV files are real-valued audio passband recordings. This script converts
them to complex baseband IQ by creating an analytic signal and mixing the
configured audio carrier down to 0 Hz.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap
from scipy import signal
from scipy.io import wavfile


DEFAULT_SNRS = [25, 20, 15, 10, 5, 0, -5, -10]

CCIR520_PROFILES = {
    "flat_0p2": {"paths": 1, "delay_ms": 0.0, "spread_hz": 0.2, "offset_hz": 0.0},
    "flat_1": {"paths": 1, "delay_ms": 0.0, "spread_hz": 1.0, "offset_hz": 0.0},
    "good": {"paths": 2, "delay_ms": 0.5, "spread_hz": 0.1, "offset_hz": 0.0},
    "moderate": {"paths": 2, "delay_ms": 1.0, "spread_hz": 0.5, "offset_hz": 0.0},
    "poor": {"paths": 2, "delay_ms": 2.0, "spread_hz": 1.0, "offset_hz": 0.0},
    "flutter": {"paths": 2, "delay_ms": 0.5, "spread_hz": 10.0, "offset_hz": 0.0},
    "doppler": {"paths": 2, "delay_ms": 0.5, "spread_hz": 0.2, "offset_hz": "random_0_10"},
}

CCIR520_RANDOM_PROFILES = ["good", "moderate", "poor", "flutter"]


def parse_snrs(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def read_mono_wav(path: Path) -> tuple[np.ndarray, int]:
    fs, audio = wavfile.read(path)
    audio = np.asarray(audio)
    if audio.ndim == 2:
        audio = audio.astype(np.float32).mean(axis=1)

    if np.issubdtype(audio.dtype, np.integer):
        max_abs = float(np.iinfo(audio.dtype).max)
        audio = audio.astype(np.float32) / max_abs
    else:
        audio = audio.astype(np.float32)

    audio = audio - float(np.mean(audio))
    return audio, int(fs)


def trim_silence(audio: np.ndarray, fs: int, frame_ms: float = 250.0) -> np.ndarray:
    frame = max(1, int(fs * frame_ms / 1000.0))
    power = np.convolve(audio * audio, np.ones(frame, dtype=np.float32) / frame, mode="same")
    rms = np.sqrt(power)

    if not np.any(rms > 0):
        return audio

    threshold = max(1e-5, float(np.percentile(rms, 95)) * 0.03)
    active = np.flatnonzero(rms > threshold)
    if active.size == 0:
        return audio

    pad = int(0.25 * fs)
    start = max(0, int(active[0]) - pad)
    stop = min(len(audio), int(active[-1]) + pad)
    return audio[start:stop]


def audio_passband_to_baseband(
    audio: np.ndarray,
    fs_in: int,
    fs_out: int,
    carrier_hz: float,
) -> np.ndarray:
    analytic = signal.hilbert(audio).astype(np.complex64)
    t = np.arange(len(analytic), dtype=np.float64) / fs_in
    baseband = analytic * np.exp(-1j * 2.0 * np.pi * carrier_hz * t)

    gcd = math.gcd(fs_in, fs_out)
    up = fs_out // gcd
    down = fs_in // gcd
    baseband = signal.resample_poly(baseband, up, down).astype(np.complex64)
    baseband = baseband - np.mean(baseband)
    return baseband.astype(np.complex64)


def normalize_power(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    power = float(np.mean(np.abs(x) ** 2))
    return (x / math.sqrt(max(power, eps))).astype(np.complex64)


def random_slice(iq: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if len(iq) >= n:
        start = int(rng.integers(0, len(iq) - n + 1))
        return iq[start : start + n].copy()

    repeats = int(math.ceil(n / max(1, len(iq))))
    tiled = np.tile(iq, repeats)
    return tiled[:n].copy()


def apply_random_phase_and_frequency(
    x: np.ndarray,
    fs: int,
    max_freq_offset_hz: float,
    rng: np.random.Generator,
) -> np.ndarray:
    n = np.arange(len(x), dtype=np.float32)
    freq_offset = float(rng.uniform(-max_freq_offset_hz, max_freq_offset_hz))
    phase_offset = float(rng.uniform(0.0, 2.0 * np.pi))
    rotator = np.exp(1j * (2.0 * np.pi * freq_offset * n / fs + phase_offset))
    return (x * rotator).astype(np.complex64)


def pathsim_gaussian_fir(spread_2sigma_hz: float, filter_fs: float, attenuation_db: float = 35.0) -> np.ndarray:
    sigma_hz = spread_2sigma_hz / 2.0
    edge = 10.0 ** (-attenuation_db / 20.0)
    half_span_sec = math.sqrt(-math.log(edge) / (2.0 * math.pi * math.pi * sigma_hz * sigma_hz))
    half_len = max(3, int(math.ceil(half_span_sec * filter_fs)))
    t = np.arange(-half_len, half_len + 1, dtype=np.float64) / filter_fs
    taps = np.exp(-2.0 * math.pi * math.pi * sigma_hz * sigma_hz * t * t)
    taps /= np.sum(taps)
    return taps.astype(np.float64)


def pathsim_scatter_tap(
    length: int,
    fs: int,
    spread_2sigma_hz: float,
    rng: np.random.Generator,
    freq_shift_hz: float = 0.0,
) -> np.ndarray:
    duration_sec = length / fs
    fade_fs = max(30.0 * spread_2sigma_hz, 3.0)
    low_length = int(math.ceil(duration_sec * fade_fs)) + 16
    white = (
        rng.standard_normal(low_length).astype(np.float64)
        + 1j * rng.standard_normal(low_length).astype(np.float64)
    )
    taps = pathsim_gaussian_fir(spread_2sigma_hz, fade_fs)
    faded_low = signal.fftconvolve(white, taps, mode="same")

    low_t = np.arange(low_length, dtype=np.float64) / fade_fs
    high_t = np.arange(length, dtype=np.float64) / fs
    faded = np.interp(high_t, low_t, faded_low.real) + 1j * np.interp(high_t, low_t, faded_low.imag)

    if freq_shift_hz:
        faded *= np.exp(1j * 2.0 * np.pi * freq_shift_hz * high_t)

    return normalize_power(faded.astype(np.complex64))


def apply_ccir520_watterson_channel(
    iq: np.ndarray,
    fs: int,
    profile_name: str,
    rng: np.random.Generator,
) -> np.ndarray:
    profile = CCIR520_PROFILES[profile_name]
    spread_hz = float(profile["spread_hz"])
    offset = profile["offset_hz"]
    if offset == "random_0_10":
        freq_shift_hz = float(rng.uniform(0.0, 10.0))
    else:
        freq_shift_hz = float(offset)

    h1 = pathsim_scatter_tap(len(iq), fs, spread_hz, rng, freq_shift_hz)
    y = h1 * iq

    if int(profile["paths"]) == 2:
        delay_samples = int(round(fs * float(profile["delay_ms"]) / 1000.0))
        h2 = pathsim_scatter_tap(len(iq), fs, spread_hz, rng, freq_shift_hz)
        delayed = np.zeros_like(iq)
        if delay_samples > 0:
            delayed[delay_samples:] = iq[:-delay_samples]
        else:
            delayed = iq.copy()
        y = (y + h2 * delayed) / math.sqrt(2.0)

    return normalize_power(y)


def resolve_ccir520_profiles(profile_arg: str) -> list[str]:
    if profile_arg == "random":
        return CCIR520_RANDOM_PROFILES.copy()
    if profile_arg == "all":
        return list(CCIR520_PROFILES)
    names = [name.strip().lower() for name in profile_arg.split(",") if name.strip()]
    unknown = [name for name in names if name not in CCIR520_PROFILES]
    if unknown:
        raise ValueError(f"Unknown CCIR 520 profile(s): {', '.join(unknown)}")
    return names


def add_awgn(x: np.ndarray, snr_db: int, rng: np.random.Generator) -> np.ndarray:
    signal_power = 1.0
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    sigma = math.sqrt(noise_power / 2.0)
    noise = sigma * (
        rng.standard_normal(len(x)).astype(np.float32)
        + 1j * rng.standard_normal(len(x)).astype(np.float32)
    )
    return (x + noise).astype(np.complex64)


def find_mode_wavs(raw_root: Path) -> list[tuple[str, Path]]:
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


def build_dataset(args: argparse.Namespace) -> None:
    raw_root = Path(args.raw_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    mode_wavs = find_mode_wavs(raw_root)
    if not mode_wavs:
        raise FileNotFoundError(f"No WAV files found under {raw_root}")

    snrs = parse_snrs(args.snrs)
    channel_profiles = resolve_ccir520_profiles(args.ccir520_profile) if args.channel == "ccir520" else ["none"]
    total = len(mode_wavs) * len(snrs) * args.vectors_per_mode_snr
    shape = (total, args.vector_len)

    x_path = out_root / "X.npy"
    y_path = out_root / "y.npy"
    snr_path = out_root / "snr_db.npy"
    channel_profile_path = out_root / "channel_profile.npy"

    print(f"Found {len(mode_wavs)} modes.")
    print(f"Writing {total:,} vectors to {out_root}")
    print(f"X shape: {shape}, dtype: complex64")

    X = open_memmap(x_path, mode="w+", dtype=np.complex64, shape=shape)
    y = open_memmap(y_path, mode="w+", dtype=np.int64, shape=(total,))
    snr_labels = open_memmap(snr_path, mode="w+", dtype=np.int16, shape=(total,))
    channel_profile_labels = open_memmap(channel_profile_path, mode="w+", dtype=np.int16, shape=(total,))

    rng = np.random.default_rng(args.seed)
    index = 0
    mode_names = []

    for mode_index, (mode_name, wav_path) in enumerate(mode_wavs):
        print(f"\n[{mode_index + 1}/{len(mode_wavs)}] {mode_name}: {wav_path}")
        mode_names.append(mode_name)

        audio, fs = read_mono_wav(wav_path)
        audio = trim_silence(audio, fs)
        iq = audio_passband_to_baseband(audio, fs, args.fs, args.carrier_hz)

        if len(iq) < args.vector_len:
            print(f"  Warning: short recording after preprocessing ({len(iq)} samples). It will be tiled.")

        if args.channel == "ccir520":
            channel_iq = {}
            for profile_name in channel_profiles:
                print(f"  Applying CCIR 520 Watterson profile: {profile_name}")
                channel_iq[profile_name] = apply_ccir520_watterson_channel(iq, args.fs, profile_name, rng)
        else:
            channel_iq = {"none": iq}

        for snr_db in snrs:
            for _ in range(args.vectors_per_mode_snr):
                profile_index = int(rng.integers(0, len(channel_profiles)))
                profile_name = channel_profiles[profile_index]
                x = random_slice(channel_iq[profile_name], args.vector_len, rng)
                x = normalize_power(x)

                x = apply_random_phase_and_frequency(x, args.fs, args.freq_offset_hz, rng)
                x = normalize_power(x)
                x = add_awgn(x, snr_db, rng)

                X[index] = x
                y[index] = mode_index
                snr_labels[index] = snr_db
                channel_profile_labels[index] = profile_index
                index += 1

            print(f"  SNR {snr_db:>3} dB done")

        X.flush()
        y.flush()
        snr_labels.flush()
        channel_profile_labels.flush()

    with (out_root / "mode_names.json").open("w", encoding="utf-8") as f:
        json.dump(mode_names, f, ensure_ascii=False, indent=2)
    with (out_root / "channel_profiles.json").open("w", encoding="utf-8") as f:
        json.dump(channel_profiles, f, ensure_ascii=False, indent=2)

    meta = {
        "dataset": "panoradio_like_from_fldigi_wav",
        "raw_root": str(raw_root),
        "num_vectors": total,
        "vector_len": args.vector_len,
        "fs_hz": args.fs,
        "duration_sec": args.vector_len / args.fs,
        "dtype": "complex64",
        "carrier_hz_removed": args.carrier_hz,
        "snr_db_values": snrs,
        "random_frequency_offset_hz": [-args.freq_offset_hz, args.freq_offset_hz],
        "random_phase_offset": True,
        "signal_power_normalized_before_awgn": True,
        "channel": args.channel,
        "ccir520_profile_arg": args.ccir520_profile if args.channel == "ccir520" else None,
        "ccir520_profiles": channel_profiles if args.channel == "ccir520" else None,
        "ccir520_profile_definitions": {name: CCIR520_PROFILES[name] for name in channel_profiles if name != "none"},
        "ccir520_frequency_spread_interpretation": "2_sigma_hz",
        "ccir520_fading_tap_implementation": "pathsim_style_complex_gaussian_noise_filtered_by_gaussian_fir",
        "ccir520_fading_filter_rate": "max(30 * frequency_spread_2sigma_hz, 3 Hz)",
        "pathsim_reference": "https://github.com/bubnikv/pathsim",
        "ccir520_equal_average_path_power": True if args.channel == "ccir520" else None,
        "vectors_per_mode_snr": args.vectors_per_mode_snr,
        "mode_names": mode_names,
    }
    with (out_root / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("\nDone.")
    print(f"X:        {x_path}")
    print(f"y:        {y_path}")
    print(f"snr_db:   {snr_path}")
    print(f"channel:  {channel_profile_path}")
    print(f"metadata: {out_root / 'meta.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Panoradio-like complex IQ dataset from WAV files.")
    parser.add_argument("--raw-root", default=r"E:\仿真\raw_wav", help="Folder containing one subfolder per mode.")
    parser.add_argument("--out-root", default=r"E:\仿真\panoradio_like_npy", help="Output dataset folder.")
    parser.add_argument("--fs", type=int, default=6000, help="Output IQ sample rate.")
    parser.add_argument("--vector-len", type=int, default=2048, help="IQ samples per vector.")
    parser.add_argument("--carrier-hz", type=float, default=1500.0, help="Audio carrier frequency used by Fldigi.")
    parser.add_argument("--freq-offset-hz", type=float, default=250.0, help="Maximum random frequency offset.")
    parser.add_argument("--snrs", default=",".join(map(str, DEFAULT_SNRS)), help="Comma-separated SNR dB values.")
    parser.add_argument("--vectors-per-mode-snr", type=int, default=1200, help="Vectors per mode per SNR.")
    parser.add_argument("--seed", type=int, default=20260607, help="Random seed.")
    parser.add_argument("--channel", choices=["ccir520", "none"], default="ccir520", help="Channel model.")
    parser.add_argument(
        "--ccir520-profile",
        default="random",
        help="CCIR 520 profile: flat_0p2, flat_1, good, moderate, poor, flutter, doppler, random, all, or comma list.",
    )
    parser.add_argument("--no-fading", action="store_const", const="none", dest="channel", help="Disable fading channel.")

    args = parser.parse_args()
    build_dataset(args)


if __name__ == "__main__":
    main()
