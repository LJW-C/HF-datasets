# Dataset Documentation for IEEE DataPort

## Dataset Title

HF Digital-Mode IQ Dataset Generated from Fldigi WAV Recordings with Watterson-Style HF Channel Impairments

## Dataset Overview

This dataset provides labeled complex baseband in-phase/quadrature (I/Q) samples for machine-learning-based high-frequency (HF) digital-mode signal classification. The data are generated from Fldigi audio WAV recordings and converted into Panoradio-like fixed-length complex I/Q vectors.

The dataset is intended for controlled experiments on HF digital-mode recognition under receiver and channel impairments. It includes multiple digital modes, multiple SNR levels, randomized frequency and phase offsets, and Watterson-style HF fading channel profiles based on CCIR/F.520/F.1487 parameter settings.

The dataset is synthetic and audio-derived. The I/Q samples are not direct over-the-air RF captures from a hardware receiver. They are analytic complex baseband representations derived from real-valued Fldigi modem audio recordings and then impaired using a configurable HF channel simulation pipeline.

## Dataset Summary

| Item | Value |
|---|---|
| Number of samples | 163,200 |
| Number of fine-grained signal classes | 17 |
| I/Q vector length | 2048 complex samples |
| Sampling rate | 6000 Hz |
| Duration per sample | 2048 / 6000 = 0.3413 s |
| Data format | NumPy `.npy` files |
| I/Q dtype | `complex64` |
| Label dtype | `int64` / `int16` |
| SNR values | 25, 20, 15, 10, 5, 0, -5, -10 dB |
| Channel model | Watterson-style HF fading profiles |
| Frequency offset | Uniform random offset in approximately +/-250 Hz |
| Phase offset | Uniform random phase in [0, 2*pi) |

The default dataset size is calculated as:

```text
17 modes x 8 SNR values x 1200 samples per mode-SNR pair = 163200 samples
```

## Directory Structure

The released dataset directory should contain the following files:

```text
dataset_root/
|-- X.npy
|-- y.npy
|-- snr_db.npy
|-- channel_profile.npy
|-- mode_names.json
|-- channel_profiles.json
|-- meta.json
```

Optional hierarchical-label files may also be included:

```text
|-- family_label.npy
|-- family_names.json
```

## File Dictionary

| File | Type / Shape | Description |
|---|---|---|
| `X.npy` | `complex64`, shape `[N, 2048]` | Complex baseband I/Q vectors. Each row is one signal sample. |
| `y.npy` | integer, shape `[N]` | Fine-grained signal-mode label for each sample. Values are indices into `mode_names.json`. |
| `snr_db.npy` | integer, shape `[N]` | SNR label in dB for each sample. |
| `channel_profile.npy` | integer, shape `[N]` | Channel-profile label for each sample. Values are indices into `channel_profiles.json`. |
| `mode_names.json` | JSON list | Ordered list of fine-grained signal-mode names. |
| `channel_profiles.json` | JSON list | Ordered list of channel-profile names. |
| `meta.json` | JSON object | Metadata describing generation parameters, signal length, sampling rate, SNR values, and channel settings. |
| `family_label.npy` | optional integer array, shape `[N]` | Optional coarse signal-family label for hierarchical classification. |
| `family_names.json` | optional JSON list | Optional ordered list of coarse signal-family names. |

Here, `N` is the total number of generated samples. In the default release, `N = 163200`.

## Fine-Grained Signal Classes

The primary classification target is the fine-grained mode label stored in `y.npy`.

| Label ID | Signal mode |
|---:|---|
| 0 | `BPSK125` |
| 1 | `BPSK31` |
| 2 | `BPSK63` |
| 3 | `Contestia_4_250` |
| 4 | `Contestia_8_500` |
| 5 | `CW` |
| 6 | `DominoEX_8` |
| 7 | `MFSK16` |
| 8 | `MFSK32` |
| 9 | `Olivia_4_250` |
| 10 | `Olivia_8_500` |
| 11 | `QPSK125` |
| 12 | `QPSK31` |
| 13 | `QPSK63` |
| 14 | `RTTY` |
| 15 | `Thor_16` |
| 16 | `Throb_2` |

## Coarse Signal Families

The 17 fine-grained classes can also be grouped into five coarse families for hierarchical classification and error analysis.

| Coarse family | Fine-grained modes |
|---|---|
| `PSK` | `BPSK31`, `BPSK63`, `BPSK125`, `QPSK31`, `QPSK63`, `QPSK125` |
| `Olivia_Contestia` | `Olivia_4_250`, `Olivia_8_500`, `Contestia_4_250`, `Contestia_8_500` |
| `MFSK_like` | `MFSK16`, `MFSK32`, `DominoEX_8`, `Thor_16`, `Throb_2` |
| `RTTY_FSK` | `RTTY` |
| `CW` | `CW` |

The coarse family labels are optional. They can be derived deterministically from `mode_names.json` and `y.npy`.

## Signal Generation Pipeline

The dataset was generated using the following processing chain:

1. Generate digital-mode audio signals using Fldigi.
2. Record one WAV file per signal mode through a virtual audio cable.
3. Load the real-valued audio waveform and convert stereo recordings to mono if necessary.
4. Remove DC offset and trim inactive leading/trailing regions.
5. Construct an analytic signal using the Hilbert transform.
6. Shift the configured Fldigi audio carrier, usually 1500 Hz, to complex baseband.
7. Resample the complex baseband signal to 6000 Hz.
8. Apply a Watterson-style HF fading channel.
9. Randomly crop fixed-length vectors of 2048 complex samples.
10. Apply random carrier frequency offset and random phase offset.
11. Normalize signal power and add complex AWGN at the target SNR.
12. Save I/Q vectors, labels, and metadata as NumPy and JSON files.

## Channel and Impairment Labels

The dataset includes two non-class labels:

| Label | Meaning |
|---|---|
| `snr_db.npy` | The SNR value used when AWGN was added. |
| `channel_profile.npy` | The fading-channel profile used for the sample. |

These labels describe the channel condition and receiver impairment, not the signal class. They can be used for per-SNR evaluation, channel-profile evaluation, or domain-generalization studies.

The channel-profile names are stored in `channel_profiles.json`. Typical profiles include flat fading, good/moderate/poor two-path fading, flutter fading, and Doppler-shifted fading conditions.

## Loading the Dataset in Python

Example loading code:

```python
from pathlib import Path
import json
import numpy as np

dataset_root = Path("path/to/dataset_root")

X = np.load(dataset_root / "X.npy", mmap_mode="r")
y = np.load(dataset_root / "y.npy")
snr_db = np.load(dataset_root / "snr_db.npy")
channel_profile = np.load(dataset_root / "channel_profile.npy")

mode_names = json.loads((dataset_root / "mode_names.json").read_text(encoding="utf-8"))
channel_profiles = json.loads((dataset_root / "channel_profiles.json").read_text(encoding="utf-8"))

print(X.shape)          # Example: (163200, 2048)
print(X.dtype)          # complex64
print(mode_names[y[0]]) # Fine-grained signal label of the first sample
print(snr_db[0])        # SNR label of the first sample
```

Many neural-network libraries use real-valued tensors. A complex I/Q vector can be converted to a two-channel real tensor:

```python
iq = X[0]
sample = np.stack([iq.real, iq.imag], axis=0).astype("float32")
print(sample.shape)  # (2, 2048)
```

## Recommended Benchmark Usage

For standard fine-grained classification, use:

```text
input:  X.npy
target: y.npy
classes: mode_names.json
```

For per-SNR evaluation, group test predictions by:

```text
snr_db.npy
```

For channel-profile evaluation, group test predictions by:

```text
channel_profile.npy
```

Recommended evaluation metrics include:

| Metric | Purpose |
|---|---|
| Overall accuracy | General classification performance |
| Macro-F1 score | Class-balanced performance |
| Balanced accuracy | Average recall over classes |
| Per-SNR accuracy | Robustness under different noise levels |
| Confusion matrix | Identification of confusing signal modes |

## Recommended Data Splitting

A stratified train/validation/test split should preserve the distributions of signal mode and SNR. For stricter evaluation, users should also consider channel-profile-aware or raw-recording-aware splitting.

Because multiple vectors are randomly cropped from longer WAV recordings, some samples may share nearby or partially overlapping raw time intervals. Therefore, a split performed only after sample generation may be useful for controlled classification experiments, but it may not represent a fully independent over-the-air recording split.

When reporting benchmark results, users should clearly state the split strategy.

## Important Limitations

1. The dataset is synthetic and audio-derived. It is not a direct RF recording dataset.
2. The I/Q vectors are analytic baseband representations created from real-valued Fldigi audio recordings.
3. The fading process is a controllable Watterson-style channel simulation and cannot reproduce every real HF propagation effect.
4. The dataset does not include uncontrolled real-world interference, receiver nonlinearity, antenna effects, or adjacent-channel emissions unless users add them separately.
5. The default vector length is 2048 samples, about 0.341 s at 6000 Hz. Some slow or highly structured modes may require longer context for best recognition accuracy.

## Citation Guidance

When using this dataset, please cite the IEEE DataPort dataset record and any related paper or repository associated with the dataset release. If the dataset generation method is discussed, users should also mention that the I/Q samples are generated from Fldigi WAV recordings and impaired using Watterson-style HF channel simulation, random frequency/phase offsets, and AWGN.

## Suggested Short Description

This dataset contains 163,200 labeled complex I/Q samples for HF digital-mode signal classification. It includes 17 fine-grained digital modes, eight SNR levels, Watterson-style HF fading profiles, random frequency/phase offsets, and JSON metadata. The samples are generated from Fldigi WAV recordings and converted to 2048-sample complex baseband vectors at 6 kHz.

