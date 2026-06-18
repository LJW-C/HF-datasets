# Data Set and Analysis Tools

## Data Set Description

This data set provides labeled complex baseband I/Q samples for machine-learning-based high-frequency (HF) digital-mode signal classification. The data are generated from Fldigi WAV recordings and converted into fixed-length complex I/Q vectors for controlled signal-recognition experiments.

The released data set contains 163,200 signal samples from 17 fine-grained HF digital modes. Each sample consists of 2048 complex I/Q samples at a sampling rate of 6 kHz, corresponding to approximately 0.341 s of signal duration. The data set includes eight SNR levels: 25, 20, 15, 10, 5, 0, -5, and -10 dB. Each sample is also associated with a Watterson-style HF fading channel profile.

The main data file is `X.npy`, which stores the complex I/Q vectors with shape `[N, 2048]` and dtype `complex64`. The primary signal labels are stored in `y.npy`, and the ordered label names are stored in `mode_names.json`. Additional labels are provided in `snr_db.npy` and `channel_profile.npy` for per-SNR and per-channel-condition evaluation.

The data set supports both fine-grained and hierarchical classification. The primary task is 17-class digital-mode classification. The 17 modes can also be grouped into five coarse signal families: PSK, Olivia/Contestia, MFSK-like modes, RTTY/FSK, and CW. This makes the data set suitable for fine-grained classification, coarse-family classification, and coarse-to-fine hierarchical recognition experiments.

The data set is synthetic and audio-derived. It should not be interpreted as direct over-the-air RF receiver recordings. The signals were first generated as real-valued Fldigi audio waveforms, then converted to analytic complex baseband I/Q, resampled to 6 kHz, impaired with Watterson-style HF fading, randomized frequency and phase offsets, and complex AWGN.

## Data Files

| File | Description |
|---|---|
| `X.npy` | Complex baseband I/Q samples, shape `[N, 2048]`, dtype `complex64`. |
| `y.npy` | Fine-grained signal-mode labels. Values index into `mode_names.json`. |
| `snr_db.npy` | SNR label in dB for each sample. |
| `channel_profile.npy` | HF fading channel-profile label for each sample. |
| `mode_names.json` | Ordered list of 17 digital-mode class names. |
| `channel_profiles.json` | Ordered list of channel-profile names. |
| `meta.json` | Metadata describing generation parameters and data format. |
| `family_label.npy` | Optional coarse signal-family labels, if generated. |
| `family_names.json` | Optional ordered list of coarse family names. |

## Analysis Tools

The accompanying analysis tools are provided as Python scripts. They are intended to make the data set reproducible, inspectable, and usable for machine-learning experiments.

| Tool | Purpose |
|---|---|
| `generate_fldigi_wavs.py` | Automatically generates and records raw WAV files from Fldigi through XML-RPC control and a virtual audio cable. |
| `validate_raw_wavs.py` | Checks raw WAV recordings for silence, abnormal bandwidth, and incorrect mode configuration. |
| `build_panoradio_like_dataset.py` | Converts raw WAV files into complex baseband I/Q data, applies HF fading, frequency/phase offsets, and AWGN, then saves `.npy` data files. |
| `visualize_iq_samples.py` | Generates I/Q waveform plots, constellation plots, FFT spectra, and spectrograms for selected samples. |
| `diagnose_mode_visuals.py` | Produces additional diagnostic plots for checking raw WAV and generated I/Q samples. |
| `train_rf_classifier.py` | Trains and evaluates one radio-signal classification model. |
| `run_model_suite.py` | Runs multiple baseline models for data set validation. |
| `summarize_model_results.py` | Summarizes model accuracy, macro-F1, balanced accuracy, and per-SNR results. |
| `make_paper_results.py` | Generates result figures and tables for paper/report use, including confusion matrices and SNR accuracy curves. |

## Basic Usage

Load the data set using Python and NumPy:

```python
from pathlib import Path
import json
import numpy as np

dataset_root = Path("path/to/dataset")

X = np.load(dataset_root / "X.npy", mmap_mode="r")
y = np.load(dataset_root / "y.npy")
snr_db = np.load(dataset_root / "snr_db.npy")
channel_profile = np.load(dataset_root / "channel_profile.npy")

mode_names = json.loads((dataset_root / "mode_names.json").read_text(encoding="utf-8"))
channel_profiles = json.loads((dataset_root / "channel_profiles.json").read_text(encoding="utf-8"))
```

For deep-learning models, each complex I/Q sample can be converted into a two-channel real tensor:

```python
iq = X[0]
sample = np.stack([iq.real, iq.imag], axis=0).astype("float32")
```

Use `y.npy` as the main classification target. Use `snr_db.npy` for per-SNR evaluation and `channel_profile.npy` for channel-profile analysis.

## Recommended Evaluation

Recommended evaluation metrics include overall accuracy, macro-F1 score, balanced accuracy, per-SNR accuracy, and confusion matrix analysis. A stratified train/validation/test split should preserve the distributions of signal mode and SNR. For stricter evaluation, users may also perform channel-profile-aware or raw-recording-aware splitting.

Because multiple I/Q vectors are randomly cropped from longer WAV recordings, users should clearly report the split strategy used in experiments. This is important for distinguishing controlled augmentation experiments from fully independent recording-based evaluation.

## Suggested IEEE DataPort Text

This data set contains labeled complex baseband I/Q samples for HF digital-mode signal classification. The data set includes 163,200 samples from 17 fine-grained digital modes, eight SNR levels, Watterson-style HF fading channel profiles, random frequency offsets, random phase offsets, and complex AWGN. Each sample contains 2048 complex I/Q samples at 6 kHz and is stored in NumPy format. The main files are `X.npy` for I/Q samples, `y.npy` for fine-grained mode labels, `snr_db.npy` for SNR labels, `channel_profile.npy` for channel labels, and JSON metadata files for label names and generation settings.

The accompanying analysis tools include scripts for Fldigi WAV recording, raw WAV validation, WAV-to-IQ data set generation, I/Q visualization, diagnostic plotting, model training, benchmark evaluation, and result summarization. These tools allow users to reproduce the data generation pipeline, inspect sample quality, train baseline classifiers, and analyze model performance across signal classes, SNR levels, and channel conditions.

The data set is synthetic and audio-derived. It is suitable for controlled HF digital-mode recognition experiments, but it should not be interpreted as direct over-the-air RF receiver data.

