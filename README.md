# HF Digital-Mode IQ Dataset

This repository contains code and documentation for generating a Panoradio-like
HF digital-mode IQ dataset from Fldigi WAV recordings.

The pipeline records standard digital-mode audio signals with Fldigi, converts
the real-valued WAV recordings to complex baseband IQ, applies a Watterson-style
HF fading channel, adds frequency/phase offsets and AWGN, and exports NumPy
dataset files for machine-learning experiments.

## Contents

```text
generate_fldigi_wavs.py              # Record raw WAV files from Fldigi
validate_raw_wavs.py                 # Validate raw WAV bandwidth and silence
build_datasets.py      # Convert WAV files to IQ dataset
visualize_iq_samples.py              # I/Q, constellation, spectrum, spectrogram plots
diagnose_mode_visuals.py             # Extra raw-WAV/IQ diagnostic plots
train_rf_classifier.py               # Train one RF classifier
run_model_suite.py                   # Train several baseline models
summarize_model_results.py           # Summarize benchmark outputs
make_paper_results.py                # Generate paper result figures and tables
rfml_benchmark/                      # Dataset loader and model definitions
docs/                                # Workflow documentation
paper/                               # IEEE-style LaTeX paper and result figures
data/                                # Dataset manifest and release instructions
scripts/                             # Dataset release/upload helper scripts
```

## Dataset Summary

Default fixed dataset:

```text
samples:      163,200
classes:      17
vector length:2048 complex IQ samples
sample rate:  6 kHz
duration:     0.341 s per vector
SNR values:   25, 20, 15, 10, 5, 0, -5, -10 dB
channel:      Watterson-style CCIR/F.520/F.1487 HF profiles
format:       NumPy .npy files
```

Dataset files:

```text
X.npy                  complex64, shape [163200, 2048]
y.npy                  int64 labels
snr_db.npy             int16 SNR labels
channel_profile.npy    int16 channel profile labels
mode_names.json        class names
channel_profiles.json  channel profile names
meta.json              generation metadata
```

The 17 fine-grained signal labels can also be organized into five coarse
families for hierarchical experiments. See:

```text
docs/HIERARCHICAL_LABELS.md
```

For an IEEE DataPort upload, use the documentation and instruction text in:

```text
docs/IEEE_DATAPORT_DATASET_DOCUMENTATION.md
docs/IEEE_DATAPORT_DATASET_AND_ANALYSIS_TOOLS.md
docs/IEEE_DATAPORT_INSTRUCTIONS.md
docs/IEEE_DATAPORT_USAGE_INSTRUCTIONS.txt
```

The dataset manifest and checksums are in:

```text
data/panoradio_like_ccir520_npy_fixed/dataset_manifest.json
```

The large binary dataset is not committed directly to normal Git history. Use
Git LFS or a GitHub Release asset. See:

```text
data/README.md
docs/GITHUB_UPLOAD.md
```

## Installation

```powershell
pip install -r requirements.txt
```

For GPU training, install the PyTorch build that matches your CUDA version.

## Generate Raw WAV Files

Start Fldigi, enable XML-RPC, and route the Fldigi audio output to a virtual
audio cable. Then run:

```powershell
python generate_fldigi_wavs.py `
  --device-id 7 `
  --out-root "E:\仿真\raw_wav" `
  --duration-sec 180 `
  --record-fs 8000 `
  --carrier-hz 1500
```

Validate the raw recordings:

```powershell
python validate_raw_wavs.py `
  --raw-root "E:\仿真\raw_wav" `
  --out-dir diagnostics\raw_wav_validation
```

## Build the IQ Dataset

```powershell
python build_datasets.py `
  --raw-root "E:\仿真\raw_wav" `
  --out-root "E:\仿真\panoradio_like_ccir520_npy_fixed" `
  --fs 6000 `
  --vector-len 2048 `
  --carrier-hz 1500 `
  --freq-offset-hz 250 `
  --snrs 25,20,15,10,5,0,-5,-10 `
  --vectors-per-mode-snr 1200 `
  --seed 20260614 `
  --channel ccir520 `
  --ccir520-profile random
```

## Visualize Samples

```powershell
python visualize_iq_samples.py `
  --dataset-root "E:\仿真\panoradio_like_ccir520_npy_fixed" `
  --random 30 `
  --out-dir "E:\仿真\panoradio_like_ccir520_npy_fixed\visualizations"
```

Each figure includes an I/Q waveform, constellation/trajectory, spectrum, and
time-frequency spectrogram.

## Train Baseline Models

```powershell
python run_model_suite.py `
  --dataset-root "E:\仿真\panoradio_like_ccir520_npy_fixed" `
  --models iq_cnn,resnet1d,tcn,cldnn,transformer,conformer `
  --epochs 30 `
  --batch-size 128 `
  --out-dir runs/model_suite_fixed
```

Summarize results:

```powershell
python summarize_model_results.py --runs-dir runs/model_suite_fixed
```

Generate paper figures and tables:

```powershell
python make_paper_results.py `
  --runs-dir runs/model_suite_fixed `
  --paper-dir paper
```

## License

Code is released under the MIT License. Check the licenses of Fldigi, external
libraries, and radio standards documents separately.
