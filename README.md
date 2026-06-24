# HF Digital-Mode I/Q Dataset

This repository contains code and documentation for generating an HF digital-mode I/Q dataset from Fldigi WAV recordings.

The pipeline records digital-mode audio signals with Fldigi, converts real-valued WAV recordings to complex baseband I/Q samples, applies a Watterson-style HF fading channel, adds carrier-frequency offset, phase offset, and AWGN, and exports NumPy dataset files for machine-learning experiments.

## Contents

```text
generate_fldigi_wavs.py              # Record raw WAV files from Fldigi
validate_raw_wavs.py                 # Validate raw WAV bandwidth and silence
build_datasets.py                    # Convert WAV recordings to an I/Q dataset
visualize_iq_samples.py              # I/Q, constellation, spectrum, and spectrogram plots
diagnose_mode_visuals.py             # Additional raw-WAV and I/Q diagnostic plots
train_rf_classifier.py               # Train one RF classifier
run_model_suite.py                   # Train several baseline models
summarize_model_results.py           # Summarize benchmark outputs
make_paper_results.py                # Generate paper figures and tables
rfml_benchmark/                      # Dataset loader and model definitions
docs/                                # Workflow documentation
data/                                # Dataset manifest and release instructions
scripts/                             # Dataset release and upload helper scripts
```

## 1. Install the required software

Install the following software before recording WAV files.

### Fldigi 4.2.11

Download and install Fldigi:

```text
https://www.w1hkj.org/files/fldigi/fldigi-4.2.11_setup.exe
```

### VB-CABLE Virtual Audio Device

Download VB-CABLE:

```text
https://vb-audio.com/Cable/
```

Extract the downloaded package, run the setup program as Administrator, and restart Windows after installation.

VB-CABLE creates two virtual audio devices:

```text
CABLE Input   playback device
CABLE Output  recording device
```

### Python dependencies

Install the required Python packages:

```powershell
pip install -r requirements.txt
```

For GPU training, install the PyTorch build matching the installed CUDA version.

## 2. Configure the virtual audio devices

Open the Windows sound-device panel:

```text
Win + R
mmsys.cpl
```

Confirm that the following devices are available:

```text
Playback tab:  CABLE Input (VB-Audio Virtual Cable)
Recording tab: CABLE Output (VB-Audio Virtual Cable)
```

Do not set VB-CABLE as the default Windows playback device, otherwise system sounds may be captured in the generated WAV files.

The audio-routing path is:

```text
Fldigi transmission
    -> CABLE Input
    -> VB-CABLE internal routing
    -> CABLE Output
    -> Python recording script
    -> WAV file
```

List the available recording devices:

```powershell
python generate_fldigi_wavs.py --list-devices
```

Find the device corresponding to:

```text
CABLE Output (VB-Audio Virtual Cable)
```

Use the number shown before the device name as the value of `--device-id`.

For example:

```text
7  CABLE Output (VB-Audio Virtual Cable)
```

Use:

```powershell
--device-id 7
```

The device id is machine-dependent. Do not reuse the device id from another computer without checking it first.

## 3. Configure Fldigi

Open Fldigi and navigate to:

```text
Configure -> Config Dialog -> Sound Card
```

On Windows, select the PortAudio driver and set:

```text
Playback device: CABLE Input (VB-Audio Virtual Cable)
Capture device:  any valid unused input device
```

The Fldigi Capture device is not used in this automated recording workflow because Python records directly from `CABLE Output`.

Use a default carrier frequency of 1500 Hz. The recording script also sets the carrier through XML-RPC.

Disable real transceiver, PTT, and rig-control connections unless they are required for another purpose. This workflow only needs Fldigi to generate audio and route it to VB-CABLE.

Start Fldigi with XML-RPC enabled:

```powershell
"C:\Program Files\fldigi-4.2.11\fldigi.exe" `
  --xmlrpc-server-address 127.0.0.1 `
  --xmlrpc-server-port 7362
```

The recording script connects to:

```text
http://127.0.0.1:7362
```

## 4. Generate raw WAV recordings

The script controls Fldigi through XML-RPC to select the modem mode, set the carrier frequency, insert random text payloads, and switch between TX and RX. Python records the generated audio from `CABLE Output` and saves one WAV file for each selected mode.

Generate WAV recordings for all default benchmark modes:

```powershell
python generate_fldigi_wavs.py `
  --device-id 7 `
  --out-root "E:\仿真\raw_wav" `
  --duration-sec 180 `
  --record-fs 8000 `
  --carrier-hz 1500
```

Generate recordings for selected modes only:

```powershell
python generate_fldigi_wavs.py `
  --device-id 7 `
  --out-root "E:\仿真\raw_wav" `
  --duration-sec 180 `
  --record-fs 8000 `
  --carrier-hz 1500 `
  --modes BPSK31,BPSK63,QPSK31,RTTY
```

The output layout is:

```text
raw_wav/
├── BPSK31/
│   └── BPSK31.wav
├── BPSK63/
│   └── BPSK63.wav
├── RTTY/
│   └── RTTY.wav
└── raw_wav_metadata.json
```

After recording each WAV file, the script reports RMS and peak amplitude values. Very low RMS or peak values usually indicate silence or incorrect audio routing.

## 5. Validate raw WAV recordings

Validate silence, duration, and bandwidth properties before dataset generation:

```powershell
python validate_raw_wavs.py `
  --raw-root "E:\仿真\raw_wav" `
  --out-dir diagnostics\raw_wav_validation
```

If the recordings are silent, check:

```text
Fldigi Playback device
VB-CABLE routing
Python recording device id
Windows audio permissions
Fldigi TX state
```

## 6. Build the I/Q dataset

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

The dataset-generation pipeline performs:

```text
WAV preprocessing
    -> DC removal and energy trimming
    -> Hilbert-transform analytic signal generation
    -> 1500 Hz baseband down-mixing
    -> resampling to 6 kHz
    -> Watterson-style HF fading
    -> random cropping to 2048 complex samples
    -> carrier-frequency and phase offsets
    -> AWGN injection
    -> NumPy dataset export
```

## 7. Default benchmark dataset

The default benchmark configuration contains:

```text
Samples:        163,200
Classes:        17
Vector length:  2048 complex I/Q samples
Sample rate:    6 kHz
Duration:       approximately 0.341 s per vector
SNR values:     25, 20, 15, 10, 5, 0, -5, -10 dB
Channel model:  Watterson-style CCIR/F.520/F.1487 HF profiles
Format:         NumPy .npy files
```

Dataset files:

```text
X.npy                  complex64, shape [163200, 2048]
y.npy                  int64 mode labels
snr_db.npy             int16 SNR labels
channel_profile.npy    int16 channel-profile labels
mode_names.json        ordered class names
channel_profiles.json  ordered channel-profile names
meta.json              generation metadata
```

The default 17 fine-grained benchmark labels are:

```text
BPSK31
BPSK63
BPSK125
QPSK31
QPSK63
QPSK125
RTTY
MFSK16
MFSK32
Olivia_4_250
Olivia_8_500
Contestia_4_250
Contestia_8_500
DominoEX_8
Thor_16
Throb_2
CW
```

For hierarchical experiments, these labels can also be grouped into broader families:

```text
docs/HIERARCHICAL_LABELS.md
```

## 8. Visualize I/Q samples

```powershell
python visualize_iq_samples.py `
  --dataset-root "E:\仿真\panoradio_like_ccir520_npy_fixed" `
  --random 30 `
  --out-dir "E:\仿真\panoradio_like_ccir520_npy_fixed\visualizations"
```

Each figure includes an I/Q waveform, constellation plot, spectrum, and time-frequency spectrogram.

## 9. Train baseline models

```powershell
python run_model_suite.py `
  --dataset-root "E:\仿真\panoradio_like_ccir520_npy_fixed" `
  --models iq_cnn,resnet1d,tcn,cldnn,transformer,conformer `
  --epochs 30 `
  --batch-size 128 `
  --out-dir runs/model_suite_fixed
```

Summarize the model results:

```powershell
python summarize_model_results.py `
  --runs-dir runs/model_suite_fixed
```

## 10. Dataset release files

The dataset manifest and checksum information are stored in:

```text
data/panoradio_like_ccir520_npy_fixed/dataset_manifest.json
```

The large binary dataset is not committed directly to normal Git history. Use Git LFS or GitHub Release assets for large dataset files.

Additional release documentation is available in:

```text
data/README.md
docs/GITHUB_UPLOAD.md
docs/IEEE_DATAPORT_DATASET_DOCUMENTATION.md
docs/IEEE_DATAPORT_DATASET_AND_ANALYSIS_TOOLS.md
docs/IEEE_DATAPORT_INSTRUCTIONS.md
docs/IEEE_DATAPORT_USAGE_INSTRUCTIONS.txt
```
