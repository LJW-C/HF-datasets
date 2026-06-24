# Raw WAV Generation with Fldigi

This document explains how to automatically generate raw WAV recordings from Fldigi digital modes.

## Purpose

The script `generate_fldigi_wavs.py` controls Fldigi through XML-RPC, sends random text payloads, records the transmitted audio from a selected recording device, and saves one WAV file per mode.

The output folder is used as the input of `build_panoradio_like_dataset.py`.

## Requirements

Install dependencies:

```powershell
pip install -r requirements.txt
```

Additional runtime requirements:

```text
Fldigi
Virtual audio cable, such as VB-CABLE
Correct Windows recording device id
```

## Audio Routing

Typical routing:

```text
Fldigi Playback device -> CABLE Input
Python recording device -> CABLE Output
```

The script records from the device specified by `--device-id`.

## List Audio Devices

```powershell
python generate_fldigi_wavs.py --list-devices
```

Find the recording device corresponding to the virtual cable output. In the original setup, the recommended device id was `7`, but this can change on different machines.

## Fldigi XML-RPC

Fldigi must be running before the script starts. The default XML-RPC URL is:

```text
http://127.0.0.1:7362
```

The script uses XML-RPC to:

```text
set modem mode
set audio carrier
clear and fill TX text
switch TX/RX
```

## Generate All WAV Files

```powershell
python generate_fldigi_wavs.py `
  --device-id 7 `
  --out-root "E:\仿真\raw_wav" `
  --duration-sec 180 `
  --record-fs 8000 `
  --carrier-hz 1500
```

## Generate Selected Modes

```powershell
python generate_fldigi_wavs.py `
  --device-id 7 `
  --out-root "E:\仿真\raw_wav" `
  --modes BPSK31,BPSK63,QPSK31,RTTY
```

## Main Parameters

| Parameter | Default | Meaning |
|---|---:|---|
| `--xmlrpc-url` | `http://127.0.0.1:7362` | Fldigi XML-RPC endpoint |
| `--device-id` | `7` | Recording device id |
| `--record-fs` | `8000` | WAV sample rate |
| `--duration-sec` | `180` | Recording duration per mode |
| `--carrier-hz` | `1500` | Fldigi audio carrier frequency |
| `--payload-chars` | `30000` | Random text payload length |
| `--out-root` | `E:\仿真\raw_wav` | Raw WAV output folder |
| `--modes` | `all` | All modes, or comma-separated selected modes |

## Output Layout

```text
raw_wav
+-- BPSK31
|   +-- BPSK31.wav
+-- BPSK63
|   +-- BPSK63.wav
+-- ...
+-- raw_wav_metadata.json
```

## Supported Default Modes

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

## Silence Check

After each recording, the script prints:

```text
RMS
Peak
```

If RMS is very low, the recording may be silent. Check:

```text
Fldigi playback device
virtual cable routing
Python recording device id
Windows audio permissions
```

