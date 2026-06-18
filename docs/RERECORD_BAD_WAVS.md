# Re-record Incorrect WAV Files

## Current Diagnosis

The full raw-WAV validation found one explicit bandwidth mismatch:

```text
Contestia_4_250: expected about 250 Hz, measured about 500.8 Hz
```

The following explicit-bandwidth modes passed the current check:

```text
Contestia_8_500
Olivia_4_250
Olivia_8_500
```

The likely cause was that the old recording script selected the generic
`CONTESTIA` modem and then tried to set Contestia parameters through the
Olivia XML-RPC interface. The updated script now uses Fldigi's exact modem
names:

```text
Contestia_4_250 -> Cont-4/250
Contestia_8_500 -> Cont-8/500
Olivia_4_250    -> OLIVIA-4/250
Olivia_8_500    -> OLIVIA-8/500
DominoEX_8      -> DOMEX8
```

## Re-record the Failed Mode

Make sure Fldigi is open, XML-RPC is enabled, and Fldigi playback is routed to
the virtual audio cable. Install the audio dependencies if needed:

```powershell
pip install sounddevice soundfile scipy
```

Then re-record only the failed mode:

```powershell
python generate_fldigi_wavs.py `
  --device-id 7 `
  --duration-sec 180 `
  --modes Contestia_4_250 `
  --out-root "E:\仿真\raw_wav" `
  --print-modems
```

The script backs up the existing WAV before overwriting it. After recording, it
prints an OBW check. For `Contestia_4_250`, the `OBW99` value should be near
250 Hz, not 500 Hz.

## Validate All Raw WAV Files Again

```powershell
python validate_raw_wavs.py `
  --raw-root "E:\仿真\raw_wav" `
  --out-dir diagnostics\raw_wav_validation_after_rerecord
```

Expected result:

```text
Contestia_4_250    PASS    OBW99 approximately 250 Hz
```

## Rebuild the IQ Dataset

After the raw WAV files pass validation, rebuild the dataset:

```powershell
python build_panoradio_like_dataset.py `
  --raw-root "E:\仿真\raw_wav" `
  --out-root "E:\仿真\panoradio_like_ccir520_npy_fixed" `
  --fs 6000 `
  --vector-len 2048 `
  --carrier-hz 1500 `
  --vectors-per-mode-snr 1200 `
  --channel ccir520 `
  --ccir520-profile random
```

Then rerun visualization and model training on the fixed dataset.
