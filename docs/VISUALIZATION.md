# Visualization

The script `visualize_iq_samples.py` visualizes complex IQ vectors stored in `X.npy`.

Each output image contains four plots:

1. I/Q waveform: `I(t)` and `Q(t)`
2. Constellation plot
3. FFT magnitude spectrum
4. Time-frequency spectrogram

## Random Samples

```powershell
python visualize_iq_samples.py `
  --dataset-root "E:\仿真\panoradio_like_ccir520_npy" `
  --random 20
```

## Selected Samples

```powershell
python visualize_iq_samples.py `
  --dataset-root "E:\仿真\panoradio_like_ccir520_npy" `
  --indices 0,1,2,100
```

## Slice Syntax

The `--indices` argument also supports Python-like slices:

```powershell
python visualize_iq_samples.py `
  --dataset-root "E:\仿真\panoradio_like_ccir520_npy" `
  --indices 0:100:5
```

This visualizes:

```text
0, 5, 10, ..., 95
```

## Visualize Every Sample

```powershell
python visualize_iq_samples.py `
  --dataset-root "E:\仿真\panoradio_like_ccir520_npy" `
  --all
```

Be careful with `--all`: a full dataset may contain more than 160,000 samples, so this command can generate more than 160,000 PNG images.

## Output Directory

By default, images are written to:

```text
dataset_root\visualizations
```

You can choose another output directory:

```powershell
python visualize_iq_samples.py `
  --dataset-root "E:\仿真\panoradio_like_ccir520_npy" `
  --random 20 `
  --out-dir "E:\仿真\sample_visualizations"
```

