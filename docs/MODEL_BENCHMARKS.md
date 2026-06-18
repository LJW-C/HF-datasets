# Model Benchmarks

This repository includes PyTorch code for validating the generated HF IQ dataset with several model families.

## Supported Models

| Model name | Script value | Description |
|---|---|---|
| IQ CNN | `iq_cnn` | Lightweight 1-D convolutional baseline for I/Q input |
| ResNet-1D | `resnet1d` | Residual 1-D CNN, strong default baseline |
| TCN | `tcn` | Dilated temporal convolutional network |
| CLDNN | `cldnn` | CNN front-end plus bidirectional GRU sequence model |
| Transformer | `transformer` | Patch embedding plus Transformer encoder |
| Conformer-style | `conformer` | Transformer attention plus depthwise temporal convolution |

All models read the same input format:

```text
shape: [batch, 2, 2048]
channels: I, Q
```

Optional mode:

```text
--include-amp-phase
```

uses:

```text
shape: [batch, 4, 2048]
channels: I, Q, amplitude, phase
```

## Install Dependencies

```powershell
pip install -r requirements.txt
```

For GPU training, install the PyTorch build that matches your CUDA version from the official PyTorch website.

## Smoke Test

Before training on the full dataset, run a small smoke test:

```powershell
python train_rf_classifier.py `
  --dataset-root "E:\仿真\panoradio_like_ccir520_npy" `
  --model resnet1d `
  --epochs 2 `
  --batch-size 64 `
  --max-samples 2000 `
  --out-dir runs/smoke
```

## Train One Model

```powershell
python train_rf_classifier.py `
  --dataset-root "E:\仿真\panoradio_like_ccir520_npy" `
  --model resnet1d `
  --epochs 30 `
  --batch-size 128 `
  --out-dir runs/hf_benchmark
```

## Run Multiple Models

```powershell
python run_model_suite.py `
  --dataset-root "E:\仿真\panoradio_like_ccir520_npy" `
  --models iq_cnn,resnet1d,tcn,cldnn,transformer,conformer `
  --epochs 30 `
  --batch-size 128 `
  --out-dir runs/model_suite
```

## Summarize Multiple Models

```powershell
python summarize_model_results.py `
  --runs-dir runs/model_suite
```

This creates:

```text
runs/model_suite/model_summary.csv
runs/model_suite/model_summary.md
```

## Output Files

Each model writes results to:

```text
runs/<experiment>/<model_name>
```

Important files:

| File | Meaning |
|---|---|
| `best.pt` | best checkpoint selected by validation accuracy |
| `config.json` | model and dataset configuration |
| `history.csv` | epoch-by-epoch train/validation metrics |
| `metrics.json` | final test accuracy, balanced accuracy, macro F1, weighted F1 |
| `metrics_by_snr.csv` | accuracy and macro F1 for each SNR |
| `metrics_by_channel.csv` | accuracy and macro F1 for each channel profile |
| `classification_report.csv` | precision/recall/F1 per class |
| `confusion_matrix.png` | confusion matrix image |
| `test_predictions.npz` | test predictions and labels |

## Recommended Reporting

For a dataset paper or benchmark report, include:

```text
overall accuracy
balanced accuracy
macro F1
accuracy by SNR
accuracy by channel profile
confusion matrix
training/validation curves
```

Do not report only overall accuracy, because HF datasets can be much harder at low SNR and under stronger fading profiles.

## Generate Paper Figures

After running the model suite, generate the LaTeX-ready result assets:

```powershell
python make_paper_results.py `
  --runs-dir runs/model_suite `
  --paper-dir paper
```

This writes:

| Output | Meaning |
|---|---|
| `paper/tables/model_summary.tex` | overall accuracy, balanced accuracy, macro F1, weighted F1, and best validation accuracy |
| `paper/tables/snr_accuracy_table.tex` | per-SNR accuracy table |
| `paper/tables/result_summary_text.tex` | short result paragraph for the paper |
| `paper/figures/overall_metrics_by_model.png` | overall model comparison |
| `paper/figures/snr_accuracy_by_model.png` | accuracy versus SNR curve |
| `paper/figures/confusion_matrix_best.png` | normalized confusion matrix of the best model |

The IEEE LaTeX document imports these files directly, so rerunning this script updates the reported results without editing the paper by hand.
