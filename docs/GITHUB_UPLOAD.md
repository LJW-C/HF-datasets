# GitHub Upload Guide

This guide publishes the code repository and the generated dataset.

## 1. Create a New GitHub Repository

Create an empty repository on GitHub, for example:

```text
hf-digital-mode-iq-dataset
```

Do not initialize it with a README, license, or `.gitignore`, because this
folder already contains those files.

## 2. Push the Code

From this repository folder:

```powershell
git init
git branch -M main
git add .
git commit -m "Initial release of HF digital-mode IQ dataset pipeline"
git remote add origin https://github.com/<your-user>/hf-digital-mode-iq-dataset.git
git push -u origin main
```

If you use GitHub CLI:

```powershell
gh repo create <your-user>/hf-digital-mode-iq-dataset `
  --public `
  --source . `
  --remote origin `
  --push
```

## 3. Publish the Dataset

The generated `X.npy` file is large, so the preferred method is to upload the
dataset as GitHub Release assets.

Create dataset shards:

```powershell
python scripts\shard_dataset_for_release.py `
  --dataset-root "E:\仿真\panoradio_like_ccir520_npy_fixed" `
  --out-dir release\panoradio_like_ccir520_npy_fixed_shards `
  --shard-size-mb 512
```

Create a release:

```powershell
gh release create v1.0.0 `
  release\panoradio_like_ccir520_npy_fixed_shards\* `
  --title "HF Digital-Mode IQ Dataset v1.0.0" `
  --notes "Fixed 17-class HF digital-mode IQ dataset generated from Fldigi WAV recordings."
```

If you do not use GitHub CLI, create a release in the GitHub web interface and
drag the shard files into the release assets area.

## 4. Verify the Upload

After downloading the release assets, restore and verify:

```powershell
python scripts\restore_dataset_from_shards.py `
  --shard-dir release\panoradio_like_ccir520_npy_fixed_shards `
  --out-root data\panoradio_like_ccir520_npy_fixed
```

Compare the restored files with:

```text
data/panoradio_like_ccir520_npy_fixed/dataset_manifest.json
```

## Notes

This environment may not have `git`, `gh`, or `git-lfs` installed. If any
command is missing, install Git for Windows and GitHub CLI first.
