````markdown
# Dataset Files

This folder stores dataset metadata and release instructions.

The full dataset generated in the current experiment is:

```text
E:\仿真\panoradio_like_ccir520_npy_fixed
````

Core files:

```text
X.npy                  2,673,868,928 bytes
y.npy                  1,305,728 bytes
snr_db.npy             326,528 bytes
channel_profile.npy    326,528 bytes
mode_names.json        256 bytes
channel_profiles.json  54 bytes
meta.json              1,850 bytes
```

The checksum manifest is:

```text
panoradio_like_ccir520_npy_fixed/dataset_manifest.json
```

## Dataset Release

The latest version of the generated dataset has been released on IEEE DataPort:

```text
Junwei Li, "A Reproducible HF Digital-Mode I/Q Dataset Generation Framework from Fldigi WAV Recordings",
IEEE DataPort, June 17, 2026, doi:10.21227/9pzn-h761
```

A mirror copy is also available through Baidu Netdisk:

```text
File: dataset.rar
Link: https://pan.baidu.com/s/1jsF8lL9DguAO7UcSWeW5Dg?pwd=1234
Extraction code: 1234
```

Users are encouraged to cite the IEEE DataPort record when using this dataset.

## Recommended GitHub Release Method

Do not upload the large dataset into normal Git history. Use one of these:

1. GitHub Release assets, preferably split into smaller shards.
2. Git LFS, if your account has enough LFS storage and file-size allowance.
3. IEEE DataPort, Zenodo, Hugging Face Datasets, or another dataset archive, then link it from this repository.

To prepare GitHub Release shards:

```powershell
python scripts\shard_dataset_for_release.py `
  --dataset-root "E:\仿真\panoradio_like_ccir520_npy_fixed" `
  --out-dir release\panoradio_like_ccir520_npy_fixed_shards `
  --shard-size-mb 512
```

Upload all files in `release\panoradio_like_ccir520_npy_fixed_shards` as Release
assets. Users can restore the dataset with:

```powershell
python scripts\restore_dataset_from_shards.py `
  --shard-dir release\panoradio_like_ccir520_npy_fixed_shards `
  --out-root data\panoradio_like_ccir520_npy_fixed
```

## Git LFS Alternative

If using Git LFS, copy the dataset into this repository and run:

```powershell
git lfs install
git lfs track "*.npy" "*.npz" "*.wav"
git add .gitattributes data/
git commit -m "Add dataset files with Git LFS"
git push origin main
```

For very large files, Release assets or an external dataset archive are usually
easier to manage than keeping the dataset in Git history.

```
```
