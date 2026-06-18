from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


@dataclass
class DatasetMetadata:
    dataset_root: Path
    mode_names: list[str]
    channel_profiles: list[str]
    num_classes: int
    num_samples: int
    vector_len: int


def _read_json_list(path: Path, fallback: list[str]) -> list[str]:
    if not path.exists():
        return fallback
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_metadata(dataset_root: str | Path) -> DatasetMetadata:
    root = Path(dataset_root)
    x_path = root / "X.npy"
    y_path = root / "y.npy"
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(f"Expected X.npy and y.npy under {root}")

    X = np.load(x_path, mmap_mode="r")
    y = np.load(y_path, mmap_mode="r")
    mode_count = int(np.max(y)) + 1
    mode_names = _read_json_list(root / "mode_names.json", [f"class_{i}" for i in range(mode_count)])
    channel_profiles = _read_json_list(root / "channel_profiles.json", ["none"])
    return DatasetMetadata(
        dataset_root=root,
        mode_names=mode_names,
        channel_profiles=channel_profiles,
        num_classes=mode_count,
        num_samples=len(X),
        vector_len=int(X.shape[1]),
    )


class HFIQDataset(Dataset):
    def __init__(
        self,
        dataset_root: str | Path,
        indices: np.ndarray,
        normalize: bool = True,
        include_amp_phase: bool = False,
    ) -> None:
        self.root = Path(dataset_root)
        self.X = np.load(self.root / "X.npy", mmap_mode="r")
        self.y = np.load(self.root / "y.npy", mmap_mode="r")
        self.snr = np.load(self.root / "snr_db.npy", mmap_mode="r") if (self.root / "snr_db.npy").exists() else None
        self.channel = (
            np.load(self.root / "channel_profile.npy", mmap_mode="r")
            if (self.root / "channel_profile.npy").exists()
            else None
        )
        self.indices = np.asarray(indices, dtype=np.int64)
        self.normalize = normalize
        self.include_amp_phase = include_amp_phase

    @property
    def input_channels(self) -> int:
        return 4 if self.include_amp_phase else 2

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        index = int(self.indices[item])
        x = np.asarray(self.X[index], dtype=np.complex64)
        if self.normalize:
            power = float(np.mean(np.abs(x) ** 2))
            x = x / np.sqrt(max(power, 1e-12))

        features = [x.real.astype(np.float32), x.imag.astype(np.float32)]
        if self.include_amp_phase:
            amp = np.abs(x).astype(np.float32)
            phase = np.angle(x).astype(np.float32)
            features.extend([amp, phase])

        sample = {
            "x": torch.from_numpy(np.stack(features, axis=0).astype(np.float32)),
            "y": torch.tensor(int(self.y[index]), dtype=torch.long),
            "index": torch.tensor(index, dtype=torch.long),
        }
        if self.snr is not None:
            sample["snr_db"] = torch.tensor(int(self.snr[index]), dtype=torch.long)
        if self.channel is not None:
            sample["channel_profile"] = torch.tensor(int(self.channel[index]), dtype=torch.long)
        return sample


def make_splits(
    dataset_root: str | Path,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    max_samples: int | None = None,
    stratify_by_snr: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    root = Path(dataset_root)
    y = np.load(root / "y.npy", mmap_mode="r")
    all_indices = np.arange(len(y), dtype=np.int64)

    if max_samples is not None and max_samples > 0 and max_samples < len(all_indices):
        rng = np.random.default_rng(seed)
        all_indices = np.sort(rng.choice(all_indices, size=max_samples, replace=False))

    labels = np.asarray(y[all_indices], dtype=np.int64)
    if stratify_by_snr and (root / "snr_db.npy").exists():
        snr = np.load(root / "snr_db.npy", mmap_mode="r")
        snr_values = np.asarray(snr[all_indices], dtype=np.int64)
        _, snr_ids = np.unique(snr_values, return_inverse=True)
        stratify = labels * 100 + snr_ids
    else:
        stratify = labels

    test_ratio = 1.0 - train_ratio - val_ratio
    if test_ratio <= 0:
        raise ValueError("train_ratio + val_ratio must be less than 1.0")

    try:
        train_idx, tmp_idx = train_test_split(
            all_indices,
            train_size=train_ratio,
            random_state=seed,
            shuffle=True,
            stratify=stratify,
        )
        stratify_by_index = {int(index): int(label) for index, label in zip(all_indices, stratify)}
        tmp_stratify = np.asarray([stratify_by_index[int(index)] for index in tmp_idx], dtype=np.int64)
        val_fraction_of_tmp = val_ratio / (val_ratio + test_ratio)
        val_idx, test_idx = train_test_split(
            tmp_idx,
            train_size=val_fraction_of_tmp,
            random_state=seed + 1,
            shuffle=True,
            stratify=tmp_stratify,
        )
    except ValueError:
        train_idx, tmp_idx = train_test_split(
            all_indices,
            train_size=train_ratio,
            random_state=seed,
            shuffle=True,
            stratify=None,
        )
        val_fraction_of_tmp = val_ratio / (val_ratio + test_ratio)
        val_idx, test_idx = train_test_split(
            tmp_idx,
            train_size=val_fraction_of_tmp,
            random_state=seed + 1,
            shuffle=True,
            stratify=None,
        )

    return np.asarray(train_idx), np.asarray(val_idx), np.asarray(test_idx)
