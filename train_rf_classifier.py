from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from rfml_benchmark.data import HFIQDataset, load_metadata, make_splits
from rfml_benchmark.models import MODEL_NAMES, build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device_from_arg(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_epoch(model, loader, criterion, device, optimizer=None) -> tuple[float, float]:
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    total = 0
    correct = 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in tqdm(loader, leave=False):
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            total_loss += float(loss.item()) * y.size(0)
            total += y.size(0)
            correct += int((logits.argmax(dim=1) == y).sum().item())
    return total_loss / max(total, 1), correct / max(total, 1)


def collect_predictions(model, loader, device) -> dict[str, np.ndarray]:
    model.eval()
    preds, targets, indices, snrs, channels = [], [], [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, leave=False):
            x = batch["x"].to(device, non_blocking=True)
            logits = model(x)
            preds.append(logits.argmax(dim=1).cpu().numpy())
            targets.append(batch["y"].numpy())
            indices.append(batch["index"].numpy())
            if "snr_db" in batch:
                snrs.append(batch["snr_db"].numpy())
            if "channel_profile" in batch:
                channels.append(batch["channel_profile"].numpy())
    result = {
        "pred": np.concatenate(preds),
        "target": np.concatenate(targets),
        "index": np.concatenate(indices),
    }
    if snrs:
        result["snr_db"] = np.concatenate(snrs)
    if channels:
        result["channel_profile"] = np.concatenate(channels)
    return result


def grouped_accuracy(values: np.ndarray, targets: np.ndarray, preds: np.ndarray, label_name: str) -> list[dict[str, object]]:
    rows = []
    for value in sorted(np.unique(values).tolist()):
        mask = values == value
        rows.append(
            {
                label_name: value,
                "samples": int(mask.sum()),
                "accuracy": float(accuracy_score(targets[mask], preds[mask])),
                "macro_f1": float(f1_score(targets[mask], preds[mask], average="macro", zero_division=0)),
            }
        )
    return rows


def save_confusion_matrix(path: Path, targets: np.ndarray, preds: np.ndarray, mode_names: list[str]) -> None:
    cm = confusion_matrix(targets, preds, labels=np.arange(len(mode_names)))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path.with_suffix(".csv"), cm, fmt="%d", delimiter=",")

    fig, ax = plt.subplots(figsize=(11.5, 9.2), constrained_layout=True)
    image = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    cbar = fig.colorbar(image, ax=ax)
    cbar.ax.tick_params(labelsize=12)
    ax.set_title("Confusion Matrix", fontsize=16)
    ax.set_xlabel("Predicted", fontsize=15)
    ax.set_ylabel("True", fontsize=15)
    ax.set_xticks(np.arange(len(mode_names)))
    ax.set_yticks(np.arange(len(mode_names)))
    ax.set_xticklabels(mode_names, rotation=45, ha="right", rotation_mode="anchor", fontsize=10)
    ax.set_yticklabels(mode_names, fontsize=10)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate RF IQ classifiers on the generated HF dataset.")
    parser.add_argument("--dataset-root", required=True, help="Folder containing X.npy and label files.")
    parser.add_argument("--out-dir", default="runs", help="Output folder for checkpoints and metrics.")
    parser.add_argument("--model", choices=MODEL_NAMES, default="resnet1d", help="Model architecture.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260614)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0, help="Optional subset size for smoke tests.")
    parser.add_argument("--include-amp-phase", action="store_true", help="Use I,Q,amplitude,phase as 4 input channels.")
    args = parser.parse_args()

    set_seed(args.seed)
    device = device_from_arg(args.device)
    meta = load_metadata(args.dataset_root)
    out_dir = Path(args.out_dir) / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    train_idx, val_idx, test_idx = make_splits(
        args.dataset_root,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        max_samples=args.max_samples if args.max_samples > 0 else None,
    )
    np.save(out_dir / "train_indices.npy", train_idx)
    np.save(out_dir / "val_indices.npy", val_idx)
    np.save(out_dir / "test_indices.npy", test_idx)

    train_ds = HFIQDataset(args.dataset_root, train_idx, include_amp_phase=args.include_amp_phase)
    val_ds = HFIQDataset(args.dataset_root, val_idx, include_amp_phase=args.include_amp_phase)
    test_ds = HFIQDataset(args.dataset_root, test_idx, include_amp_phase=args.include_amp_phase)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")

    model = build_model(args.model, meta.num_classes, input_channels=train_ds.input_channels, width=args.width, dropout=args.dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    config = vars(args) | {
        "num_classes": meta.num_classes,
        "mode_names": meta.mode_names,
        "channel_profiles": meta.channel_profiles,
        "num_samples": meta.num_samples,
        "vector_len": meta.vector_len,
        "device": str(device),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    best_val_acc = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device, optimizer=None)
        scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(row)
        write_csv(out_dir / "history.csv", history)
        print(f"[{args.model}] epoch {epoch:03d}: train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"model": model.state_dict(), "config": config}, out_dir / "best.pt")

    checkpoint = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model"])
    pred_data = collect_predictions(model, test_loader, device)
    pred = pred_data["pred"]
    target = pred_data["target"]

    metrics = {
        "model": args.model,
        "test_samples": int(len(target)),
        "accuracy": float(accuracy_score(target, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(target, pred)),
        "macro_f1": float(f1_score(target, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(target, pred, average="weighted", zero_division=0)),
        "best_val_acc": float(best_val_acc),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    report = classification_report(target, pred, target_names=meta.mode_names, output_dict=True, zero_division=0)
    (out_dir / "classification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv(out_dir / "classification_report.csv", [{"label": k, **v} for k, v in report.items() if isinstance(v, dict)])

    if "snr_db" in pred_data:
        write_csv(out_dir / "metrics_by_snr.csv", grouped_accuracy(pred_data["snr_db"], target, pred, "snr_db"))
    if "channel_profile" in pred_data:
        rows = grouped_accuracy(pred_data["channel_profile"], target, pred, "channel_profile")
        for row in rows:
            idx = int(row["channel_profile"])
            if 0 <= idx < len(meta.channel_profiles):
                row["channel_name"] = meta.channel_profiles[idx]
        write_csv(out_dir / "metrics_by_channel.csv", rows)

    save_confusion_matrix(out_dir / "confusion_matrix.png", target, pred, meta.mode_names)
    np.savez_compressed(out_dir / "test_predictions.npz", **pred_data)
    print(json.dumps(metrics, indent=2))
    print(f"Saved results to {out_dir}")


if __name__ == "__main__":
    main()
