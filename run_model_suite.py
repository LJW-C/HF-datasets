from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from rfml_benchmark.models import MODEL_NAMES


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multiple RF classifier models on the same dataset split settings.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--out-dir", default="runs/model_suite")
    parser.add_argument("--models", default="iq_cnn,resnet1d,tcn,cldnn,transformer,conformer")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260614)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--extra-args", default="", help="Extra arguments passed verbatim to train_rf_classifier.py.")
    args = parser.parse_args()

    selected = [name.strip() for name in args.models.split(",") if name.strip()]
    unknown = sorted(set(selected) - set(MODEL_NAMES))
    if unknown:
        raise ValueError(f"Unknown model(s): {', '.join(unknown)}")

    root = Path(args.out_dir)
    root.mkdir(parents=True, exist_ok=True)

    for model in selected:
        cmd = [
            sys.executable,
            "train_rf_classifier.py",
            "--dataset-root",
            args.dataset_root,
            "--out-dir",
            str(root),
            "--model",
            model,
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--lr",
            str(args.lr),
            "--seed",
            str(args.seed),
            "--device",
            args.device,
        ]
        if args.max_samples > 0:
            cmd.extend(["--max-samples", str(args.max_samples)])
        if args.extra_args.strip():
            cmd.extend(args.extra_args.split())
        print("\n" + "=" * 88)
        print("Running:", " ".join(cmd))
        print("=" * 88)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

