from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_metrics(model_dir: Path) -> dict[str, object] | None:
    path = model_dir / "metrics.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    data["model_dir"] = str(model_dir)
    data["model"] = data.get("model", model_dir.name)
    return data


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row.keys()})
    preferred = ["model", "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "best_val_acc", "test_samples", "model_dir"]
    fields = [x for x in preferred if x in fields] + [x for x in fields if x not in preferred]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# Model Benchmark Summary",
        "",
        "| Model | Accuracy | Balanced Acc. | Macro F1 | Weighted F1 | Best Val Acc. | Test Samples |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {model} | {accuracy:.4f} | {balanced_accuracy:.4f} | {macro_f1:.4f} | "
            "{weighted_f1:.4f} | {best_val_acc:.4f} | {test_samples} |".format(
                model=row.get("model", ""),
                accuracy=float(row.get("accuracy", 0.0)),
                balanced_accuracy=float(row.get("balanced_accuracy", 0.0)),
                macro_f1=float(row.get("macro_f1", 0.0)),
                weighted_f1=float(row.get("weighted_f1", 0.0)),
                best_val_acc=float(row.get("best_val_acc", 0.0)),
                test_samples=int(row.get("test_samples", 0)),
            )
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize trained model metrics.")
    parser.add_argument("--runs-dir", required=True, help="Folder containing one subfolder per model.")
    parser.add_argument("--out-csv", default=None)
    parser.add_argument("--out-md", default=None)
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    rows = []
    for child in sorted(runs_dir.iterdir()):
        if not child.is_dir():
            continue
        metrics = read_metrics(child)
        if metrics:
            rows.append(metrics)

    rows.sort(key=lambda row: float(row.get("accuracy", 0.0)), reverse=True)
    out_csv = Path(args.out_csv) if args.out_csv else runs_dir / "model_summary.csv"
    out_md = Path(args.out_md) if args.out_md else runs_dir / "model_summary.md"
    write_csv(out_csv, rows)
    write_markdown(out_md, rows)
    print(f"Found {len(rows)} model result(s).")
    print(f"CSV: {out_csv}")
    print(f"Markdown: {out_md}")


if __name__ == "__main__":
    main()

