#!/usr/bin/env python3
"""
Generate paper-ready tables and figures from model benchmark outputs.

Expected run directory layout:
  runs/model_suite/<model>/metrics.json
  runs/model_suite/<model>/metrics_by_snr.csv
  runs/model_suite/<model>/confusion_matrix.csv
  runs/model_suite/<model>/config.json

The script does not invent benchmark values. If no model results are found, it
creates explicit placeholder assets so the LaTeX paper can still compile.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DISPLAY_NAMES = {
    "iq_cnn": "IQ-CNN",
    "resnet1d": "ResNet-1D",
    "tcn": "TCN",
    "cldnn": "CLDNN",
    "transformer": "Transformer",
    "conformer": "Conformer",
}

TITLE_FONTSIZE = 15
AXIS_LABEL_FONTSIZE = 14
TICK_FONTSIZE = 12
LEGEND_FONTSIZE = 11
CONFUSION_TICK_FONTSIZE = 9
CONFUSION_CELL_FONTSIZE = 7


def display_name(model: str) -> str:
    return DISPLAY_NAMES.get(model, model.replace("_", "-"))


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def find_model_dirs(runs_dir: Path) -> list[Path]:
    if not runs_dir.exists():
        return []
    dirs = []
    for item in runs_dir.iterdir():
        if item.is_dir() and (item / "metrics.json").exists():
            dirs.append(item)
    return sorted(dirs)


def percent(value: float) -> str:
    return f"{100.0 * value:.2f}"


def tex_escape(text: str) -> str:
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
    )


def write_placeholder_assets(paper_dir: Path) -> None:
    figures_dir = paper_dir / "figures"
    tables_dir = paper_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    for name, title in [
        ("overall_metrics_by_model.png", "No model results found"),
        ("snr_accuracy_by_model.png", "No SNR results found"),
        ("confusion_matrix_best.png", "No confusion matrix found"),
    ]:
        fig, ax = plt.subplots(figsize=(6.8, 4.0), dpi=180)
        ax.text(0.5, 0.55, title, ha="center", va="center", fontsize=TITLE_FONTSIZE)
        ax.text(
            0.5,
            0.42,
            "Run model training first, then run make_paper_results.py.",
            ha="center",
            va="center",
            fontsize=11,
        )
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(figures_dir / name, bbox_inches="tight")
        plt.close(fig)

    (tables_dir / "model_summary.tex").write_text(
        "\n".join(
            [
                r"\begin{table}[!t]",
                r"\caption{Overall Classification Results}",
                r"\label{tab:overall_results}",
                r"\centering",
                r"\footnotesize",
                r"\begin{tabular}{lc}",
                r"\hline",
                r"\hline",
                r"Status & Value \\",
                r"\hline",
                r"No benchmark results found & -- \\",
                r"\hline",
                r"\hline",
                r"\end{tabular}",
                r"\end{table}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (tables_dir / "snr_accuracy_table.tex").write_text(
        "\n".join(
            [
                r"\begin{table}[!t]",
                r"\caption{Per-SNR Accuracy Results}",
                r"\label{tab:snr_results}",
                r"\centering",
                r"\footnotesize",
                r"\begin{tabular}{lc}",
                r"\hline",
                r"\hline",
                r"Status & Value \\",
                r"\hline",
                r"No per-SNR benchmark results found & -- \\",
                r"\hline",
                r"\hline",
                r"\end{tabular}",
                r"\end{table}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (tables_dir / "result_summary_text.tex").write_text(
        "No benchmark result files were found when the paper assets were generated.\n",
        encoding="utf-8",
    )


def collect_results(model_dirs: list[Path]) -> list[dict]:
    results = []
    for model_dir in model_dirs:
        metrics_path = model_dir / "metrics.json"
        snr_path = model_dir / "metrics_by_snr.csv"
        config_path = model_dir / "config.json"
        if not metrics_path.exists() or not snr_path.exists():
            continue

        metrics = read_json(metrics_path)
        config = read_json(config_path) if config_path.exists() else {}
        snr_rows = read_csv_rows(snr_path)
        snr_points = []
        for row in snr_rows:
            snr_points.append(
                {
                    "snr_db": float(row["snr_db"]),
                    "samples": int(row["samples"]),
                    "accuracy": float(row["accuracy"]),
                    "macro_f1": float(row.get("macro_f1", "nan")),
                }
            )
        snr_points.sort(key=lambda x: x["snr_db"])

        model = str(metrics.get("model", model_dir.name))
        results.append(
            {
                "model": model,
                "model_dir": model_dir,
                "metrics": metrics,
                "config": config,
                "snr_points": snr_points,
            }
        )

    results.sort(key=lambda x: float(x["metrics"].get("accuracy", 0.0)), reverse=True)
    return results


def write_model_summary_table(results: list[dict], out_path: Path) -> None:
    lines = [
        r"\begin{table}[!t]",
        r"\caption{Overall Classification Results}",
        r"\label{tab:overall_results}",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.12}",
        r"\begin{tabular}{lccccc}",
        r"\hline",
        r"\hline",
        r"Model & Acc. & BAcc. & Macro-F1 & Weighted-F1 & Best Val. \\",
        r"\hline",
    ]
    for item in results:
        metrics = item["metrics"]
        lines.append(
            " & ".join(
                [
                    tex_escape(display_name(item["model"])),
                    percent(float(metrics["accuracy"])),
                    percent(float(metrics["balanced_accuracy"])),
                    percent(float(metrics["macro_f1"])),
                    percent(float(metrics["weighted_f1"])),
                    percent(float(metrics["best_val_acc"])),
                ]
            )
            + r" \\"
        )
    lines.extend(
        [
            r"\hline",
            r"\hline",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_snr_table(results: list[dict], out_path: Path) -> None:
    snrs = sorted({point["snr_db"] for item in results for point in item["snr_points"]})
    lines = [
        r"\begin{table*}[!t]",
        r"\caption{Per-SNR Test Accuracy of Baseline Models (\%)}",
        r"\label{tab:snr_results}",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.12}",
        r"\begin{tabular}{l" + "c" * len(snrs) + "}",
        r"\hline",
        r"\hline",
        "Model & " + " & ".join(f"{snr:g} dB" for snr in snrs) + r" \\",
        r"\hline",
    ]
    for item in results:
        by_snr = {point["snr_db"]: point["accuracy"] for point in item["snr_points"]}
        cells = [tex_escape(display_name(item["model"]))]
        for snr in snrs:
            value = by_snr.get(snr)
            cells.append("--" if value is None or math.isnan(value) else percent(value))
        lines.append(" & ".join(cells) + r" \\")
    lines.extend(
        [
            r"\hline",
            r"\hline",
            r"\end{tabular}",
            r"\end{table*}",
            "",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_summary_text(results: list[dict], out_path: Path) -> None:
    best = results[0]
    metrics = best["metrics"]
    config = best["config"]
    best_name = display_name(best["model"])
    test_samples = int(metrics.get("test_samples", 0))
    total_samples = int(config.get("num_samples", 0))
    num_classes = int(config.get("num_classes", 0))
    num_profiles = len(config.get("channel_profiles", []))
    snr_points = best["snr_points"]
    low = snr_points[0] if snr_points else None
    high = snr_points[-1] if snr_points else None

    first_sentence = (
        f"The generated benchmark contains {total_samples:,} IQ vectors"
        if total_samples
        else "The generated benchmark was evaluated with the available IQ vectors"
    )
    if num_classes and num_profiles:
        first_sentence += f", with {num_classes} digital-mode classes and {num_profiles} channel profiles"
    elif num_classes:
        first_sentence += f", with {num_classes} digital-mode classes"
    elif num_profiles:
        first_sentence += f", with {num_profiles} channel profiles"
    if test_samples:
        first_sentence += f". The reported test split contains {test_samples:,} vectors."
    else:
        first_sentence += "."

    second_sentence = (
        f"The best overall result is obtained by {best_name}, with "
        f"{percent(float(metrics['accuracy']))}\\% accuracy, "
        f"{percent(float(metrics['balanced_accuracy']))}\\% balanced accuracy, and "
        f"{percent(float(metrics['macro_f1']))}\\% macro-F1."
    )
    if low and high:
        second_sentence += (
            f" For this model, accuracy increases from "
            f"{percent(low['accuracy'])}\\% at {low['snr_db']:g} dB to "
            f"{percent(high['accuracy'])}\\% at {high['snr_db']:g} dB."
        )

    out_path.write_text(first_sentence + "\n\n" + second_sentence + "\n", encoding="utf-8")


def plot_overall_metrics(results: list[dict], out_path: Path) -> None:
    ordered = list(reversed(results))
    names = [display_name(item["model"]) for item in ordered]
    accuracy = [100.0 * float(item["metrics"]["accuracy"]) for item in ordered]
    macro_f1 = [100.0 * float(item["metrics"]["macro_f1"]) for item in ordered]

    x = np.arange(len(names))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=220)
    ax.bar(x - width / 2, accuracy, width, label="Accuracy", color="#2f6f9f")
    ax.bar(x + width / 2, macro_f1, width, label="Macro-F1", color="#b86b2d")
    ax.set_ylabel("Score (%)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylim(0, 100)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=TICK_FONTSIZE)
    ax.tick_params(axis="y", labelsize=TICK_FONTSIZE)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(frameon=False, ncol=2, loc="upper left", fontsize=LEGEND_FONTSIZE)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_snr_accuracy(results: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.9), dpi=220)
    for item in results:
        points = item["snr_points"]
        snr = [point["snr_db"] for point in points]
        acc = [100.0 * point["accuracy"] for point in points]
        ax.plot(snr, acc, marker="o", linewidth=1.8, markersize=4.2, label=display_name(item["model"]))

    ax.set_xlabel("SNR (dB)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Accuracy (%)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylim(0, 100)
    ax.set_xticks(sorted({point["snr_db"] for item in results for point in item["snr_points"]}))
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(frameon=False, ncol=2, fontsize=LEGEND_FONTSIZE)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(best: dict, out_path: Path) -> None:
    cm_path = best["model_dir"] / "confusion_matrix.csv"
    if not cm_path.exists():
        raise FileNotFoundError(f"Missing confusion matrix: {cm_path}")

    cm = np.loadtxt(cm_path, delimiter=",")
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
    labels = best["config"].get("mode_names") or [str(i) for i in range(cm.shape[0])]
    labels = [str(label) for label in labels]

    fig_size = max(8.2, 0.48 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size + 1.0), dpi=220)
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_title(f"Normalized confusion matrix: {display_name(best['model'])}", fontsize=TITLE_FONTSIZE)
    ax.set_xlabel("Predicted label", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("True label", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", rotation_mode="anchor", fontsize=CONFUSION_TICK_FONTSIZE)
    ax.set_yticklabels(labels, fontsize=CONFUSION_TICK_FONTSIZE)
    ax.tick_params(length=0)

    threshold = 0.55
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            value = cm_norm[i, j]
            if value >= 0.08 or i == j:
                ax.text(
                    j,
                    i,
                    f"{100.0 * value:.0f}",
                    ha="center",
                    va="center",
                    fontsize=CONFUSION_CELL_FONTSIZE,
                    color="white" if value > threshold else "black",
                )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row-normalized accuracy", fontsize=AXIS_LABEL_FONTSIZE)
    cbar.ax.tick_params(labelsize=TICK_FONTSIZE)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def write_summary_json(results: list[dict], out_path: Path) -> None:
    serializable = []
    for item in results:
        serializable.append(
            {
                "model": item["model"],
                "display_name": display_name(item["model"]),
                "model_dir": str(item["model_dir"]),
                "metrics": item["metrics"],
                "snr_points": item["snr_points"],
            }
        )
    out_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs/model_suite"))
    parser.add_argument("--paper-dir", type=Path, default=Path("paper"))
    args = parser.parse_args()

    paper_dir = args.paper_dir
    figures_dir = paper_dir / "figures"
    tables_dir = paper_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    model_dirs = find_model_dirs(args.runs_dir)
    results = collect_results(model_dirs)
    if not results:
        write_placeholder_assets(paper_dir)
        print(f"No benchmark results found under {args.runs_dir}. Placeholder assets were written.")
        return

    write_model_summary_table(results, tables_dir / "model_summary.tex")
    write_snr_table(results, tables_dir / "snr_accuracy_table.tex")
    write_summary_text(results, tables_dir / "result_summary_text.tex")
    plot_overall_metrics(results, figures_dir / "overall_metrics_by_model.png")
    plot_snr_accuracy(results, figures_dir / "snr_accuracy_by_model.png")
    plot_confusion_matrix(results[0], figures_dir / "confusion_matrix_best.png")
    plot_overall_metrics(results, figures_dir / "overall_metrics_by_model.pdf")
    plot_snr_accuracy(results, figures_dir / "snr_accuracy_by_model.pdf")
    plot_confusion_matrix(results[0], figures_dir / "confusion_matrix_best.pdf")
    write_summary_json(results, paper_dir / "results_summary.json")

    best = results[0]
    print(
        "Generated paper results for "
        f"{len(results)} models. Best model: {display_name(best['model'])} "
        f"({percent(float(best['metrics']['accuracy']))}% accuracy)."
    )
    print(f"Figures: {figures_dir}")
    print(f"Tables: {tables_dir}")


if __name__ == "__main__":
    main()
