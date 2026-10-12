"""Plot training/evaluation results and sample ROCO captions for CNN+LSTM."""

from __future__ import annotations

import argparse
import csv
import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/cnn_lstm_roco_full"))
    parser.add_argument("--data-root", type=Path, default=Path(r"C:\AIE Files\Projects\S5\Dl\DATA\rocov2"))
    parser.add_argument("--samples", type=int, default=6)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def plot_training(metrics: list[dict], output_dir: Path) -> None:
    epochs = [row["epoch"] for row in metrics]
    fig, ax = plt.subplots(figsize=(8, 5), dpi=180)
    ax.plot(epochs, [row["train_loss"] for row in metrics], "o-", label="Train")
    ax.plot(epochs, [row["valid_loss"] for row in metrics], "s-", label="Validation")
    best = min(metrics, key=lambda row: row["valid_loss"])
    ax.axvline(best["epoch"], color="green", linestyle="--", alpha=0.7,
               label=f"Best validation epoch {best['epoch']}")
    ax.set(title="CNN+LSTM caption loss", xlabel="Epoch", ylabel="Token cross-entropy loss")
    ax.set_xticks(epochs)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "training_validation_loss.png", bbox_inches="tight")
    plt.close(fig)

    if "learning_rate_head" in metrics[0]:
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=180)
        ax.plot(epochs, [row["learning_rate_head"] for row in metrics], "o-", label="Decoder/projection")
        ax.plot(epochs, [row["learning_rate_encoder"] for row in metrics], "s-", label="ResNet encoder")
        ax.set(title="Learning rates", xlabel="Epoch", ylabel="Learning rate")
        ax.set_yscale("log")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "learning_rates.png", bbox_inches="tight")
        plt.close(fig)


def plot_test_metrics(metrics: dict, output_dir: Path) -> None:
    names = ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "METEOR", "ROUGE-L"]
    keys = ["bleu1", "bleu2", "bleu3", "bleu4", "meteor", "rougeL"]
    values = [metrics.get(key) for key in keys]
    names_and_values = [(name, value) for name, value in zip(names, values) if value is not None]
    if not names_and_values:
        return
    labels, scores = zip(*names_and_values)
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=180)
    bars = ax.bar(labels, scores, color="#3670a6")
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.set(title="Caption metrics on held-out test set", ylabel="Score")
    ax.set_ylim(0, max(scores) * 1.2 if max(scores) > 0 else 1)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "test_caption_metrics.png", bbox_inches="tight")
    plt.close(fig)


def plot_timing(metrics: dict, output_dir: Path) -> None:
    keys = ["evaluation_loop_seconds", "model_forward_seconds", "generation_seconds"]
    labels = ["Evaluation loop", "Teacher-forced forward", "Caption generation"]
    items = [(label, metrics.get(key)) for label, key in zip(labels, keys)]
    items = [(label, value) for label, value in items if value is not None]
    if not items:
        return
    labels, seconds = zip(*items)
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=180)
    bars = ax.bar(labels, seconds, color=["#527a9e", "#36678c", "#e28e3c"][:len(labels)])
    ax.bar_label(bars, fmt="%.1f s", padding=3)
    ax.set(title=f"Test computation time ({metrics.get('device', 'device unknown')})", ylabel="Seconds")
    ax.set_ylim(0, max(seconds) * 1.2 if max(seconds) > 0 else 1)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "test_computation_time.png", bbox_inches="tight")
    plt.close(fig)


def find_test_image(data_root: Path, image_id: str) -> Path | None:
    parent = data_root / "test_images"
    image_dir = parent / "test" if (parent / "test").is_dir() else parent
    for suffix in (".jpg", ".jpeg", ".png"):
        path = image_dir / f"{image_id}{suffix}"
        if path.is_file():
            return path
    return None


def plot_examples(predictions_path: Path, data_root: Path, count: int, output_dir: Path) -> None:
    with predictions_path.open(newline="", encoding="utf-8") as file:
        examples = list(csv.DictReader(file))[:count]
    if not examples:
        print(f"No caption rows found in {predictions_path}")
        return

    columns = 2
    rows = (len(examples) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(13, rows * 5), dpi=150, squeeze=False)
    for ax, item in zip(axes.flat, examples):
        image_path = find_test_image(data_root, item["image_id"])
        if image_path is None:
            ax.text(0.5, 0.5, f"Image not found:\n{item['image_id']}", ha="center", va="center")
            ax.axis("off")
            continue
        with Image.open(image_path) as image:
            ax.imshow(image.convert("RGB"))
        title = (
            f"{item['image_id']}\n"
            f"Reference: {textwrap.fill(item['reference'], 70)}\n"
            f"Generated: {textwrap.fill(item['generated'], 70)}"
        )
        ax.set_title(title, fontsize=9, loc="left", pad=8)
        ax.axis("off")
    for ax in list(axes.flat)[len(examples):]:
        ax.axis("off")
    fig.suptitle("ROCOv2 test examples: reference vs generated caption", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_dir / "qualitative_test_captions.png", bbox_inches="tight")
    plt.close(fig)


def write_summary(metrics: dict, output_path: Path) -> None:
    fields = [
        ("Test images", "samples"),
        ("Device", "device"),
        ("Evaluation loop time (s)", "evaluation_loop_seconds"),
        ("Teacher-forced forward time (s)", "model_forward_seconds"),
        ("Model compute time (s)", "model_compute_seconds"),
        ("Caption generation time (s)", "generation_seconds"),
        ("Evaluation throughput (images/s)", "images_per_second"),
        ("Evaluation time per image (s)", "seconds_per_image"),
        ("Token cross-entropy", "test_loss"),
        ("Teacher-forced token accuracy", "teacher_forced_token_accuracy"),
        ("Distinct caption rate", "distinct_caption_rate"),
        ("BLEU-1", "bleu1"), ("BLEU-2", "bleu2"),
        ("BLEU-3", "bleu3"), ("BLEU-4", "bleu4"),
        ("METEOR", "meteor"), ("ROUGE-L", "rougeL"),
    ]
    lines = ["# CNN+LSTM test results", "", "| Metric | Value |", "|---|---:|"]
    for label, key in fields:
        value = metrics.get(key)
        if value is None:
            shown = "not available"
        elif key in {"samples", "device"}:
            shown = str(value)
        else:
            shown = f"{value:.3f}" if key.endswith("seconds") or key == "images_per_second" else f"{value:.4f}"
        lines.append(f"| {label} | {shown} |")
    lines.extend([
        "",
        "Evaluation loop time includes test data loading, model calls, and batch decoding; it excludes checkpoint loading and corpus metric calculation. Model compute and caption generation times are measured with CUDA synchronization when running on GPU. Results vary with GPU load, batch size, data storage, and worker count.",
        "",
        "RMSE is not applicable to generated captions: token IDs are categorical labels, so numeric distances between IDs have no meaning.",
        "Teacher-forced token accuracy is included as a supplementary diagnostic; BLEU/METEOR/ROUGE and the example grid are more informative for caption quality.",
    ])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = args.artifact_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    history_path = args.artifact_dir / "metrics.jsonl"
    if history_path.is_file():
        history = read_jsonl(history_path)
        if history:
            plot_training(history, figures_dir)
            print(f"Plotted {len(history)} completed training epochs.")
    else:
        print(f"Training history not found: {history_path}")

    metrics_path = args.artifact_dir / "test_metrics.json"
    predictions_path = args.artifact_dir / "test_predictions.csv"
    if metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        plot_test_metrics(metrics, figures_dir)
        plot_timing(metrics, figures_dir)
        write_summary(metrics, args.artifact_dir / "results_summary.md")
        print(f"Wrote metrics summary: {args.artifact_dir / 'results_summary.md'}")
    else:
        print(f"Test metrics not found: {metrics_path}; run evaluate.py first.")

    if predictions_path.is_file():
        plot_examples(predictions_path, args.data_root, args.samples, figures_dir)
        print(f"Saved example grid in {figures_dir}")
    else:
        print(f"Test predictions not found: {predictions_path}; run evaluate.py with --output-csv first.")


if __name__ == "__main__":
    main()
