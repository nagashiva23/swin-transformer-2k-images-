"""Evaluate both ROCO captioners and write comparable figures under result/."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import textwrap
import time
from collections import Counter
from contextlib import nullcontext
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
import torchvision.transforms as T
from PIL import Image

from caption_model import SwinCaptioningModel
from metrics import corpus_bleu, corpus_cider_d, corpus_rouge_l
from vocab import Vocab

ROOT = Path(__file__).resolve().parent
CNN_DIR = ROOT / "CNN+LSTM"
sys.path.insert(0, str(CNN_DIR))
from src.cnn_lstm import CNNLSTMCaptioner  # noqa: E402
from src.vocabulary import Vocabulary as CNNVocabulary  # noqa: E402

DEFAULT_DATA = Path(r"C:\AIE Files\Projects\S5\Dl\DATA\rocov2")
DEFAULT_SWIN_CKPT = DEFAULT_DATA / "swin_caption_best.pt"
DEFAULT_CNN_CKPT = CNN_DIR / "artifacts" / "cnn_lstm_roco_full" / "best.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def args_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("ROCO_ROOT", str(DEFAULT_DATA))))
    parser.add_argument("--swin-checkpoint", type=Path, default=DEFAULT_SWIN_CKPT)
    parser.add_argument("--cnn-checkpoint", type=Path, default=DEFAULT_CNN_CKPT)
    parser.add_argument("--limit", type=int, default=0, help="test rows; 0 evaluates the complete split")
    parser.add_argument("--batch-size", type=int, default=8, help="small default suitable for an 8 GB GPU")
    parser.add_argument("--top-tokens", type=int, default=20, help="number of common lexical token classes in confusion matrices")
    parser.add_argument("--examples", type=int, default=8)
    parser.add_argument("--feature-samples", type=int, default=256, help="images used for visual feature/PCA analysis")
    return parser.parse_args()


def read_test_rows(path: Path, limit: int) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        names = {name.lower(): name for name in (reader.fieldnames or [])}
        if "id" not in names or "caption" not in names:
            raise ValueError(f"Expected ID and Caption columns in {path}")
        rows = []
        for row in reader:
            rows.append({"ID": row[names["id"]], "Caption": row[names["caption"]]})
            if limit > 0 and len(rows) >= limit:
                break
    return rows


def image_path(images_dir: Path, image_id: str) -> Path | None:
    if Path(image_id).suffix:
        candidate = images_dir / image_id
        return candidate if candidate.is_file() else None
    for suffix in (".jpg", ".jpeg", ".png"):
        candidate = images_dir / f"{image_id}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def normalize_token(token: str) -> list[str]:
    if token.startswith("<") and token.endswith(">"):  # model control/unknown tokens
        return []
    return Vocab.tokenize(token)


def load_model(kind: str, checkpoint_path: Path):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"{kind} checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    if kind == "swin":
        vocabulary = Vocab.__new__(Vocab)
        vocabulary.itos = checkpoint["vocab_itos"]
        vocabulary.stoi = {word: index for index, word in enumerate(vocabulary.itos)}
        vocabulary.pad_token, vocabulary.sos_token = "<pad>", "<sos>"
        vocabulary.eos_token, vocabulary.unk_token = "<eos>", "<unk>"
        vocabulary.pad_id = vocabulary.stoi[vocabulary.pad_token]
        vocabulary.sos_id = vocabulary.stoi[vocabulary.sos_token]
        vocabulary.eos_id = vocabulary.stoi[vocabulary.eos_token]
        vocabulary.unk_id = vocabulary.stoi[vocabulary.unk_token]
        model = SwinCaptioningModel(vocab_size=len(vocabulary), max_len=40, pretrained=True).to(DEVICE)
        max_length = 40
        specials = {vocabulary.pad_id, vocabulary.sos_id, vocabulary.eos_id}
    else:
        vocab_data = checkpoint.get("vocabulary")
        if vocab_data is None:
            vocab_data = json.loads((checkpoint_path.parent / "vocabulary.json").read_text(encoding="utf-8"))
            vocab_data["token_to_id"] = {word: index for index, word in enumerate(vocab_data["id_to_token"])}
        vocabulary = CNNVocabulary(
            token_to_id=vocab_data["token_to_id"],
            id_to_token=vocab_data["id_to_token"],
            document_frequency=vocab_data.get("document_frequency", {}),
        )
        model = CNNLSTMCaptioner(**checkpoint["model_config"]).to(DEVICE)
        max_length = 48
        specials = {vocabulary.pad_id, vocabulary.bos_id, vocabulary.eos_id}
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, vocabulary, max_length, specials, checkpoint


def encode_caption(kind: str, vocabulary, caption: str, max_length: int) -> list[int]:
    if kind == "swin":
        return vocabulary.encode(caption, max_len=max_length)
    return vocabulary.encode(caption, max_length=max_length)


def model_forward(kind: str, model, images, captions):
    if kind == "swin":
        return model(images, captions, word_drop_p=0.0)
    logits, _ = model(images, captions)
    return logits


def generate(kind: str, model, images, vocabulary, max_length: int):
    if kind == "swin":
        return model.generate(images, vocabulary, max_len=max_length, device=DEVICE)
    return model.generate(images, bos_id=vocabulary.bos_id, eos_id=vocabulary.eos_id, max_length=max_length)


def decode(vocabulary, ids: torch.Tensor) -> str:
    return vocabulary.decode(ids.tolist())


def cuda_sync() -> None:
    if DEVICE.type == "cuda":
        torch.cuda.synchronize(DEVICE)


def autocast_context():
    if DEVICE.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def gpu_info() -> dict:
    if DEVICE.type != "cuda":
        return {"gpu_name": "N/A (CPU)", "gpu_memory_total_gib": None, "cuda_runtime": None}
    props = torch.cuda.get_device_properties(DEVICE)
    return {"gpu_name": props.name, "gpu_memory_total_gib": props.total_memory / (1024 ** 3),
            "cuda_runtime": torch.version.cuda}


def hardware_label(info: dict) -> str:
    memory = info.get("gpu_memory_total_gib")
    memory_text = f"{memory:.1f} GiB VRAM" if memory is not None else "VRAM unavailable"
    cuda_text = f"CUDA {info['cuda_runtime']}" if info.get("cuda_runtime") else "CUDA unavailable"
    return f"{info.get('gpu_name', 'Unknown device')} · {memory_text} · {cuda_text}"


def token_matrix_update(counter: Counter, logits, captions, vocabulary, specials: set[int]):
    predicted = logits.argmax(dim=-1).detach().cpu()
    targets = captions[:, 1:].detach().cpu()
    id_to_token = vocabulary.itos if hasattr(vocabulary, "itos") else vocabulary.id_to_token
    for gold_row, pred_row in zip(targets.tolist(), predicted.tolist()):
        for gold_id, pred_id in zip(gold_row, pred_row):
            if gold_id in specials:
                continue
            gold_tokens = normalize_token(id_to_token[gold_id])
            if not gold_tokens:
                continue
            pred_tokens = normalize_token(id_to_token[pred_id]) if 0 <= pred_id < len(id_to_token) else []
            gold = gold_tokens[0] if len(gold_tokens) == 1 else "<OTHER>"
            pred = pred_tokens[0] if len(pred_tokens) == 1 else "<OTHER>"
            counter[(gold, pred)] += 1


def make_confusion_outputs(counter: Counter, common_labels: list[str], out_dir: Path, prefix: str) -> dict:
    labels = common_labels + ["<OTHER>"]
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    index = {label: i for i, label in enumerate(labels)}
    for (gold, pred), count in counter.items():
        matrix[index.get(gold, index["<OTHER>"]), index.get(pred, index["<OTHER>"])] += count
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_sums, out=np.zeros_like(matrix, dtype=np.float64), where=row_sums != 0)

    with (out_dir / f"{prefix}_token_confusion_counts.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["actual\\predicted", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *row.tolist()])
    with (out_dir / f"{prefix}_token_confusion_row_normalized.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["actual\\predicted", *labels])
        for label, row in zip(labels, normalized):
            writer.writerow([label, *[round(float(value), 6) for value in row]])

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.62), max(8, len(labels) * 0.52)), dpi=180)
    image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.set_xlabel("Predicted token class")
    ax.set_ylabel("Reference token class")
    ax.set_title(f"{prefix.upper()} token confusion matrix (row normalized)")
    fig.colorbar(image, ax=ax, label="Fraction within reference token class")
    for i in range(len(labels)):
        for j in range(len(labels)):
            if normalized[i, j] >= 0.025:
                ax.text(j, i, f"{normalized[i, j]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if normalized[i, j] > 0.55 else "black")
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_token_confusion_matrix.png", bbox_inches="tight")
    plt.close(fig)

    total = int(matrix.sum())
    exact_tokens = sum(count for (gold, pred), count in counter.items() if gold == pred)
    report = []
    full_labels = sorted({token for pair in counter for token in pair if token != "<OTHER>"}) + ["<OTHER>"]
    actual_counts = Counter()
    predicted_counts = Counter()
    true_positives = Counter()
    for (gold, pred), count in counter.items():
        actual_counts[gold] += count
        predicted_counts[pred] += count
        if gold == pred:
            true_positives[gold] += count
    for label in full_labels:
        tp = true_positives[label]
        support = actual_counts[label]
        predicted_count = predicted_counts[label]
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        report.append({"token_class": label, "support": support, "precision": precision, "recall": recall, "f1": f1})
    with (out_dir / f"{prefix}_token_class_report.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=["token_class", "support", "precision", "recall", "f1"])
        writer.writeheader()
        writer.writerows(report)
    return {"token_accuracy_exact_lexical": float(exact_tokens / total) if total else 0.0,
            "token_matrix_total_lexical_positions": total, "token_classes": labels}


def draw_architecture(kind: str, out_dir: Path) -> None:
    if kind == "swin":
        title = "Swin Transformer + Transformer decoder"
        blocks = [
            ("ROCO image", "RGB 224 × 224"),
            ("Swin-T image encoder", "patch embedding → shifted-window stages"),
            ("Spatial visual tokens", "7 × 7 × 768 = 49 tokens"),
            ("Autoregressive Transformer decoder", "6 layers · causal self-attention + cross-attention"),
            ("Word distribution", "vocabulary logits → greedy / beam caption"),
        ]
        prefix = "Swin"
    else:
        title = "ResNet-50 + LSTM"
        blocks = [
            ("ROCO image", "RGB 224 × 224"),
            ("ImageNet ResNet-50", "convolution stages + global average pooling"),
            ("Projected visual vector", "2048 → 512 · initializes LSTM hidden and cell states"),
            ("Word embedding + LSTM", "previous tokens → recurrent hidden state"),
            ("Word distribution", "vocabulary logits → greedy caption"),
        ]
        prefix = "CNN-LSTM"
    fig, ax = plt.subplots(figsize=(11, 8), dpi=180)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, len(blocks) + 0.6)
    ax.axis("off")
    ax.set_title(f"{title}: model computation flow", fontsize=16, pad=14)
    colors = ["#edf4fb", "#d7e9f8", "#c8e6df", "#fff0c2", "#f7d8d8"]
    for i, (name, detail) in enumerate(blocks):
        y = len(blocks) - i - 0.1
        ax.text(0.5, y, f"{name}\n{detail}", ha="center", va="center", fontsize=10,
                bbox={"boxstyle": "round,pad=0.65", "facecolor": colors[i], "edgecolor": "#496273", "linewidth": 1.2})
        if i < len(blocks) - 1:
            ax.annotate("", xy=(0.5, y - 0.48), xytext=(0.5, y - 0.78),
                        arrowprops={"arrowstyle": "-|>", "color": "#34495e", "lw": 1.5})
    fig.tight_layout()
    fig.savefig(out_dir / "computation_graph.png", bbox_inches="tight")
    plt.close(fig)


def plot_training_curve(kind: str, out_dir: Path) -> None:
    if kind == "swin":
        path = ROOT / "training_history.csv"
        if not path.is_file():
            return
        with path.open(newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        train = [float(row["train_loss"]) for row in rows]
        valid = [float(row["val_loss"]) for row in rows]
    else:
        path = CNN_DIR / "artifacts" / "cnn_lstm_roco_full" / "metrics.jsonl"
        if not path.is_file():
            return
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        train = [float(row["train_loss"]) for row in rows]
        valid = [float(row["valid_loss"]) for row in rows]
    epochs = range(1, len(train) + 1)
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=180)
    ax.plot(epochs, train, "o-", label="Train loss")
    ax.plot(epochs, valid, "s-", label="Validation loss")
    ax.set(title=f"{kind.upper()} training history", xlabel="Epoch", ylabel="Token cross-entropy")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "training_validation_loss.png", bbox_inches="tight")
    plt.close(fig)


def plot_examples(rows: list[dict], root: Path, count: int, out_path: Path) -> None:
    examples = rows[:count]
    if not examples:
        return
    cols = 2
    fig, axes = plt.subplots(math.ceil(len(examples) / cols), cols,
                             figsize=(13, math.ceil(len(examples) / cols) * 5.2), dpi=150, squeeze=False)
    images_dir = root / "test_images" / "test"
    for ax, row in zip(axes.flat, examples):
        path = image_path(images_dir, row["image_id"])
        if path:
            with Image.open(path) as image:
                ax.imshow(image.convert("RGB"))
        else:
            ax.text(0.5, 0.5, f"Missing image: {row['image_id']}", ha="center", va="center")
        ax.set_title(f"{row['image_id']}\nGT: {textwrap.fill(row['reference'], 64)}\nPred: {textwrap.fill(row['generated'], 64)}",
                     loc="left", fontsize=8)
        ax.axis("off")
    for ax in list(axes.flat)[len(examples):]:
        ax.axis("off")
    fig.suptitle("Held-out ROCO test samples", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_runtime(metrics: dict, out_dir: Path) -> None:
    labels = ["Teacher-forced\nforward", "Greedy\ngeneration"]
    values = [metrics["teacher_forced_forward_seconds"], metrics["caption_generation_seconds"]]
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=180)
    bars = ax.bar(labels, values, color=["#547b9d", "#dd9148"])
    ax.bar_label(bars, fmt="%.2f s", padding=3)
    ax.set_title(f"{metrics['model']} model time ({metrics['samples_evaluated']} images)\n{hardware_label(metrics)}")
    ax.set_ylabel("Seconds")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "computation_time.png", bbox_inches="tight")
    plt.close(fig)


def write_feature_analysis(kind: str, model, vocabulary, prediction_rows: list[dict], data_root: Path,
                           common_labels: list[str], sample_limit: int, batch_size: int,
                           out_dir: Path, hardware: dict) -> float:
    if sample_limit <= 0:
        return 0.0
    selected = prediction_rows[:sample_limit]
    vectors, records, activation_maps = [], [], []
    images_dir = data_root / "test_images" / "test"
    started = time.perf_counter()
    for start in range(0, len(selected), batch_size):
        batch_records, tensors = [], []
        for row in selected[start:start + batch_size]:
            path = image_path(images_dir, row["image_id"])
            if path is None:
                continue
            with Image.open(path) as image:
                tensors.append(IMAGE_TRANSFORM(image.convert("RGB")))
            batch_records.append(row)
        if not tensors:
            continue
        images = torch.stack(tensors).to(DEVICE, non_blocking=DEVICE.type == "cuda")
        with torch.inference_mode(), autocast_context():
            if kind == "swin":
                spatial = model.encoder(images)  # B x 49 x 768
                feature_vectors = spatial.mean(dim=1)
                activation = spatial.norm(dim=-1).reshape(-1, 7, 7)
            else:
                spatial = model.encoder.backbone[:-1](images)  # B x 2048 x 7 x 7 before global pooling
                feature_vectors = spatial.mean(dim=(2, 3))
                activation = spatial.norm(dim=1)
        feature_batch = feature_vectors.float().cpu().numpy()
        vectors.extend(feature_batch)
        for row, heatmap, feature_vector in zip(batch_records, activation.float().cpu().numpy(), feature_batch):
            words = set(Vocab.tokenize(row["reference"]))
            label = next((word for word in common_labels[:10] if word in words), "other")
            records.append({"image_id": row["image_id"], "caption": row["reference"], "keyword": label,
                            "feature_norm": float(np.linalg.norm(feature_vector))})
            activation_maps.append(heatmap)

    if not vectors:
        return time.perf_counter() - started
    feature_array = np.asarray(vectors, dtype=np.float32)
    centered = feature_array - feature_array.mean(axis=0, keepdims=True)
    # Exact two-component SVD is inexpensive at this bounded sample count and needs no sklearn dependency.
    u, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    coordinates = u[:, :2] * singular_values[:2]
    if coordinates.shape[1] < 2:
        coordinates = np.pad(coordinates, ((0, 0), (0, 2 - coordinates.shape[1])))
    variance = singular_values ** 2
    explained = variance[:2].sum() / max(variance.sum(), 1e-12)
    fig = go.Figure()
    labels = sorted({record["keyword"] for record in records})
    for label in labels:
        indices = [i for i, record in enumerate(records) if record["keyword"] == label]
        fig.add_trace(go.Scatter(
            x=coordinates[indices, 0], y=coordinates[indices, 1], mode="markers", name=label,
            text=[records[i]["image_id"] for i in indices],
            customdata=[[records[i]["caption"], records[i]["feature_norm"]] for i in indices],
            hovertemplate="ID: %{text}<br>Caption: %{customdata[0]}<br>Feature norm: %{customdata[1]:.3f}<extra></extra>",
            marker={"size": 7, "opacity": 0.75},
        ))
    fig.update_layout(title=f"{kind.upper()} visual feature PCA · {hardware['gpu_name']} · 2 PCs explain {explained:.1%}",
                      xaxis_title="Principal component 1", yaxis_title="Principal component 2",
                      legend_title="Caption keyword proxy", hovermode="closest")
    fig.write_html(out_dir / "interactive_feature_pca.html", include_plotlyjs="directory")

    hist = go.Figure()
    for label in labels:
        norms = [record["feature_norm"] for record in records if record["keyword"] == label]
        hist.add_trace(go.Histogram(x=norms, name=label, opacity=0.65, nbinsx=30))
    hist.update_layout(title=f"{kind.upper()} visual feature-vector norms · {hardware['gpu_name']}",
                       xaxis_title="L2 norm of pooled image feature", yaxis_title="Images", barmode="overlay")
    hist.write_html(out_dir / "interactive_feature_norms.html", include_plotlyjs="directory")

    sample_count = min(6, len(activation_maps))
    cols = 3
    fig, axes = plt.subplots(math.ceil(sample_count / cols), cols, figsize=(12, 3.8 * math.ceil(sample_count / cols)),
                             dpi=170, squeeze=False)
    for index in range(sample_count):
        ax = axes.flat[index]
        path = image_path(images_dir, records[index]["image_id"])
        if path:
            with Image.open(path) as image:
                ax.imshow(image.convert("RGB"))
            ax.imshow(activation_maps[index], cmap="magma", alpha=0.42,
                      extent=(0, 224, 224, 0), interpolation="bilinear")
        ax.set_title(f"{records[index]['image_id']} · {records[index]['keyword']}", fontsize=8)
        ax.axis("off")
    for ax in list(axes.flat)[sample_count:]:
        ax.axis("off")
    fig.suptitle(f"{kind.upper()} spatial feature activation norm (not a clinical saliency map)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "feature_activation_examples.png", bbox_inches="tight")
    plt.close(fig)
    return time.perf_counter() - started


def write_interactive_training_history(kind: str, output_path: Path, hardware: str) -> None:
    if kind == "swin":
        history_path = ROOT / "training_history.json"
        if not history_path.is_file():
            return
        rows = json.loads(history_path.read_text(encoding="utf-8"))
        train_key, valid_key = "train_loss", "val_loss"
    else:
        history_path = CNN_DIR / "artifacts" / "cnn_lstm_roco_full" / "metrics.jsonl"
        if not history_path.is_file():
            return
        rows = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        train_key, valid_key = "train_loss", "valid_loss"
    epochs = [row["epoch"] for row in rows]
    fig = go.Figure()
    for key, label in ((train_key, "Train loss"), (valid_key, "Validation loss")):
        fig.add_trace(go.Scatter(x=epochs, y=[row.get(key) for row in rows], mode="lines+markers", name=label,
                                 hovertemplate="Epoch %{x}<br>Loss %{y:.4f}<extra></extra>"))
    fig.update_layout(title=f"{kind.upper()} training history · {hardware}", xaxis_title="Epoch",
                      yaxis_title="Cross-entropy loss", hovermode="x unified")
    fig.write_html(output_path, include_plotlyjs="directory")


def write_epoch_runtime_comparison(output_path: Path, hardware: str) -> None:
    swin_path = ROOT / "training_history.json"
    cnn_path = CNN_DIR / "artifacts" / "cnn_lstm_roco_full" / "metrics.jsonl"
    swin_rows = json.loads(swin_path.read_text(encoding="utf-8")) if swin_path.is_file() else []
    cnn_rows = ([json.loads(line) for line in cnn_path.read_text(encoding="utf-8").splitlines() if line.strip()]
                if cnn_path.is_file() else [])
    fig = go.Figure()
    for name, rows in (("Swin Transformer", swin_rows), ("CNN+LSTM", cnn_rows)):
        epochs = [row["epoch"] for row in rows]
        times = [row.get("epoch_seconds") for row in rows]
        per_gpu = sorted({row.get("gpu_name") for row in rows if row.get("gpu_name")})
        label = f"{name} · {', '.join(per_gpu) if per_gpu else 'historic timing not logged'}"
        fig.add_trace(go.Scatter(x=epochs, y=times, mode="lines+markers", name=label, connectgaps=False,
                                 hovertemplate="Epoch %{x}<br>Epoch duration %{y:.1f} s<extra></extra>"))
    fig.update_layout(title=f"Training duration per epoch · {hardware}",
                      xaxis_title="Training epoch", yaxis_title="Wall time (seconds)",
                      annotations=[{"text": "Earlier training histories did not save epoch duration. Timing points appear for epochs logged after the timer update.",
                                    "xref": "paper", "yref": "paper", "x": 0.5, "y": 1.12,
                                    "showarrow": False, "font": {"size": 11}}], hovermode="x unified")
    fig.write_html(output_path, include_plotlyjs="directory")


def write_interactive_comparison(results: list[dict], output_path: Path) -> None:
    fig = make_subplots(rows=2, cols=2, subplot_titles=("Caption metrics", "Model compute latency",
                                                        "Allocated GPU memory", "Evaluation throughput"))
    for result in results:
        model = result["model"]
        device_label = result.get("gpu_name", result["device"])
        fig.add_trace(go.Bar(x=["BLEU-1", "BLEU-4", "ROUGE-L", "CIDEr-D"],
                             y=[result["bleu1"], result["bleu4"], result["rougeL"], result["ciderD"]],
                             name=model, legendgroup=model,
                             hovertemplate=f"{model}<br>%{{x}}: %{{y:.4f}}<extra>{device_label}</extra>"), row=1, col=1)
        fig.add_trace(go.Bar(x=[model], y=[result["model_compute_ms_per_image"]], name=model,
                             legendgroup=model, showlegend=False,
                             hovertemplate=f"{model}<br>%{{y:.2f}} ms/image<extra>{device_label}</extra>"), row=1, col=2)
        memory = result.get("peak_gpu_memory_gib")
        if memory is not None:
            fig.add_trace(go.Bar(x=[model], y=[memory], name=model, legendgroup=model, showlegend=False,
                                 hovertemplate=f"{model}<br>%{{y:.2f}} GiB allocated<extra>{device_label}</extra>"), row=2, col=1)
        fig.add_trace(go.Bar(x=[model], y=[result["images_per_second_including_load"]], name=model,
                             legendgroup=model, showlegend=False,
                             hovertemplate=f"{model}<br>%{{y:.1f}} images/s<extra>{device_label}</extra>"), row=2, col=2)
    hw = hardware_label(results[0])
    fig.update_layout(title=f"ROCOv2 model comparison · {hw} · same test rows",
                      barmode="group", height=760, hovermode="closest")
    fig.update_yaxes(title_text="Score", row=1, col=1)
    fig.update_yaxes(title_text="Milliseconds / image", row=1, col=2)
    fig.update_yaxes(title_text="GiB allocated", row=2, col=1)
    fig.update_yaxes(title_text="Images / second", row=2, col=2)
    fig.write_html(output_path, include_plotlyjs="directory")


def plot_prediction_comparison(swin_rows: list[dict], cnn_rows: list[dict], root: Path,
                               count: int, out_path: Path) -> None:
    cnn_by_id = {row["image_id"]: row for row in cnn_rows}
    paired = [(row, cnn_by_id[row["image_id"]]) for row in swin_rows if row["image_id"] in cnn_by_id][:count]
    if not paired:
        return
    fig, axes = plt.subplots(len(paired), 2, figsize=(13, max(5, len(paired) * 4.4)),
                             dpi=150, squeeze=False, gridspec_kw={"width_ratios": [1, 1.5]})
    images_dir = root / "test_images" / "test"
    for index, (swin_row, cnn_row) in enumerate(paired):
        image_ax, text_ax = axes[index]
        path = image_path(images_dir, swin_row["image_id"])
        if path:
            with Image.open(path) as image:
                image_ax.imshow(image.convert("RGB"))
        image_ax.set_title(swin_row["image_id"], fontsize=9)
        image_ax.axis("off")
        text_ax.axis("off")
        text_ax.text(0, 0.82, "Reference", weight="bold", fontsize=9, transform=text_ax.transAxes)
        text_ax.text(0, 0.67, textwrap.fill(swin_row["reference"], 68), fontsize=8, transform=text_ax.transAxes)
        text_ax.text(0, 0.43, "Swin", weight="bold", color="#315f8a", fontsize=9, transform=text_ax.transAxes)
        text_ax.text(0, 0.29, textwrap.fill(swin_row["generated"], 68), fontsize=8, transform=text_ax.transAxes)
        text_ax.text(0, 0.12, "CNN+LSTM", weight="bold", color="#b56b25", fontsize=9, transform=text_ax.transAxes)
        text_ax.text(0, -0.02, textwrap.fill(cnn_row["generated"], 68), fontsize=8, transform=text_ax.transAxes)
    fig.suptitle("Same ROCO test images: reference and both model predictions", fontsize=14)
    fig.tight_layout(rect=(0, 0.01, 1, 0.97))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def evaluate_model(kind: str, checkpoint_path: Path, rows: list[dict], common_labels: list[str], args) -> dict:
    out_dir = ROOT / "result" / "swin" if kind == "swin" else CNN_DIR / "result"
    out_dir.mkdir(parents=True, exist_ok=True)
    hardware = gpu_info()
    draw_architecture(kind, out_dir)
    plot_training_curve(kind, out_dir)
    write_interactive_training_history(kind, out_dir / "interactive_training_history.html", hardware_label(hardware))

    model, vocabulary, max_length, specials, checkpoint = load_model(kind, checkpoint_path)
    model_parameters = sum(parameter.numel() for parameter in model.parameters())
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats(DEVICE)
    # A short untimed warm-up avoids counting CUDA's first-call initialization in latency.
    dummy_images = torch.zeros((1, 3, 224, 224), device=DEVICE)
    pad_id = vocabulary.pad_id
    dummy_caption = torch.full((1, max_length), pad_id, dtype=torch.long, device=DEVICE)
    if kind == "swin":
        dummy_caption[0, 0] = vocabulary.sos_id
        dummy_caption[0, 1] = vocabulary.eos_id
    else:
        dummy_caption[0, 0] = vocabulary.bos_id
        dummy_caption[0, 1] = vocabulary.eos_id
    with torch.inference_mode():
        with autocast_context():
            _ = model_forward(kind, model, dummy_images, dummy_caption)
        _ = generate(kind, model, dummy_images, vocabulary, 4)
    cuda_sync()
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats(DEVICE)

    metrics = {"model": "Swin Transformer" if kind == "swin" else "CNN+LSTM",
               "checkpoint": str(checkpoint_path), "checkpoint_epoch": checkpoint.get("epoch"),
               "device": str(DEVICE), "samples_requested": len(rows), "parameter_count": model_parameters,
               "decoding": "greedy", "batch_size": args.batch_size, "amp": DEVICE.type == "cuda",
               **hardware}
    refs, hyps, prediction_rows = [], [], []
    matrix_counter: Counter = Counter()
    images_dir = args.data_root / "test_images" / "test"
    skipped = 0
    eval_start = time.perf_counter()
    forward_seconds = 0.0
    generation_seconds = 0.0
    for start in range(0, len(rows), args.batch_size):
        batch_rows = []
        image_tensors = []
        for row in rows[start:start + args.batch_size]:
            path = image_path(images_dir, row["ID"])
            if path is None:
                skipped += 1
                continue
            try:
                with Image.open(path) as image:
                    image_tensors.append(IMAGE_TRANSFORM(image.convert("RGB")))
                batch_rows.append(row)
            except (OSError, ValueError):
                skipped += 1
        if not batch_rows:
            continue
        images = torch.stack(image_tensors).to(DEVICE, non_blocking=DEVICE.type == "cuda")
        caption_ids = [encode_caption(kind, vocabulary, row["Caption"], max_length) for row in batch_rows]
        captions = torch.tensor(caption_ids, dtype=torch.long, device=DEVICE)

        cuda_sync()
        forward_start = time.perf_counter()
        with torch.inference_mode():
            with autocast_context():
                logits = model_forward(kind, model, images, captions)
        cuda_sync()
        forward_seconds += time.perf_counter() - forward_start
        token_matrix_update(matrix_counter, logits, captions, vocabulary, specials)

        cuda_sync()
        generation_start = time.perf_counter()
        with torch.inference_mode(), autocast_context():
            generated_ids = generate(kind, model, images, vocabulary, max_length)
        cuda_sync()
        generation_seconds += time.perf_counter() - generation_start

        for row, ids in zip(batch_rows, generated_ids.detach().cpu()):
            reference = row["Caption"]
            generated = decode(vocabulary, ids)
            refs.append(Vocab.tokenize(reference))
            hyps.append(Vocab.tokenize(generated))
            prediction_rows.append({"image_id": row["ID"], "reference": reference, "generated": generated})
    cuda_sync()
    loop_seconds = time.perf_counter() - eval_start
    if not refs:
        raise RuntimeError(f"No test images were processed for {kind}; verify {images_dir}")

    bleu = corpus_bleu(refs, hyps)
    rouge = corpus_rouge_l(refs, hyps)
    cider = corpus_cider_d(refs, hyps)
    confusion_metrics = make_confusion_outputs(matrix_counter, common_labels, out_dir, kind)
    metrics.update(bleu)
    metrics.update({"rougeL": rouge, "ciderD": cider,
                    "distinct_caption_rate": len({row["generated"] for row in prediction_rows}) / len(prediction_rows),
                    "samples_evaluated": len(prediction_rows), "missing_or_unreadable_images": skipped,
                    "evaluation_loop_seconds": loop_seconds, "teacher_forced_forward_seconds": forward_seconds,
                    "caption_generation_seconds": generation_seconds,
                    "model_compute_seconds": forward_seconds + generation_seconds,
                    "images_per_second_including_load": len(prediction_rows) / max(loop_seconds, 1e-12),
                    "model_compute_ms_per_image": 1000 * (forward_seconds + generation_seconds) / len(prediction_rows),
                    "peak_gpu_memory_gib": torch.cuda.max_memory_allocated(DEVICE) / (1024 ** 3) if DEVICE.type == "cuda" else None})
    metrics.update(confusion_metrics)
    metrics["feature_analysis_seconds"] = write_feature_analysis(
        kind, model, vocabulary, prediction_rows, args.data_root, common_labels,
        args.feature_samples, args.batch_size, out_dir, hardware,
    )
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (out_dir / "predictions.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=["image_id", "reference", "generated"])
        writer.writeheader()
        writer.writerows(prediction_rows)
    plot_examples(prediction_rows, args.data_root, args.examples, out_dir / "qualitative_examples.png")
    plot_runtime(metrics, out_dir)
    total_vram = metrics["gpu_memory_total_gib"]
    vram_description = "unknown VRAM" if total_vram is None else f"{total_vram:.2f} GiB total"
    summary = [f"# {metrics['model']} results", "", "## Evaluation settings", "",
               f"- Checkpoint epoch: {metrics['checkpoint_epoch']}", f"- Device: {metrics['device']}",
               f"- GPU: {metrics['gpu_name']} ({vram_description})",
               f"- Samples: {metrics['samples_evaluated']} of {metrics['samples_requested']}",
               f"- Decoding: greedy; autocast: {metrics['amp']}",
               "- Confusion matrix: teacher-forced token classification; same frequent word classes for both models, with all other words grouped as OTHER.",
               "- `*_token_class_report.csv` contains per-token precision/recall/F1 for all observed lexical token classes.", "",
               "## Caption metrics", "", "| Metric | Score |", "|---|---:|"]
    for name in ("bleu1", "bleu2", "bleu3", "bleu4", "rougeL", "ciderD", "distinct_caption_rate"):
        summary.append(f"| {name} | {metrics[name]:.4f} |")
    summary.extend(["", "## Runtime", "", "| Measurement | Result |", "|---|---:|"])
    for name, label in (("evaluation_loop_seconds", "Loop time, includes image loading"),
                        ("teacher_forced_forward_seconds", "Teacher-forced forward"),
                        ("caption_generation_seconds", "Greedy generation"),
                        ("model_compute_seconds", "Combined model compute"),
                        ("feature_analysis_seconds", "Feature analysis (extra forward passes)"),
                        ("images_per_second_including_load", "Images/second including loading"),
                        ("model_compute_ms_per_image", "Model compute ms/image"),
                        ("peak_gpu_memory_gib", "Peak allocated GPU memory (GiB)")):
        value = metrics.get(name)
        summary.append(f"| {label} | {'N/A (CPU)' if value is None else f'{value:.3f}'} |")
    summary.extend(["", "Model compute times exclude checkpoint loading and metric computation. GPU synchronization is used around timed model calls.",
                    "The confusion matrix is over next-word token classes, not image diagnosis classes. It is teacher-forced and should be interpreted as a language/token diagnostic, not clinical classification accuracy."])
    (out_dir / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    del model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return metrics


def plot_comparison(results: list[dict], comparison_dir: Path) -> None:
    labels = [result["model"] for result in results]
    metric_names = [("BLEU-1", "bleu1"), ("BLEU-4", "bleu4"), ("ROUGE-L", "rougeL"), ("CIDEr-D", "ciderD")]
    x = np.arange(len(metric_names))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 5), dpi=180)
    for i, result in enumerate(results):
        values = [result[key] for _, key in metric_names]
        bars = ax.bar(x + (i - 0.5) * width, values, width, label=labels[i])
        ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)
    ax.set_xticks(x, [name for name, _ in metric_names])
    ax.set_title(f"Caption quality on the same ROCO test images\n{hardware_label(results[0])}")
    ax.set_ylabel("Score")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(comparison_dir / "model_caption_metrics_comparison.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=180)
    times = [result["model_compute_ms_per_image"] for result in results]
    bars = ax.bar(labels, times, color=["#567b9d", "#dd9148"][:len(labels)])
    ax.bar_label(bars, fmt="%.2f ms/image", padding=3)
    ax.set_title(f"Model compute time per image (teacher-forced + generation)\n{hardware_label(results[0])}")
    ax.set_ylabel("Milliseconds per image")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(comparison_dir / "model_compute_time_comparison.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = args_parser()
    if args.batch_size < 1 or args.top_tokens < 1:
        raise ValueError("--batch-size and --top-tokens must be positive")
    if args.feature_samples < 0 or args.examples < 0:
        raise ValueError("--feature-samples and --examples cannot be negative")
    root_result = ROOT / "result"
    (root_result / "swin").mkdir(parents=True, exist_ok=True)
    cnn_result = CNN_DIR / "result"
    cnn_result.mkdir(parents=True, exist_ok=True)
    comparison_dir = cnn_result / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.data_root / "test_captions.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"ROCO test captions not found: {csv_path}")
    rows = read_test_rows(csv_path, args.limit)
    common_counts = Counter(token for row in rows for token in Vocab.tokenize(row["Caption"]))
    common_labels = [token for token, _ in common_counts.most_common(args.top_tokens)]
    if not rows or not common_labels:
        raise ValueError("No test captions/words found")

    print(f"Device: {DEVICE}; test rows selected: {len(rows)}; batch size: {args.batch_size}")
    results = [
        evaluate_model("swin", args.swin_checkpoint, rows, common_labels, args),
        evaluate_model("cnn_lstm", args.cnn_checkpoint, rows, common_labels, args),
    ]
    plot_comparison(results, comparison_dir)
    hw_name = hardware_label(gpu_info())
    write_interactive_comparison(results, comparison_dir / "interactive_model_dashboard.html")
    write_epoch_runtime_comparison(comparison_dir / "interactive_epoch_runtime.html", hw_name)
    predictions = []
    for kind in ("swin", "cnn_lstm"):
        prediction_dir = root_result / "swin" if kind == "swin" else cnn_result
        with (prediction_dir / "predictions.csv").open(newline="", encoding="utf-8-sig") as file:
            predictions.append(list(csv.DictReader(file)))
    plot_prediction_comparison(predictions[0], predictions[1], args.data_root, args.examples,
                               comparison_dir / "qualitative_model_comparison.png")
    with (comparison_dir / "model_comparison.csv").open("w", newline="", encoding="utf-8-sig") as file:
        fields = ["model", "device", "gpu_name", "gpu_memory_total_gib", "samples_evaluated", "parameter_count",
                  "bleu1", "bleu4", "rougeL", "ciderD",
                  "model_compute_seconds", "model_compute_ms_per_image", "images_per_second_including_load",
                  "peak_gpu_memory_gib"]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: result.get(field) for field in fields} for result in results)
    table = ["# Swin vs CNN+LSTM on ROCO v2", "", f"Test samples: {results[0]['samples_evaluated']} per model. Greedy decoding is used for both.",
             f"Hardware: {results[0]['gpu_name']} · CUDA {results[0]['cuda_runtime']} · {results[0]['gpu_memory_total_gib'] or 'N/A'} GiB total VRAM.", "",
             "| Model | Parameters (M) | BLEU-1 | BLEU-4 | ROUGE-L | CIDEr-D | Model ms/image | Images/s incl. load | Peak GPU GiB |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for result in results:
        memory = "N/A" if result["peak_gpu_memory_gib"] is None else f"{result['peak_gpu_memory_gib']:.2f}"
        table.append(f"| {result['model']} | {result['parameter_count'] / 1e6:.2f} | {result['bleu1']:.4f} | {result['bleu4']:.4f} | {result['rougeL']:.4f} | {result['ciderD']:.4f} | {result['model_compute_ms_per_image']:.2f} | {result['images_per_second_including_load']:.2f} | {memory} |")
    table.extend(["", "Both models were evaluated using the same test rows, image preprocessing, reference text, and metric implementation. Captioning is not image classification: the per-model confusion matrices classify next-token words under teacher forcing and are diagnostic only.", "", "Swin-only outputs are in `result/swin/`. CNN+LSTM-only outputs are kept in `CNN+LSTM/result/`. This combined report and comparison dashboard are in `CNN+LSTM/result/comparison/`."])
    (comparison_dir / "model_comparison.md").write_text("\n".join(table) + "\n", encoding="utf-8")
    print(f"Results written under: {root_result} and {CNN_DIR / 'result'}")


if __name__ == "__main__":
    main()
