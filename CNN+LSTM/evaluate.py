"""Evaluate CNN+LSTM model on ROCO v2 test split with BLEU, METEOR, ROUGE-L, and test loss."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from torch.utils.data import DataLoader

from src.cnn_lstm import CNNLSTMCaptioner
from src.dataset import RoCoCaptionDataset
from src.losses import CaptionCrossEntropyLoss
from src.vocabulary import Vocabulary


def infer_default_data_root() -> Path | None:
    project_root = Path(__file__).resolve().parent
    candidates = [
        Path.cwd() / "DATA" / "rocov2",
        project_root / "DATA" / "rocov2",
        project_root.parent / "DATA" / "rocov2",
        Path(r"C:\AIE Files\Projects\S5\Dl\DATA\rocov2"),
    ]
    for candidate in candidates:
        if (candidate / "test_captions.csv").is_file() and (candidate / "test_images").is_dir():
            return candidate
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CNN+LSTM captioning model.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-length", type=int, default=48)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def decode_ids(token_ids: torch.Tensor, vocabulary: Vocabulary) -> list[str]:
    decoded: list[str] = []
    for sample in token_ids.tolist():
        decoded.append(vocabulary.decode(sample))
    return decoded


def main() -> None:
    args = parse_args()
    data_root = args.data_root or infer_default_data_root()
    if data_root is None:
        raise FileNotFoundError(
            "Could not locate ROCO v2 data root. Provide --data-root explicitly, e.g. "
            '"C:\\AIE Files\\Projects\\S5\\Dl\\DATA\\rocov2".'
        )
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    vocab_payload = checkpoint.get("vocabulary")
    if vocab_payload is None:
        vocabulary_path = args.checkpoint.parent / "vocabulary.json"
        if not vocabulary_path.is_file():
            raise FileNotFoundError(
                f"Vocabulary not found in checkpoint and missing at {vocabulary_path}. "
                "Use a checkpoint produced by train.py or restore vocabulary.json."
            )
        vocabulary = Vocabulary.load(vocabulary_path)
    else:
        vocabulary = Vocabulary(
            token_to_id=vocab_payload["token_to_id"],
            id_to_token=vocab_payload["id_to_token"],
            document_frequency=vocab_payload.get("document_frequency", {}),
        )

    model_config = checkpoint["model_config"]
    model = CNNLSTMCaptioner(**model_config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    test_set = RoCoCaptionDataset(data_root, "test", vocabulary, args.image_size, args.max_length, args.test_limit)
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    criterion = CaptionCrossEntropyLoss(pad_id=vocabulary.pad_id).to(device)

    total_loss = 0.0
    total_samples = 0
    references: list[str] = []
    hypotheses: list[str] = []

    with torch.no_grad():
        for images, captions, _ in test_loader:
            images = images.to(device)
            captions = captions.to(device)
            targets = captions[:, 1:]
            logits, _ = model(images, captions)
            loss, _ = criterion(logits, targets)
            batch_size = images.size(0)
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size

            generated_ids = model.generate(images, bos_id=vocabulary.bos_id, eos_id=vocabulary.eos_id, max_length=args.max_length)
            hypotheses.extend(decode_ids(generated_ids.cpu(), vocabulary))
            references.extend(decode_ids(captions.cpu(), vocabulary))

    smoothing = SmoothingFunction().method1
    ref_tokens = [[reference.split()] for reference in references]
    hyp_tokens = [hypothesis.split() for hypothesis in hypotheses]
    bleu1 = corpus_bleu(ref_tokens, hyp_tokens, weights=(1.0, 0.0, 0.0, 0.0), smoothing_function=smoothing)
    bleu2 = corpus_bleu(ref_tokens, hyp_tokens, weights=(0.5, 0.5, 0.0, 0.0), smoothing_function=smoothing)
    bleu3 = corpus_bleu(ref_tokens, hyp_tokens, weights=(1 / 3, 1 / 3, 1 / 3, 0.0), smoothing_function=smoothing)
    bleu4 = corpus_bleu(ref_tokens, hyp_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)

    meteor_scores = [
        meteor_score([reference.split()], hypothesis.split())
        for reference, hypothesis in zip(references, hypotheses)
    ]
    meteor = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0.0

    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_scores = [rouge.score(reference, hypothesis)["rougeL"].fmeasure for reference, hypothesis in zip(references, hypotheses)]
    rouge_l = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0

    metrics = {
        "test_loss": total_loss / max(total_samples, 1),
        "bleu1": bleu1,
        "bleu2": bleu2,
        "bleu3": bleu3,
        "bleu4": bleu4,
        "meteor": meteor,
        "rougeL": rouge_l,
        "samples": len(references),
    }
    print(json.dumps(metrics, indent=2))

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

