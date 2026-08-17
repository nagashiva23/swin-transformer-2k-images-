"""Greedy caption generation for CNN+LSTM model."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from src.cnn_lstm import CNNLSTMCaptioner
from src.dataset import RoCoCaptionDataset, build_image_transform
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
        if (candidate / "train_captions.csv").is_file():
            return candidate
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CNN+LSTM caption inference.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--image-path", type=Path, default=None)
    parser.add_argument("--split", type=str, choices=["train", "valid", "test"], default="test")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-length", type=int, default=48)
    return parser.parse_args()


def load_vocabulary(checkpoint: dict, checkpoint_path: Path) -> Vocabulary:
    payload = checkpoint.get("vocabulary")
    if payload is not None:
        return Vocabulary(
            token_to_id=payload["token_to_id"],
            id_to_token=payload["id_to_token"],
            document_frequency=payload.get("document_frequency", {}),
        )
    vocabulary_path = checkpoint_path.parent / "vocabulary.json"
    if not vocabulary_path.is_file():
        raise FileNotFoundError(
            f"Vocabulary not found in checkpoint and missing at {vocabulary_path}. "
            "Use a checkpoint produced by train.py or restore vocabulary.json."
        )
    return Vocabulary.load(vocabulary_path)


def load_image(image_path: Path, image_size: int) -> torch.Tensor:
    transform = build_image_transform(image_size)
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        tensor = transform(image)
    return tensor


def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    data_root = args.data_root or infer_default_data_root()
    if data_root is None and args.image_path is None:
        raise FileNotFoundError(
            "Could not locate ROCO v2 data root and no --image-path provided. "
            'Provide --data-root (e.g. "C:\\AIE Files\\Projects\\S5\\Dl\\DATA\\rocov2") '
            "or provide --image-path."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    vocabulary = load_vocabulary(checkpoint, args.checkpoint)

    model = CNNLSTMCaptioner(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    if args.image_path is None:
        dataset = RoCoCaptionDataset(
            data_root,
            args.split,
            vocabulary,
            image_size=args.image_size,
            max_length=args.max_length,
            limit=None,
        )
        image_tensor, reference_ids, image_id = dataset[args.index]
        reference_caption = vocabulary.decode(reference_ids.tolist())
        print(f"image_id: {image_id}")
        print(f"reference: {reference_caption}")
    else:
        image_tensor = load_image(args.image_path, args.image_size)
        image_id = args.image_path.stem
        print(f"image_id: {image_id}")

    with torch.no_grad():
        batch = image_tensor.unsqueeze(0).to(device)
        predicted_ids = model.generate(batch, bos_id=vocabulary.bos_id, eos_id=vocabulary.eos_id, max_length=args.max_length)
        predicted_caption = vocabulary.decode(predicted_ids[0].tolist())
    print(f"predicted: {predicted_caption}")


if __name__ == "__main__":
    main()

