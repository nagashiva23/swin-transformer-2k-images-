"""Train ResNet50 + LSTM captioning model on ROCO v2."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.cnn_lstm import CNNLSTMCaptioner
from src.dataset import RoCoCaptionDataset
from src.losses import CaptionCrossEntropyLoss
from src.vocabulary import Vocabulary, read_captions


def infer_default_data_root() -> Path | None:
    project_root = Path(__file__).resolve().parent
    candidates = [
        Path.cwd() / "DATA" / "rocov2",
        project_root / "DATA" / "rocov2",
        project_root.parent / "DATA" / "rocov2",
        Path(r"C:\AIE Files\Projects\S5\Dl\DATA\rocov2"),
    ]
    for candidate in candidates:
        if (candidate / "train_captions.csv").is_file() and (candidate / "train_images").is_dir():
            return candidate
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CNN+LSTM captioning model.")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/cnn_lstm_pipeline"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-length", type=int, default=48)
    parser.add_argument("--max-vocab", type=int, default=8000)
    parser.add_argument("--embedding-dim", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--feature-dim", type=int, default=512)
    parser.add_argument("--lstm-layers", type=int, default=1)
    parser.add_argument("--decoder-dropout", type=float, default=0.1)
    parser.add_argument("--unfreeze-encoder", action="store_true")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--valid-limit", type=int, default=None)
    parser.add_argument("--checkpoint-every-steps", type=int, default=2000)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def run_epoch(
    model: CNNLSTMCaptioner,
    loader: DataLoader,
    criterion: CaptionCrossEntropyLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    training: bool,
    start_step: int = 0,
    checkpoint_every_steps: int = 0,
    on_step_checkpoint=None,
) -> tuple[float, int]:
    model.train(training)
    total_loss, total_items = 0.0, 0
    global_step = start_step
    progress = tqdm(loader, leave=False, desc="train" if training else "valid")

    for images, captions, _ in progress:
        images = images.to(device)
        captions = captions.to(device)
        targets = captions[:, 1:]

        with torch.set_grad_enabled(training):
            logits, _ = model(images, captions)
            loss, details = criterion(logits, targets)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                global_step += 1
                if (
                    checkpoint_every_steps > 0
                    and on_step_checkpoint is not None
                    and global_step % checkpoint_every_steps == 0
                ):
                    on_step_checkpoint(global_step, float(loss.item()))

        batch_size = images.size(0)
        total_loss += float(loss.item()) * batch_size
        total_items += batch_size
        progress.set_postfix(loss=f"{loss.item():.4f}", caption=f"{details['caption']:.4f}")

    return total_loss / max(total_items, 1), global_step


def main() -> None:
    args = parse_args()
    data_root = args.data_root or infer_default_data_root()
    if data_root is None:
        raise FileNotFoundError(
            "Could not locate ROCO v2 data root. Provide --data-root explicitly, e.g. "
            '"C:\\AIE Files\\Projects\\S5\\Dl\\DATA\\rocov2".'
        )

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = args.output_dir / "vocabulary.json"

    train_rows = read_captions(data_root / "train_captions.csv", args.train_limit)
    vocabulary = None
    if vocab_path.is_file():
        vocabulary = Vocabulary.load(vocab_path)
    if vocabulary is None:
        vocabulary = Vocabulary.build((caption for _, caption in train_rows), max_size=args.max_vocab)
        vocabulary.save(vocab_path)

    train_set = RoCoCaptionDataset(data_root, "train", vocabulary, args.image_size, args.max_length, args.train_limit)
    valid_set = RoCoCaptionDataset(data_root, "valid", vocabulary, args.image_size, args.max_length, args.valid_limit)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = CNNLSTMCaptioner(
        vocab_size=len(vocabulary.id_to_token),
        feature_dim=args.feature_dim,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        lstm_layers=args.lstm_layers,
        pad_id=vocabulary.pad_id,
        freeze_encoder=not args.unfreeze_encoder,
        decoder_dropout=args.decoder_dropout,
    ).to(device)

    criterion = CaptionCrossEntropyLoss(pad_id=vocabulary.pad_id).to(device)
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
    )

    model_config = {
        "vocab_size": len(vocabulary.id_to_token),
        "feature_dim": args.feature_dim,
        "embedding_dim": args.embedding_dim,
        "hidden_dim": args.hidden_dim,
        "lstm_layers": args.lstm_layers,
        "pad_id": vocabulary.pad_id,
        "freeze_encoder": not args.unfreeze_encoder,
        "decoder_dropout": args.decoder_dropout,
    }

    best_valid = float("inf")
    global_step = 0
    start_epoch = 1
    metrics_path = args.output_dir / "metrics.jsonl"

    if args.resume:
        checkpoint_path = Path(args.resume)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        if "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        global_step = int(checkpoint.get("global_step", 0))
        start_epoch = int(checkpoint.get("epoch", 0)) + 1

    print(f"device={device} train={len(train_set)} valid={len(valid_set)} vocab={len(vocabulary.id_to_token)}")

    def save_step_checkpoint(step: int, train_batch_loss: float) -> None:
        checkpoint = {
            "epoch": epoch,
            "global_step": step,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "train_batch_loss": train_batch_loss,
            "vocab_size": len(vocabulary.id_to_token),
            "model_config": model_config,
            "vocabulary": {
                "token_to_id": vocabulary.token_to_id,
                "id_to_token": vocabulary.id_to_token,
                "document_frequency": vocabulary.document_frequency,
            },
        }
        torch.save(checkpoint, args.output_dir / f"step_{step:07d}.pt")

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, global_step = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            True,
            start_step=global_step,
            checkpoint_every_steps=args.checkpoint_every_steps,
            on_step_checkpoint=save_step_checkpoint,
        )
        valid_loss, _ = run_epoch(model, valid_loader, criterion, optimizer, device, False, start_step=global_step)
        print(f"epoch={epoch:02d} train_loss={train_loss:.4f} valid_loss={valid_loss:.4f}")

        checkpoint = {
            "epoch": epoch,
            "global_step": global_step,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "valid_loss": valid_loss,
            "vocab_size": len(vocabulary.id_to_token),
            "model_config": model_config,
            "vocabulary": {
                "token_to_id": vocabulary.token_to_id,
                "id_to_token": vocabulary.id_to_token,
                "document_frequency": vocabulary.document_frequency,
            },
        }
        torch.save(checkpoint, args.output_dir / "last.pt")
        if valid_loss < best_valid:
            best_valid = valid_loss
            torch.save(checkpoint, args.output_dir / "best.pt")

        with metrics_path.open("a", encoding="utf-8") as metrics_file:
            metrics_file.write(
                json.dumps(
                    {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "valid_loss": valid_loss,
                        "best_valid": best_valid,
                        "global_step": global_step,
                        "train_size": len(train_set),
                        "valid_size": len(valid_set),
                        "vocab_size": len(vocabulary.id_to_token),
                    }
                )
                + "\n"
            )


if __name__ == "__main__":
    main()

