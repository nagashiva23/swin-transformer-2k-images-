"""
Training script for Swin + Transformer-decoder captioning on ROCOv2,
using a 2000-image subset (train) as requested.

Before running:
  1. pip install torch torchvision pandas pillow
  2. Check your CSV column names (see dataset.py IMAGE_COL / CAPTION_COL)
     and fix them if they don't match.
"""

import os
import torch
from torch.utils.data import DataLoader

from dataset import ROCODataset, build_vocab_from_csv
from caption_model import SwinCaptioningModel

# ---------------- paths (yours) ----------------
ROOT = "/Users/nagashiva/Downloads/rocov2"
TRAIN_IMAGES = os.path.join(ROOT, "train_images", "train")
VALID_IMAGES = os.path.join(ROOT, "valid_images", "valid")
TRAIN_CSV = os.path.join(ROOT, "train_captions.csv")
VALID_CSV = os.path.join(ROOT, "valid_captions.csv")

# ---------------- hyperparams ----------------
MAX_SAMPLES = 2000     # only 2k images as requested
MAX_LEN = 40
BATCH_SIZE = 16
EPOCHS = 10
LR = 3e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                       "mps" if torch.backends.mps.is_available() else "cpu")


def main():
    print("Device:", DEVICE)

    # BLOCK: build vocab from (a subset of) the training captions
    vocab = build_vocab_from_csv(TRAIN_CSV, min_freq=2, max_samples=MAX_SAMPLES)
    print("Vocab size:", len(vocab))

    # BLOCK: datasets / dataloaders (2000 train images, small valid slice)
    train_ds = ROCODataset(TRAIN_CSV, TRAIN_IMAGES, vocab, MAX_LEN, max_samples=MAX_SAMPLES)
    valid_ds = ROCODataset(VALID_CSV, VALID_IMAGES, vocab, MAX_LEN, max_samples=200)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # BLOCK: model
    model = SwinCaptioningModel(vocab_size=len(vocab), max_len=MAX_LEN).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=vocab.pad_id)

    # BLOCK: training loop
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for images, captions in train_loader:
            images, captions = images.to(DEVICE), captions.to(DEVICE)

            logits = model(images, captions)              # (B, T-1, V)
            targets = captions[:, 1:]                      # shifted target

            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # ---- quick validation ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, captions in valid_loader:
                images, captions = images.to(DEVICE), captions.to(DEVICE)
                logits = model(images, captions)
                targets = captions[:, 1:]
                loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                val_loss += loss.item()
        avg_val_loss = val_loss / max(1, len(valid_loader))

        print(f"Epoch {epoch}/{EPOCHS}  train_loss={avg_train_loss:.4f}  val_loss={avg_val_loss:.4f}")

        torch.save({
            "model_state": model.state_dict(),
            "vocab_itos": vocab.itos,
            "epoch": epoch,
        }, os.path.join(ROOT, f"swin_caption_epoch{epoch}.pt"))

    print("Training complete.")


if __name__ == "__main__":
    main()
