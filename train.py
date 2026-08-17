import os
import copy
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from dataset import ROCODataset, build_vocab_from_csv
from caption_model import SwinCaptioningModel

ROOT = "/Users/nagashiva/Downloads/rocov2"
TRAIN_IMAGES = os.path.join(ROOT, "train_images", "train")
VALID_IMAGES = os.path.join(ROOT, "valid_images", "valid")
TRAIN_CSV = os.path.join(ROOT, "train_captions.csv")
VALID_CSV = os.path.join(ROOT, "valid_captions.csv")

TRAIN_SAMPLES = 5000
VALID_SAMPLES = 2000
MAX_LEN = 40
BATCH_SIZE = 32
EPOCHS = 60
PEAK_LR = 2e-4
WARMUP_EPOCHS = 5
WEIGHT_DECAY = 0.05
LABEL_SMOOTHING = 0.1
EARLY_STOP_PATIENCE = 6
NUM_WORKERS = 4   # parallel image loading -- matters more now with 15k images
DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                       "mps" if torch.backends.mps.is_available() else "cpu")

train_transform = T.Compose([
    T.Resize((224, 224)),
    T.RandomRotation(degrees=5),
    T.ColorJitter(brightness=0.1, contrast=0.1),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def lr_lambda(epoch, warmup_epochs, total_epochs):
    import math
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / max(1, (total_epochs - warmup_epochs))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def build_param_groups(model, weight_decay):
    """Split params into decay / no-decay groups, matching the Swin paper's
    own training recipe: exclude all 1-D params (LayerNorm weight+bias,
    linear biases) and the relative-position-bias table (2-D, so the ndim
    check alone would miss it) from weight decay. Decaying these distorts
    normalization scale and the learned window position bias without
    helping generalization -- see FINDINGS.md."""
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or "relative_position_bias_table" in name:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def main():
    print("Device:", DEVICE)

    vocab = build_vocab_from_csv(TRAIN_CSV, min_freq=2, max_samples=TRAIN_SAMPLES)
    print("Vocab size:", len(vocab))

    train_ds = ROCODataset(TRAIN_CSV, TRAIN_IMAGES, vocab, MAX_LEN,
                            max_samples=TRAIN_SAMPLES, transform=train_transform)
    valid_ds = ROCODataset(VALID_CSV, VALID_IMAGES, vocab, MAX_LEN, max_samples=VALID_SAMPLES)

    print(f"Train samples: {len(train_ds)}  Valid samples: {len(valid_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = SwinCaptioningModel(vocab_size=len(vocab), max_len=MAX_LEN).to(DEVICE)

    optimizer = torch.optim.AdamW(build_param_groups(model, WEIGHT_DECAY), lr=PEAK_LR)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda ep: lr_lambda(ep, WARMUP_EPOCHS, EPOCHS)
    )
    criterion = torch.nn.CrossEntropyLoss(ignore_index=vocab.pad_id, label_smoothing=LABEL_SMOOTHING)

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        num_batches = 0
        for images, captions in train_loader:
            images, captions = images.to(DEVICE), captions.to(DEVICE)

            logits = model(images, captions)
            targets = captions[:, 1:]

            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            if num_batches % 100 == 0:
                print(f"  batch {num_batches}/{len(train_loader)}  running_loss={total_loss/num_batches:.4f}")

        avg_train_loss = total_loss / len(train_loader)

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

        current_lr = optimizer.param_groups[0]["lr"]
        improved = avg_val_loss < best_val_loss
        marker = "  <-- best so far" if improved else ""
        print(f"Epoch {epoch}/{EPOCHS}  train_loss={avg_train_loss:.4f}  "
              f"val_loss={avg_val_loss:.4f}  lr={current_lr:.2e}{marker}")
        scheduler.step()

        if improved:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            torch.save({
                "model_state": model.state_dict(),
                "vocab_itos": vocab.itos,
                "epoch": epoch,
                "val_loss": best_val_loss,
            }, os.path.join(ROOT, "swin_caption_best.pt"))
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= EARLY_STOP_PATIENCE:
            print(f"No val_loss improvement for {EARLY_STOP_PATIENCE} epochs. Stopping early.")
            break

    torch.save({
        "model_state": model.state_dict(),
        "vocab_itos": vocab.itos,
        "epoch": epoch,
    }, os.path.join(ROOT, "swin_caption_last.pt"))

    print(f"Training complete. Best val_loss={best_val_loss:.4f} -> saved as swin_caption_best.pt")


if __name__ == "__main__":
    main()
