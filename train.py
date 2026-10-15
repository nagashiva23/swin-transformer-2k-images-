import os
import math
import json
import argparse
import time
import pandas as pd
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from dataset import ROCODataset, build_vocab_from_csv
from caption_model import SwinCaptioningModel
from plot_training import plot_training_curves
from visualize_inferences import run_visualize_inferences

ROOT = os.environ.get("ROCO_ROOT", r"C:\AIE Files\Projects\S5\Dl\DATA\rocov2")
TRAIN_IMAGES = os.path.join(ROOT, "train_images", "train")
VALID_IMAGES = os.path.join(ROOT, "valid_images", "valid")
TRAIN_CSV = os.path.join(ROOT, "train_captions.csv")
VALID_CSV = os.path.join(ROOT, "valid_captions.csv")

TRAIN_SAMPLES = None   # None = train on full dataset (59,958 images)
VALID_SAMPLES = None   # None = validate on full dataset (9,904 images)
MAX_LEN = 40
BATCH_SIZE = 32
EPOCHS = 30
FREEZE_BACKBONE_EPOCHS = 2
LR_PHASE1 = 1.5e-4
LR_BACKBONE_PHASE2 = 1.5e-5
LR_DECODER_PHASE2 = 1.0e-4
WEIGHT_DECAY = 0.05
LABEL_SMOOTHING = 0.1
WORD_DROP_P = 0.10
DROPOUT = 0.20
EARLY_STOP_PATIENCE = 8
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                       "mps" if torch.backends.mps.is_available() else "cpu")
GPU_NAME = torch.cuda.get_device_name(DEVICE) if DEVICE.type == "cuda" else "N/A"


train_transform = T.Compose([
    T.Resize((224, 224)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomAffine(degrees=10, translate=(0.05, 0.05)),
    T.ColorJitter(brightness=0.1, contrast=0.1),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def set_backbone_trainable(model, trainable: bool):
    for param in model.encoder.parameters():
        param.requires_grad = trainable


def build_optimizer_and_scheduler(model, epoch, total_epochs, steps_per_epoch):
    if epoch <= FREEZE_BACKBONE_EPOCHS:
        # Phase 1: Train decoder only
        set_backbone_trainable(model, False)
        dec_decay, dec_no_decay = [], []
        for name, param in model.decoder.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim <= 1 or "embed" in name or "norm" in name:
                dec_no_decay.append(param)
            else:
                dec_decay.append(param)
        param_groups = [
            {"params": dec_decay, "lr": LR_PHASE1, "weight_decay": WEIGHT_DECAY},
            {"params": dec_no_decay, "lr": LR_PHASE1, "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(param_groups)
        total_phase1_steps = FREEZE_BACKBONE_EPOCHS * steps_per_epoch
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_phase1_steps), eta_min=1e-5)
    else:
        # Phase 2: Fine-tune backbone and decoder end-to-end
        set_backbone_trainable(model, True)
        enc_decay, enc_no_decay = [], []
        dec_decay, dec_no_decay = [], []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            is_enc = name.startswith("encoder")
            is_no_decay = (param.ndim <= 1 or "relative_position_bias_table" in name or "embed" in name or "norm" in name)
            if is_enc:
                if is_no_decay:
                    enc_no_decay.append(param)
                else:
                    enc_decay.append(param)
            else:
                if is_no_decay:
                    dec_no_decay.append(param)
                else:
                    dec_decay.append(param)

        param_groups = [
            {"params": enc_decay, "lr": LR_BACKBONE_PHASE2, "weight_decay": WEIGHT_DECAY},
            {"params": enc_no_decay, "lr": LR_BACKBONE_PHASE2, "weight_decay": 0.0},
            {"params": dec_decay, "lr": LR_DECODER_PHASE2, "weight_decay": WEIGHT_DECAY},
            {"params": dec_no_decay, "lr": LR_DECODER_PHASE2, "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(param_groups)
        remaining_steps = max(1, (total_epochs - FREEZE_BACKBONE_EPOCHS) * steps_per_epoch)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining_steps, eta_min=5e-6)

    return optimizer, scheduler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--min_freq", type=int, default=5)
    parser.add_argument("--train_samples", type=int, default=None)
    parser.add_argument("--valid_samples", type=int, default=None)
    args = parser.parse_args()

    print("Device:", DEVICE)
    if DEVICE.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    vocab = build_vocab_from_csv(TRAIN_CSV, min_freq=args.min_freq, max_samples=args.train_samples)
    print("Vocab size:", len(vocab))

    train_ds = ROCODataset(TRAIN_CSV, TRAIN_IMAGES, vocab, MAX_LEN,
                            max_samples=args.train_samples, transform=train_transform)
    valid_ds = ROCODataset(VALID_CSV, VALID_IMAGES, vocab, MAX_LEN, max_samples=args.valid_samples)

    print(f"Train samples: {len(train_ds)}  Valid samples: {len(valid_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=NUM_WORKERS, persistent_workers=(NUM_WORKERS > 0),
        pin_memory=(DEVICE.type == "cuda")
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=NUM_WORKERS, persistent_workers=(NUM_WORKERS > 0),
        pin_memory=(DEVICE.type == "cuda")
    )

    model = SwinCaptioningModel(
        vocab_size=len(vocab), max_len=MAX_LEN, dropout=DROPOUT, pretrained=True
    ).to(DEVICE)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=vocab.pad_id, label_smoothing=LABEL_SMOOTHING)

    use_amp = (DEVICE.type == "cuda")
    amp_dtype = torch.float16 if DEVICE.type == "cuda" else torch.float32

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    start_epoch = 1
    training_history = []

    last_ckpt_path = os.path.join(ROOT, "swin_caption_last.pt")
    best_ckpt_path = os.path.join(ROOT, "swin_caption_best.pt")

    if args.resume and os.path.exists(last_ckpt_path):
        print(f"Resuming from {last_ckpt_path}...")
        ckpt = torch.load(last_ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        if os.path.exists("training_history.json"):
            with open("training_history.json", "r") as f:
                training_history = json.load(f)
        print(f"Resumed at Epoch {start_epoch}, Best Val Loss: {best_val_loss:.4f}")

    steps_per_epoch = len(train_loader)
    optimizer, scheduler = None, None
    current_phase = None

    for epoch in range(start_epoch, args.epochs + 1):
        if DEVICE.type == "cuda":
            torch.cuda.reset_peak_memory_stats(DEVICE)
        epoch_started = time.perf_counter()
        phase = 1 if epoch <= FREEZE_BACKBONE_EPOCHS else 2
        if phase != current_phase:
            current_phase = phase
            print(f"\n{'='*60}\n--> Starting Phase {phase}: {'Encoder Frozen (Decoder Warmup)' if phase == 1 else 'End-to-End Fine-Tuning'}\n{'='*60}")
            optimizer, scheduler = build_optimizer_and_scheduler(model, epoch, args.epochs, steps_per_epoch)

        model.train()
        total_loss = 0.0
        num_batches = 0

        for images, captions in train_loader:
            images, captions = images.to(DEVICE), captions.to(DEVICE)

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    logits = model(images, captions, word_drop_p=WORD_DROP_P,
                                   unk_id=vocab.unk_id, pad_id=vocab.pad_id,
                                   sos_id=vocab.sos_id, eos_id=vocab.eos_id)
                    targets = captions[:, 1:]
                    loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            else:
                logits = model(images, captions, word_drop_p=WORD_DROP_P,
                               unk_id=vocab.unk_id, pad_id=vocab.pad_id,
                               sos_id=vocab.sos_id, eos_id=vocab.eos_id)
                targets = captions[:, 1:]
                loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            num_batches += 1
            if num_batches % 200 == 0:
                current_lr = optimizer.param_groups[-1]["lr"]
                print(f"  [Ep {epoch}] batch {num_batches}/{len(train_loader)}  running_loss={total_loss/num_batches:.4f}  lr={current_lr:.2e}")

        avg_train_loss = total_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, captions in valid_loader:
                images, captions = images.to(DEVICE), captions.to(DEVICE)
                logits = model(images, captions, word_drop_p=0.0)
                targets = captions[:, 1:]
                loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                val_loss += loss.item()
        avg_val_loss = val_loss / max(1, len(valid_loader))
        if DEVICE.type == "cuda":
            torch.cuda.synchronize(DEVICE)
        epoch_seconds = time.perf_counter() - epoch_started
        peak_gpu_memory_gib = (torch.cuda.max_memory_allocated(DEVICE) / (1024 ** 3)
                               if DEVICE.type == "cuda" else None)

        dec_lr = optimizer.param_groups[-1]["lr"]
        enc_lr = optimizer.param_groups[0]["lr"] if current_phase == 2 else 0.0

        improved = avg_val_loss < best_val_loss
        marker = "  <-- best so far" if improved else ""
        print(f"Epoch {epoch}/{args.epochs}  train_loss={avg_train_loss:.4f}  "
              f"val_loss={avg_val_loss:.4f}  lr_dec={dec_lr:.2e}  "
              f"time={epoch_seconds:.1f}s gpu={GPU_NAME}{marker}")

        training_history.append({
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(avg_val_loss, 4),
            "lr_backbone": enc_lr,
            "lr_decoder": dec_lr,
            "epoch_seconds": epoch_seconds,
            "gpu_name": GPU_NAME,
            "peak_gpu_memory_gib": peak_gpu_memory_gib,
        })

        with open("training_history.json", "w") as f:
            json.dump(training_history, f, indent=2)
        pd.DataFrame(training_history).to_csv("training_history.csv", index=False)

        plot_training_curves("training_history.json", "figures/swin_training_curves.png")

        ckpt_payload = {
            "model_state": model.state_dict(),
            "vocab_itos": vocab.itos,
            "epoch": epoch,
            "val_loss": avg_val_loss,
            "best_val_loss": min(best_val_loss, avg_val_loss),
        }

        if improved:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            torch.save(ckpt_payload, best_ckpt_path)
            torch.save(ckpt_payload, last_ckpt_path)
        else:
            epochs_without_improvement += 1
            torch.save(ckpt_payload, last_ckpt_path)

        if epochs_without_improvement >= EARLY_STOP_PATIENCE:
            print(f"No val_loss improvement for {EARLY_STOP_PATIENCE} epochs. Stopping early.")
            break

    print(f"Training complete. Best val_loss={best_val_loss:.4f} -> saved as swin_caption_best.pt")

    print("Generating qualitative inferences with Beam Search...")
    run_visualize_inferences(best_ckpt_path, VALID_CSV, VALID_IMAGES, num_samples=6, beam_width=3,
                            output_figure="figures/swin_qualitative_predictions.png",
                            output_csv="artifacts/swin_sample_predictions.csv")


if __name__ == "__main__":
    main()
