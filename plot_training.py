import os
import json
import matplotlib.pyplot as plt
import numpy as np

def plot_training_curves(history_path="training_history.json", output_path="figures/swin_training_curves.png"):
    if not os.path.exists(history_path):
        return

    try:
        with open(history_path, "r") as f:
            history = json.load(f)
    except Exception:
        return

    if not history:
        return

    epochs = [entry["epoch"] for entry in history]
    train_losses = [entry["train_loss"] for entry in history]
    val_losses = [entry["val_loss"] for entry in history]
    lr_backbones = [entry.get("lr_backbone", entry.get("lr", 0.0)) for entry in history]
    lr_decoders = [entry.get("lr_decoder", entry.get("lr", 0.0)) for entry in history]

    best_val_idx = int(np.argmin(val_losses))
    best_epoch = epochs[best_val_idx]
    best_val_loss = val_losses[best_val_idx]

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5), dpi=300)

    # 1. Training & Validation Loss
    ax1.plot(epochs, train_losses, 'o-', color='#1f77b4', linewidth=2.2, label='Train Loss', markersize=5)
    ax1.plot(epochs, val_losses, 's-', color='#d62728', linewidth=2.2, label='Val Loss', markersize=5)
    ax1.scatter([best_epoch], [best_val_loss], color='#2ca02c', s=140, zorder=5, edgecolors='black',
                label=f'Best Val (Ep {best_epoch}: {best_val_loss:.4f})')
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Cross Entropy Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Training & Validation Loss', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', frameon=True)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # 2. Validation Perplexity
    val_ppl = [np.exp(min(vl, 20)) for vl in val_losses]
    ax2.plot(epochs, val_ppl, 'd-', color='#9467bd', linewidth=2.2, label='Val Perplexity', markersize=5)
    ax2.scatter([best_epoch], [val_ppl[best_val_idx]], color='#2ca02c', s=140, zorder=5, edgecolors='black',
                label=f'Min PPL: {val_ppl[best_val_idx]:.2f}')
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Perplexity (exp(Loss))', fontsize=12, fontweight='bold')
    ax2.set_title('Validation Perplexity', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right', frameon=True)
    ax2.grid(True, linestyle='--', alpha=0.6)

    # 3. Learning Rate Schedule
    ax3.plot(epochs, lr_backbones, '^-', color='#ff7f0e', linewidth=2, label='Backbone LR', markersize=5)
    ax3.plot(epochs, lr_decoders, 'v-', color='#2ca02c', linewidth=2, label='Decoder LR', markersize=5)
    ax3.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
    ax3.set_title('Learning Rate Schedule', fontsize=13, fontweight='bold')
    ax3.set_yscale('log')
    ax3.legend(loc='lower left', frameon=True)
    ax3.grid(True, linestyle='--', alpha=0.6)

    plt.suptitle('Swin Transformer Medical Image Captioning (ROCOv2)', fontsize=15, fontweight='bold', y=1.03)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f'[Plotting] Saved training curve figure to {output_path}')

if __name__ == '__main__':
    plot_training_curves()
