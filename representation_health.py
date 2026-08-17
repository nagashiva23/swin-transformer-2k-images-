"""
Encoder representation-health monitor.

This is the collapse detector from MODEL_VALIDATION_REPORT.md section 10,
packaged so it can run *during* training instead of being discovered afterwards.

The key idea: repetitive captions alone do NOT prove the encoder collapsed --
a healthy encoder feeding a lazy decoder produces the same symptom. To separate
those two hypotheses you must look at `memory = encoder(image)` directly, and
you must include synthetic controls (black / noise / grey). If a real medical
image's representation is no more distinct from RANDOM NOISE than it is from
another real image, the encoder is not encoding image content.

Usage as a library (recommended -- call from train.py every few epochs):
    from representation_health import check
    check(model.encoder, sample_images, device, epoch=epoch)

Usage as a script (audit an existing checkpoint):
    python representation_health.py /path/to/swin_caption_best.pt
"""

import torch
import torch.nn.functional as F


@torch.no_grad()
def token_stats(memory):
    """memory: (1, N, D) -> dict of within-image diversity statistics."""
    M = memory[0]
    N = M.size(0)
    norms = M.norm(dim=-1)
    Mn = F.normalize(M, dim=-1)
    cos = Mn @ Mn.T
    off = cos[~torch.eye(N, dtype=torch.bool, device=M.device)]
    total_var = M.var(unbiased=False)
    across_tokens = M.var(dim=0, unbiased=False).mean()
    return {
        "mean": M.mean().item(),
        "std": M.std().item(),
        "norm_mean": norms.mean().item(),
        "norm_std": norms.std().item(),
        "cos_mean": off.mean().item(),
        "cos_std": off.std().item(),
        "cos_min": off.min().item(),
        "frac_cos_gt_099": (off > 0.99).float().mean().item(),
        # THE headline number: what % of the signal distinguishes one spatial
        # position from another. Healthy encoders are well above 1%.
        "pct_spatial_variance": (100.0 * across_tokens / total_var).item(),
    }


@torch.no_grad()
def check(encoder, real_images, device, epoch=None, verbose=True):
    """
    encoder     : the SwinEncoder module (model.encoder)
    real_images : (K, 3, 224, 224) tensor of a few REAL, already-normalized images.
                  Two or three is plenty; use the same ones every epoch so the
                  numbers are comparable across time.

    Returns a dict; also prints a compact one-line-per-input report.
    """
    was_training = encoder.training
    encoder.eval()

    real_images = real_images.to(device)
    K = real_images.size(0)

    # synthetic controls, normalized identically to the real inputs
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    g = torch.Generator(device="cpu").manual_seed(0)
    controls = {
        "black": torch.zeros(1, 3, 224, 224),
        "noise": torch.rand(1, 3, 224, 224, generator=g),
        "grey": torch.full((1, 3, 224, 224), 0.5),
    }
    controls = {k: ((v.to(device) - mean) / std) for k, v in controls.items()}

    names, mems = [], []
    for i in range(K):
        names.append(f"real[{i}]")
        mems.append(encoder(real_images[i : i + 1]))
    for k, v in controls.items():
        names.append(f"CONTROL:{k}")
        mems.append(encoder(v))

    stats = {n: token_stats(m) for n, m in zip(names, mems)}

    pooled = F.normalize(torch.cat([m[0].mean(0, keepdim=True) for m in mems], 0), dim=-1)
    sim = pooled @ pooled.T

    # the decisive comparison
    real_vs_real = sim[0, 1].item() if K >= 2 else float("nan")
    real_vs_noise = sim[0, names.index("CONTROL:noise")].item()

    if verbose:
        tag = f" (epoch {epoch})" if epoch is not None else ""
        print(f"\n--- representation health{tag} ---")
        print(f"  {'input':<16} {'tok-cos':>8} {'%spatial-var':>13} {'norm std':>9}")
        for n in names:
            s = stats[n]
            print(f"  {n:<16} {s['cos_mean']:>8.4f} {s['pct_spatial_variance']:>13.3f} {s['norm_std']:>9.4f}")
        if K >= 2:
            print(f"  cos(real0, real1) = {real_vs_real:+.4f}")
        print(f"  cos(real0, NOISE) = {real_vs_noise:+.4f}")

        collapsed_tokens = stats["real[0]"]["cos_mean"] > 0.99
        blind_to_content = (K >= 2) and (real_vs_noise >= real_vs_real)
        if collapsed_tokens or blind_to_content:
            print("  >>> WARNING:")
            if collapsed_tokens:
                print("      the N tokens are near-identical (cos > 0.99) -> spatial collapse")
            if blind_to_content:
                print("      a real image is no more distinct from NOISE than from another real")
                print("      image -> the encoder is not encoding image content")
        else:
            print("  >>> healthy: tokens are differentiated and real images separate from noise")

    if was_training:
        encoder.train()

    return {"stats": stats, "sim": sim.cpu(),
            "real_vs_real": real_vs_real, "real_vs_noise": real_vs_noise}


if __name__ == "__main__":
    import os
    import sys

    import pandas as pd
    import torchvision.transforms as T
    from PIL import Image

    from caption_model import SwinCaptioningModel   # audits the ORIGINAL model
    from vocab import Vocab

    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else \
        "/Users/nagashiva/Downloads/rocov2/swin_caption_best.pt"
    root = "/Users/nagashiva/Downloads/rocov2"
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")

    ckpt = torch.load(ckpt_path, map_location=device)
    v = Vocab.__new__(Vocab)
    v.itos = ckpt["vocab_itos"]
    v.stoi = {w: i for i, w in enumerate(v.itos)}
    model = SwinCaptioningModel(vocab_size=len(v.itos), max_len=40).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"loaded {os.path.basename(ckpt_path)}  epoch={ckpt.get('epoch')} "
          f"val_loss={ckpt.get('val_loss')} vocab={len(v.itos)}")

    tf = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    df = pd.read_csv(os.path.join(root, "valid_captions.csv")).iloc[:3]
    imgs = torch.stack([
        tf(Image.open(os.path.join(root, "valid_images", "valid", f"{i}.jpg")).convert("RGB"))
        for i in df["ID"].astype(str)
    ])
    check(model.encoder, imgs, device)
