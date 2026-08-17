"""collapse_check.py -- representation-health audit (§10 of MODEL_VALIDATION_REPORT.md),
rerun against a fresh checkpoint. Measures:
  1. within-image token cosine similarity + variance decomposition (token vs channel variance)
  2. a noise-control experiment: does the encoder output for a real image look more like another
     real image, or more like black/white/grey/noise synthetic input?
Usage: python collapse_check.py <checkpoint.pt> [image_id ...]
"""
import sys
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

from caption_model import SwinCaptioningModel
from vocab import Vocab

ROOT = "/Users/nagashiva/Downloads/rocov2"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    vocab = Vocab.__new__(Vocab)
    vocab.itos = ckpt["vocab_itos"]
    vocab.stoi = {w: i for i, w in enumerate(vocab.itos)}
    vocab.pad_id, vocab.sos_id, vocab.eos_id, vocab.unk_id = 0, 1, 2, 3
    model = SwinCaptioningModel(vocab_size=len(vocab.itos), max_len=40).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, vocab, ckpt.get("epoch"), ckpt.get("val_loss")


def token_stats(memory):
    # memory: (1, 49, 768)
    x = memory[0]
    x_norm = F.normalize(x, dim=-1)
    cos = x_norm @ x_norm.T
    off_diag = cos[~torch.eye(cos.size(0), dtype=torch.bool, device=cos.device)]
    total_var = x.var(unbiased=False).item() * x.numel() / x.size(0)  # match report's "total variance" scale
    token_var = x.mean(dim=1).var(unbiased=False).item()  # variance across tokens of per-token mean -- proxy
    # more faithful: variance across tokens (avg over dims) vs variance across dims (avg over tokens)
    var_across_tokens = x.var(dim=0, unbiased=False).mean().item()
    var_across_dims = x.var(dim=1, unbiased=False).mean().item()
    return {
        "mean_cos": off_diag.mean().item(),
        "std_cos": off_diag.std().item(),
        "min_cos": off_diag.min().item(),
        "frac_cos_gt_099": (off_diag > 0.99).float().mean().item(),
        "norm_mean": x.norm(dim=-1).mean().item(),
        "norm_std": x.norm(dim=-1).std().item(),
        "var_across_tokens": var_across_tokens,
        "var_across_dims": var_across_dims,
    }


def load_real_image(image_id):
    if not image_id.startswith("ROCOv2_2023_"):
        image_id = f"ROCOv2_2023_{image_id}"
    path = f"{ROOT}/test_images/test/{image_id}.jpg"
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0)


def synthetic(kind):
    if kind == "black":
        img = torch.zeros(3, 224, 224)
    elif kind == "white":
        img = torch.ones(3, 224, 224)
    elif kind == "grey":
        img = torch.full((3, 224, 224), 0.5)
    elif kind == "noise":
        img = torch.rand(3, 224, 224)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return ((img - mean) / std).unsqueeze(0)


def main():
    ckpt_path = sys.argv[1]
    image_ids = sys.argv[2:] or ["test_007757", "test_000004"]

    model, vocab, epoch, val_loss = load_model(ckpt_path)
    print(f"Checkpoint: {ckpt_path}  epoch={epoch}  val_loss={val_loss}\n")

    pooled = {}
    with torch.no_grad():
        print("=== Within-image token statistics ===")
        for iid in image_ids:
            imgs = load_real_image(iid).to(DEVICE)
            mem = model.encoder(imgs)
            stats = token_stats(mem)
            pooled[iid] = F.normalize(mem[0].mean(dim=0), dim=0)
            print(f"{iid}: mean_cos={stats['mean_cos']:.4f} std={stats['std_cos']:.4f} "
                  f"min={stats['min_cos']:.4f} frac>0.99={stats['frac_cos_gt_099']*100:.1f}%  "
                  f"norm_mean={stats['norm_mean']:.3f} norm_std={stats['norm_std']:.4f}  "
                  f"var_across_tokens={stats['var_across_tokens']:.6f}  var_across_dims={stats['var_across_dims']:.6f}")

        print("\n=== Control experiment: real vs synthetic inputs ===")
        for kind in ["black", "white", "noise", "grey"]:
            imgs = synthetic(kind).to(DEVICE)
            mem = model.encoder(imgs)
            stats = token_stats(mem)
            pooled[kind] = F.normalize(mem[0].mean(dim=0), dim=0)
            ids = model.generate(imgs, vocab, max_len=40, device=DEVICE)
            print(f"{kind:6s}: within-image mean_cos={stats['mean_cos']:.4f}  caption='{vocab.decode(ids[0].tolist())}'")

        print("\n=== Pairwise cosine of mean-pooled representations ===")
        keys = list(pooled.keys())
        header = "        " + "  ".join(f"{k:>10s}" for k in keys)
        print(header)
        for k1 in keys:
            row = f"{k1:8s}" + "  ".join(f"{(pooled[k1] @ pooled[k2]).item():10.4f}" for k2 in keys)
            print(row)


if __name__ == "__main__":
    main()
