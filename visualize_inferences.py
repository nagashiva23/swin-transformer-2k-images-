import os
import argparse
import textwrap
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T

from caption_model import SwinCaptioningModel
from vocab import Vocab

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
MAX_LEN = 40

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_vocab_from_checkpoint(ckpt):
    v = Vocab.__new__(Vocab)
    v.itos = ckpt["vocab_itos"]
    v.stoi = {w: i for i, w in enumerate(v.itos)}
    v.pad_token, v.sos_token, v.eos_token, v.unk_token = "<pad>", "<sos>", "<eos>", "<unk>"
    v.pad_id = v.stoi[v.pad_token]
    v.sos_id = v.stoi[v.sos_token]
    v.eos_id = v.stoi[v.eos_token]
    v.unk_id = v.stoi[v.unk_token]
    return v

def resolve_image_path(images_dir, image_id):
    image_id = str(image_id)
    if os.path.splitext(image_id)[1]:
        return os.path.join(images_dir, image_id)
    for ext in (".jpg", ".jpeg", ".png"):
        cand = os.path.join(images_dir, image_id + ext)
        if os.path.exists(cand):
            return cand
    return os.path.join(images_dir, image_id + ".jpg")

def run_visualize_inferences(checkpoint_path, csv_path, images_dir, num_samples=6, beam_width=3, output_figure="figures/swin_qualitative_predictions.png", output_csv="artifacts/swin_sample_predictions.csv"):
    if not os.path.exists(checkpoint_path):
        print(f"[Inference] Checkpoint not found: {checkpoint_path}")
        return

    print(f"[Inference] Loading checkpoint from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    vocab = load_vocab_from_checkpoint(ckpt)

    model = SwinCaptioningModel(vocab_size=len(vocab), max_len=MAX_LEN, pretrained=True).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    df = pd.read_csv(csv_path)
    sample_df = df.iloc[:num_samples]

    rows = []
    images_pil = []
    image_tensors = []

    for _, row in sample_df.iterrows():
        img_p = resolve_image_path(images_dir, row["ID"])
        if os.path.exists(img_p):
            pil_img = Image.open(img_p).convert("RGB")
        else:
            pil_img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        images_pil.append(pil_img)
        image_tensors.append(transform(pil_img))

    batch = torch.stack(image_tensors).to(DEVICE)
    with torch.no_grad():
        if beam_width > 1:
            ids_batch = model.generate_beam(batch, vocab, beam_width=beam_width, max_len=MAX_LEN, device=DEVICE)
        else:
            ids_batch = model.generate(batch, vocab, max_len=MAX_LEN, device=DEVICE)

    for row, ids in zip(sample_df.to_dict(orient="records"), ids_batch):
        gen_caption = vocab.decode(ids.tolist())
        gt_caption = str(row["Caption"])
        rows.append({
            "ID": row["ID"],
            "Ground_Truth": gt_caption,
            "Generated_Caption": gen_caption
        })

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    res_df = pd.DataFrame(rows)
    res_df.to_csv(output_csv, index=False)
    print(f"[Inference] Saved prediction table to {output_csv}")

    cols = 3
    num_items = len(rows)
    grid_rows = (num_items + cols - 1) // cols

    fig, axes = plt.subplots(grid_rows, cols, figsize=(cols * 5.5, grid_rows * 6.5), dpi=300)
    if hasattr(axes, "flatten"):
        axes = axes.flatten()
    elif not isinstance(axes, (list, tuple)):
        axes = [axes]

    for idx in range(len(axes)):
        ax = axes[idx]
        if idx < num_items:
            img = images_pil[idx]
            ax.imshow(img)
            ax.axis("off")
            item = rows[idx]
            gt_wrapped = "\n".join(textwrap.wrap("GT: " + item["Ground_Truth"], width=42))
            gen_wrapped = "\n".join(textwrap.wrap("Pred: " + item["Generated_Caption"], width=42))
            caption_text = f"{item['ID']}\n\n{gt_wrapped}\n\n{gen_wrapped}"
            ax.set_title(caption_text, fontsize=9.5, ha="center", va="top", y=-0.08,
                         bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#cccccc", alpha=0.9))
        else:
            ax.axis("off")

    plt.suptitle("Swin Transformer Captioning: Qualitative Inference Samples (ROCOv2)", fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])

    os.makedirs(os.path.dirname(output_figure), exist_ok=True)
    plt.savefig(output_figure, bbox_inches="tight")
    plt.close()
    print(f"[Inference] Saved qualitative inference figure to {output_figure}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=r"C:\AIE Files\Projects\S5\Dl\DATA\rocov2\swin_caption_best.pt")
    parser.add_argument("--csvpath", default=r"C:\AIE Files\Projects\S5\Dl\DATA\rocov2\valid_captions.csv")
    parser.add_argument("--images", default=r"C:\AIE Files\Projects\S5\Dl\DATA\rocov2\valid_images\valid")
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--beam", type=int, default=3)
    parser.add_argument("--output_fig", default="figures/swin_qualitative_predictions.png")
    parser.add_argument("--output_csv", default="artifacts/swin_sample_predictions.csv")
    args = parser.parse_args()
    run_visualize_inferences(args.checkpoint, args.csvpath, args.images, args.n, args.beam, args.output_fig, args.output_csv)
