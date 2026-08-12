"""
Evaluation script -- BLEU-1..4, ROUGE-L, CIDEr-D of the trained captioning
model against ROCOv2 ground-truth captions, plus a caption-collapse check
(% of generated captions that are actually distinct from each other).

Computes metrics from scratch (see metrics.py) so no extra installs
(nltk / pycocoevalcap / java) are needed beyond what train.py already uses.

Usage:
    python eval.py /Users/nagashiva/Downloads/rocov2/swin_caption_best.pt --split test --n 200

    --split   which split to evaluate on: train | valid | test   (default: test)
    --n       how many samples to evaluate, 0 = all               (default: 200; full test set is slow on CPU/MPS)
    --root    dataset root                                        (default: /Users/nagashiva/Downloads/rocov2)
    --out     optional CSV path to dump (ID, reference, generated) for manual inspection

ROCOv2 benchmark reference points (for comparison, see FINDINGS.md for sources):
    ImageCLEFmedical Caption 2023 CNN-LSTM (DenseNet169) baseline on ROCOv2:
        BLEU-1 0.586  BLEU-2 0.498  BLEU-3 0.347  BLEU-4 0.127  ROUGE 0.453  CIDEr 0.463
"""

import argparse
import os

import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image

from caption_model import SwinCaptioningModel
from metrics import corpus_bleu, corpus_cider_d, corpus_rouge_l
from vocab import Vocab

MAX_LEN = 40
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)

transform = T.Compose(
    [
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

SPLIT_PATHS = {
    "train": ("train_captions.csv", os.path.join("train_images", "train")),
    "valid": ("valid_captions.csv", os.path.join("valid_images", "valid")),
    "test": ("test_captions.csv", os.path.join("test_images", "test")),
}


def load_vocab_from_checkpoint(ckpt):
    v = Vocab.__new__(Vocab)  # bypass __init__, restore state directly
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--split", choices=["train", "valid", "test"], default="test")
    ap.add_argument("--n", type=int, default=200, help="number of samples to evaluate, 0 = all")
    ap.add_argument("--root", default="/Users/nagashiva/Downloads/rocov2")
    ap.add_argument("--out", default=None, help="optional CSV path to dump generated vs reference captions")
    args = ap.parse_args()

    csv_name, img_subdir = SPLIT_PATHS[args.split]
    csv_path = os.path.join(args.root, csv_name)
    images_dir = os.path.join(args.root, img_subdir)

    print("Device:", DEVICE)
    ckpt = torch.load(args.checkpoint, map_location=DEVICE)
    vocab = load_vocab_from_checkpoint(ckpt)
    print("Checkpoint epoch:", ckpt.get("epoch"), " | vocab size in checkpoint:", len(vocab))
    if len(vocab) < 5000:
        print(
            "  NOTE: a vocab this small strongly suggests this checkpoint was trained on the "
            "2,000-sample config, not the current 15,000-sample train.py. See FINDINGS.md."
        )

    model = SwinCaptioningModel(vocab_size=len(vocab), max_len=MAX_LEN).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    df = pd.read_csv(csv_path)
    if args.n and args.n > 0:
        df = df.iloc[: args.n]
    print(f"Evaluating on {len(df)} samples from {csv_name}")

    refs, hyps, rows = [], [], []
    with torch.no_grad():
        for i, row in df.iterrows():
            img_path = resolve_image_path(images_dir, row["ID"])
            if not os.path.exists(img_path):
                continue
            image = Image.open(img_path).convert("RGB")
            image_t = transform(image).unsqueeze(0).to(DEVICE)
            ids = model.generate(image_t, vocab, max_len=MAX_LEN, device=DEVICE)
            gen_caption = vocab.decode(ids[0].tolist())

            ref_tokens = Vocab.tokenize(str(row["Caption"]))
            hyp_tokens = Vocab.tokenize(gen_caption)
            refs.append(ref_tokens)
            hyps.append(hyp_tokens)
            rows.append((row["ID"], row["Caption"], gen_caption))

            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(df)} captioned")

    if not hyps:
        print("No images were found/captioned -- check --root and the split's image folder.")
        return

    bleu = corpus_bleu(refs, hyps)
    rouge = corpus_rouge_l(refs, hyps)
    cider = corpus_cider_d(refs, hyps)
    unique_hyps = len(set(tuple(h) for h in hyps))

    print()
    print("=" * 56)
    print(f"Split: {args.split}  |  N: {len(hyps)}")
    print(f"BLEU-1: {bleu['bleu1']:.4f}   BLEU-2: {bleu['bleu2']:.4f}")
    print(f"BLEU-3: {bleu['bleu3']:.4f}   BLEU-4: {bleu['bleu4']:.4f}")
    print(f"ROUGE-L: {rouge:.4f}")
    print(f"CIDEr-D: {cider:.4f}")
    print(
        f"Distinct generated captions: {unique_hyps}/{len(hyps)} "
        f"({100 * unique_hyps / len(hyps):.1f}% unique) "
        "-- a low % here means the model is collapsing to generic captions"
    )
    print("=" * 56)
    print("Reference -- ROCOv2 CNN-LSTM baseline: BLEU-1 0.586 BLEU-4 0.127 ROUGE 0.453 CIDEr 0.463")

    if args.out:
        pd.DataFrame(rows, columns=["ID", "reference", "generated"]).to_csv(args.out, index=False)
        print("Saved per-image captions to", args.out)


if __name__ == "__main__":
    main()
