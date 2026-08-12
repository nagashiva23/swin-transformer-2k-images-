# Swin Transformer Image Captioning — ROCOv2 (manual implementation)


Dataset link: https://www.kaggle.com/datasets/claudiopisa9884/roco-v2




Every component is written from scratch with plain `torch.nn` — no `timm`,
no `transformers`, no pretrained weights, no third-party captioning repo.

## Files

| File | What it is |
|---|---|
| `vocab.py` | Manual regex tokenizer + vocab builder |
| `dataset.py` | ROCOv2 `Dataset` that loads images + captions |
| `swin_model.py` | The Swin encoder: PatchEmbed → WindowAttention → SwinBlock (W-MSA/SW-MSA) → PatchMerging → BasicLayer stages → SwinEncoder |
| `decoder_model.py` | The caption decoder: PositionalEncoding → manual MultiHeadAttention → DecoderLayer (masked self-attn + cross-attn + FFN) → CaptionDecoder |
| `caption_model.py` | Wires encoder + decoder into `SwinCaptioningModel`, plus greedy `.generate()` |
| `train.py` | Training loop — currently `TRAIN_SAMPLES = 15000`, `VALID_SAMPLES = 2000`, 60 epochs w/ early stopping |
| `generate.py` | Loads a checkpoint and captions a single image |
| `metrics.py` | **New.** BLEU-1..4 / ROUGE-L / CIDEr-D, implemented from scratch (no nltk/pycocoevalcap needed) |
| `eval.py` | **New.** Runs the model over a whole split (train/valid/test) and reports the metrics above, plus a caption-collapse check |

For the analysis behind the "known issues" below (checkpoint/vocab
mismatch, early loss plateau, dataset stats, benchmark comparison), see
**`FINDINGS.md`**.

## Dataset structure

The ROCOv2 export (`/Users/nagashiva/Downloads/rocov2`) is laid out as:

```
rocov2/
├── train_images/train/*.jpg          59,958 images  (ROCOv2_2023_train_XXXXXX.jpg)
├── valid_images/valid/*.jpg           9,904 images  (ROCOv2_2023_valid_XXXXXX.jpg)
├── test_images/test/*.jpg             9,927 images  (ROCOv2_2023_test_XXXXXX.jpg)
│
├── train_captions.csv                 59,958 rows — columns: ID, Caption
├── valid_captions.csv                  9,904 rows — columns: ID, Caption
├── test_captions.csv                   9,927 rows — columns: ID, Caption
│
├── train_concepts.csv / train_concepts_manual.csv    UMLS CUI concept labels per image (not used by any script here)
├── valid_concepts.csv / valid_concepts_manual.csv
├── test_concepts.csv  / test_concepts_manual.csv
├── cui_mapping.csv                    CUI -> human-readable concept name lookup (not used here)
├── license_information.csv            per-image source/license metadata (not used here)
│
└── swin_caption_*.pt                  checkpoints written here by train.py (swin_caption_best.pt, swin_caption_last.pt)
```

Image ID ↔ caption row count matches exactly 1:1 for all three splits
(verified: 59,958 / 9,904 / 9,927 jpg files line up with the CSV row
counts). `dataset.py`'s `IMAGE_COL="ID"` / `CAPTION_COL="Caption"` already
match this CSV schema — no column renaming needed.

**What the code actually trains/evaluates on today** — this is the gap
worth knowing about before your faculty meeting:

| Split | Available on disk | Actually used by `train.py` |
|---|---|---|
| train | 59,958 | first 15,000 (`TRAIN_SAMPLES`) |
| valid | 9,904 | first 2,000 (`VALID_SAMPLES`) |
| test | 9,927 | **0 — no script currently touches the test set** |

`eval.py` (added here) is what closes that last gap — it's the first
script in the project that actually scores against `test_captions.csv`.

The `*_concepts*.csv` / `cui_mapping.csv` files are part of the official
ROCOv2 release (multi-label concept tags per image, used for the
ImageCLEFmedical "concept detection" subtask) but nothing in this project
reads them — they're only relevant if you later want to add a concept-tag
auxiliary loss.

## Before running

Your CSV column names already match (`ID` / `Caption`) — no edit needed in
`dataset.py` unless you're pointed at a different export.

## Run

```bash
cd "swin 2k"
pip install torch torchvision pandas pillow
python train.py
```

Trains on the first 15,000 rows of `train_captions.csv` / `train_images`,
validates on 2,000 rows of `valid_captions.csv` / `valid_images`, saves
`swin_caption_best.pt` (best val loss) and `swin_caption_last.pt` into
`/Users/nagashiva/Downloads/rocov2`.

To caption a single image:

```bash
python generate.py /path/to/image.jpg /Users/nagashiva/Downloads/rocov2/swin_caption_best.pt
```

To score a checkpoint against a whole split with BLEU/ROUGE/CIDEr:

```bash
python eval.py /Users/nagashiva/Downloads/rocov2/swin_caption_best.pt --split test --n 200
```

## Architecture summary

**Encoder (Swin-T sized):** 224×224×3 image → patch_size 4 → 56×56 tokens
(embed_dim 96) → 4 stages with depths `(2,2,6,2)` and heads `(3,6,12,24)`,
window_size 7, each stage's odd blocks shift windows by `window_size // 2`.
Patch merging halves resolution / doubles channels between stages, ending
at 7×7×768 = 49 tokens.

**Decoder:** standard 6-layer Transformer decoder — masked self-attention
over the caption so far, cross-attention over the 49 Swin tokens, then a
GELU feed-forward block — trained with teacher forcing and cross-entropy
(padding ignored, label smoothing 0.1).

## Known issues (short version — see FINDINGS.md for the full writeup)

- `swin_caption_best.pt`'s embedded vocab (2,682 words) matches training on
  only **2,000** samples, not the 15,000 the current `train.py` targets —
  the checkpoint you're testing with is from an older run.
- That checkpoint's best validation loss was logged at **epoch 3** and
  never improved again — the model converged to a generic, image-agnostic
  caption almost immediately (this is the "encoder representation
  collapse" `train.py`'s own docstring warns about).
- `AdamW(model.parameters(), weight_decay=0.05)` applies weight decay to
  every parameter, including LayerNorm/bias/relative-position-bias —
  standard Swin/Transformer recipes exclude 1-D params from decay.
- `decoder_model.py` masks with literal `-inf` while `swin_model.py`
  deliberately uses `-100.0` for the same purpose (likely to avoid a known
  MPS softmax NaN issue) — worth making consistent given you're training
  on Apple Silicon (M-series) MPS.
- `irdid.ipynb.py` is a dead, unimported duplicate of `vocab.py` — safe to
  delete.
