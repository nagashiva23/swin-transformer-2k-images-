# Swin Transformer Image Captioning — ROCOv2 (manual implementation)


Dataset link: https://www.kaggle.com/datasets/claudiopisa9884/roco-v2




Every component is written from scratch with plain `torch.nn` — no `timm`,
no `transformers`, no pretrained weights, no third-party captioning repo.

## Files

| File | What it is |
|---|---|
| `vocab.py` | Manual regex tokenizer + vocab builder |
| `dataset.py` | ROCOv2 `Dataset` that loads images + captions |
| `swin_model.py` | The Swin encoder: PatchEmbed → WindowAttention → SwinBlock (W-MSA/SW-MSA) → PatchMerging → BasicLayer stages → SwinEncoder. Includes weight init (`trunc_normal_`) and `DropPath` (stochastic depth, linearly increasing 0→0.1 across the 12 encoder blocks). |
| `decoder_model.py` | The caption decoder: PositionalEncoding → manual MultiHeadAttention → DecoderLayer (masked self-attn + cross-attn + FFN) → CaptionDecoder |
| `caption_model.py` | Wires encoder + decoder into `SwinCaptioningModel` (applies the model-wide weight init), plus greedy `.generate()` with trigram blocking (masking constant fixed to `-100.0`, matching the rest of the model) |
| `train.py` | Training loop — `TRAIN_SAMPLES = 5000`, `VALID_SAMPLES = 2000`, `PEAK_LR = 2e-4`, 60 epochs w/ early stopping (patience 6) |
| `generate.py` | Loads a checkpoint and captions a single image |
| `metrics.py` | BLEU-1..4 / ROUGE-L / CIDEr-D, implemented from scratch (no nltk/pycocoevalcap needed) |
| `eval.py` | Runs the model over a whole split (train/valid/test) and reports the metrics above, plus a caption-collapse check |
| `tiny_overfit.py` | **New.** Mandatory architecture-verification test — trains on 10 fixed image/caption pairs for 400 steps; if loss doesn't collapse to ~0 and reproduce the captions, the defect is architectural, not a training-recipe issue |
| `collapse_check.py` | **New.** Representation-health audit for a checkpoint: within-image token cosine similarity, variance decomposition, and a synthetic black/white/grey/noise control experiment |

For the full audit trail (bug list, gradient-flow analysis, representation-collapse
evidence, before/after metrics), see **`MODEL_VALIDATION_REPORT.md`** and
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
| train | 59,958 | first 5,000 (`TRAIN_SAMPLES`) |
| valid | 9,904 | first 2,000 (`VALID_SAMPLES`) |
| test | 9,927 | used only for qualitative diagnostics (`collapse_check.py`), never for training/checkpoint selection |

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

Trains on the first 5,000 rows of `train_captions.csv` / `train_images`,
validates on 2,000 rows of `valid_captions.csv` / `valid_images`, saves
`swin_caption_best.pt` (best val loss) and `swin_caption_last.pt` into
`/Users/nagashiva/Downloads/rocov2`.

To run the mandatory architecture-verification test before any full training run:

```bash
python tiny_overfit.py
```

To audit a checkpoint's encoder for representation collapse:

```bash
python collapse_check.py /Users/nagashiva/Downloads/rocov2/swin_caption_best.pt test_007757 test_000004
```

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

## Status (see `MODEL_VALIDATION_REPORT.md` for the full audit)

An early checkpoint showed severe **encoder representation collapse**: the
49 output tokens were nearly identical regardless of image content, and the
decoder converged to a handful of generic captions. A diagnostic audit
separated this into two questions:

1. **Is the architecture broken?** `tiny_overfit.py` (10 fixed image/caption
   pairs, 400 steps) passes cleanly both before and after the fixes below —
   loss falls to ~0.001 and all 10 captions are reproduced correctly. This
   rules out a gradient-flow/masking/wiring defect.
2. **Is the training recipe broken?** Yes — three concrete bugs, now fixed:
   - No weight initialization beyond the relative-position-bias table —
     fixed with `trunc_normal_(std=0.02)` on Linear weights and standard
     LayerNorm init, applied via `self.apply(_init_weights)`.
   - No stochastic depth (DropPath) in the encoder — added, linearly
     increasing 0→0.1 across the 12 encoder blocks.
   - `PEAK_LR = 5e-4` combined with `EARLY_STOP_PATIENCE = 6` meant training
     always stopped right as the LR schedule peaked (best epoch = 3 in every
     run). Lowered to `2e-4`.
   - (Lower severity) `generate()`'s two anti-repetition guards used
     `float("-inf")` instead of the `-100.0` convention used everywhere else
     in the model — now consistent.

**Result after fixes** (`collapse_check.py` + `eval.py`, see
`MODEL_VALIDATION_REPORT.md` for full tables): representation collapse is
reduced, monotonically with more training data, but not yet fully resolved
— a high-contrast image (chest CT) shows large token-diversity gains at
both 2,000 and 5,000 training images, while a lower-contrast image
(angiogram) stays fully collapsed at 2,000 images and only starts improving
at 5,000. Caption diversity and BLEU-1..3 improve monotonically with data
scale; the model remains far below the pretrained-encoder CNN-LSTM baseline,
as expected for a from-scratch encoder at this data scale.

Open items: scale training further (10k–60k images) now that the recipe is
verified; consider an auxiliary concept-classification loss
(`train_concepts.csv` is present in the dataset but unused) to give the
encoder a more direct gradient signal for low-contrast images; `irdid.ipynb.py`
is a dead, unimported duplicate of `vocab.py` — safe to delete.
