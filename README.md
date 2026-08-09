# Swin Transformer Image Captioning — ROCOv2 (manual implementation)

Every component is written from scratch with plain `torch.nn` — no `timm`,
no `transformers`, no pretrained weights, no third-party captioning repo.

## Files

| File | What it is |
|---|---|
| `vocab.py` | **Block 0** — manual regex tokenizer + vocab builder |
| `dataset.py` | ROCOv2 `Dataset` that loads images + captions, capped at 2000 samples |
| `swin_model.py` | **The Swin encoder**, block by block: PatchEmbed → WindowAttention → SwinBlock (W-MSA/SW-MSA) → PatchMerging → BasicLayer stages → SwinEncoder |
| `decoder_model.py` | **The caption decoder**: PositionalEncoding → manual MultiHeadAttention → DecoderLayer (masked self-attn + cross-attn + FFN) → CaptionDecoder |
| `caption_model.py` | Wires encoder + decoder into `SwinCaptioningModel`, plus greedy `.generate()` |
| `train.py` | Training loop, already pointed at your local paths, `MAX_SAMPLES = 2000` |
| `generate.py` | Loads a checkpoint and captions a single image |

## Before running

Your CSV column names might not be exactly `ID` / `Caption`. Check with:

```python
import pandas as pd
print(pd.read_csv("/Users/nagashiva/Downloads/rocov2/train_captions.csv").columns)
```

Then edit `IMAGE_COL` / `CAPTION_COL` at the top of `dataset.py` if needed.

## Run

```bash
cd swin_caption
pip install torch torchvision pandas pillow
python train.py
```

This trains on the first 2000 rows of `train_captions.csv` / `train_images`,
validates on 200 rows of `valid_captions.csv` / `valid_images`, and saves a
checkpoint per epoch into your `rocov2` folder as `swin_caption_epochN.pt`.

To caption a new image:

```bash
python generate.py /path/to/image.jpg /Users/nagashiva/Downloads/rocov2/swin_caption_epoch10.pt
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
(padding ignored).

## Notes / things to tune

- `BATCH_SIZE=16`, `EPOCHS=10` in `train.py` are starting points — with only
  2000 images you may want to raise epochs and/or add light data augmentation.
- Training a Swin encoder from random init (no pretraining) on 2k images is
  small-data territory — expect it to need more epochs than an ImageNet-
  pretrained backbone would. If quality is too low, consider raising
  `max_samples` later once the pipeline is confirmed working.
- `MultiHeadAttention` in `decoder_model.py` is hand-written (not
  `nn.MultiheadAttention`) to match the manual style of the Swin blocks.
