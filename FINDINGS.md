# Codebase & dataset analysis — swin 2k / ROCOv2

Analysis based on this codebase and the full ROCOv2 dataset export
(including `swin_caption_epoch*.pt` / `_best.pt` / `_last.pt` checkpoints,
which were inspected directly).

## 1. Dataset sizes (ground truth)

| Split | Images on disk | Caption rows | Match? |
|---|---|---|---|
| train | 59,958 (`train_images/train/*.jpg`) | 59,958 (`train_captions.csv`) | 1:1 ✓ |
| valid | 9,904 (`valid_images/valid/*.jpg`) | 9,904 (`valid_captions.csv`) | 1:1 ✓ |
| test | 9,927 (`test_images/test/*.jpg`) | 9,927 (`test_captions.csv`) | 1:1 ✓ |

No missing files, no orphaned captions, no column mismatch — `dataset.py`'s
`IMAGE_COL="ID"` / `CAPTION_COL="Caption"` already match the real CSV
headers exactly.

Current `train.py` only *uses* a slice of this: `TRAIN_SAMPLES = 15000` of
the 59,958 available, `VALID_SAMPLES = 2000` of the 9,904 available, and
the **test set (9,927 images) is never touched by any script** — there was
no evaluation path against it before `eval.py` was added here.

## 2. The main finding: the checkpoint being tested is stale

Running `generate.py` against `swin_caption_best.pt` produced the *same*
caption for multiple different test images:

```
axial computed tomography image of the abdomen shows the small bowel
loops of the small air fluid levels
```

Loading the checkpoint's metadata directly (`epoch`, `val_loss`,
`vocab_itos`) without needing the full model gives a concrete explanation:

| file | epoch | val_loss | vocab size |
|---|---|---|---|
| `swin_caption_best.pt` | 3 | 4.997 | **2,682** |
| `swin_caption_last.pt` | 11 | (not recorded) | 2,682 |
| `swin_caption_epoch1/6/20/40/60.pt` | 1/6/20/40/60 | — | 2,682 (stale, per-epoch saving isn't even in the current `train.py`) |

Rebuilding the vocabulary directly from `train_captions.csv` at different
`max_samples` reproduces these vocab sizes exactly:

| max_samples | vocab size (min_freq=2) |
|---|---|
| **2,000** | **2,682** ← matches the checkpoint |
| 5,000 | 4,493 |
| 8,000 | 5,780 |
| 10,000 | 6,439 |
| 15,000 | 7,886 |

`swin_caption_best.pt`'s vocab size (2,682) matches training on exactly
**2,000** samples — not the 15,000 the current `train.py` docstring
describes ("v3: scaled up from 2k to 15k ... to fix encoder representation
collapse"). In other words: **the fix described in the script was never
actually run to produce this checkpoint.** You're testing an artifact from
the old 2k-sample configuration, which is exactly the setup the docstring
says caused collapse in the first place.

The `swin_caption_epochN.pt` files (per-epoch saving up to epoch 60) also
don't match current `train.py` at all — that script only ever writes
`swin_caption_best.pt` and `swin_caption_last.pt`. Those per-epoch files
are leftovers from an even older version of the script (matching the
original README's description of "saves a checkpoint per epoch").

**Action:** delete/archive the old `.pt` files and re-run the current
`train.py` (15k/2k, 60 epochs, early stopping) from scratch to get a
checkpoint that actually reflects the "v3" fix.

## 3. Even within the old run, training plateaued almost immediately

Best validation loss was logged at **epoch 3** (val_loss ≈ 4.997) and
never improved again for at least 8 more epochs before stopping. A
cross-entropy val loss of ~5.0 with label smoothing 0.1 is consistent with
a model that's stopped using the image at all and settled into predicting
a generic, frequency-weighted caption regardless of input — early,
degenerate convergence, not a slow decline.

## 4. The ground-truth data itself is not the problem

Ruled out explicitly, in case it looked like a data quality issue:

- **Caption diversity is high.** 1,998/2,000 (99.9%) unique captions in
  the first 2,000 training rows; 14,956/15,000 (99.7%) unique in the first
  15,000. The model's repeated output is not literally the single most
  common training caption — no caption in the first 2,000 rows repeats
  more than twice.
- **No ordering bias.** Modality keyword frequency (`ct`, `mri`, `x-ray`,
  `chest`, `abdomen`, `bowel`, etc.) in the first 15,000 rows is within ~1
  percentage point of the remaining 44,958 rows — the CSV isn't grouped by
  modality in a way that would starve the training slice of diversity.
- **Caption length varies enormously**, though: mean 21 words, std 15,
  range 1–181 words, and **10.4% of training captions exceed the 38-word
  budget** (`MAX_LEN=40` minus `<sos>`/`<eos>`) and get truncated. This
  isn't a bug, but it is a real difficulty factor: a huge structural
  variance in caption complexity, learned from a small, non-pretrained
  model, makes the "predict something generically plausible" shortcut even
  more attractive to the optimizer.

## 5. Code-level issues found

*(Status: the two fixes below have been applied directly to `train.py` and
`decoder_model.py`, and verified — see "Verification" at the end of this
section.)*

**AdamW weight decay applied to every parameter. — FIXED**
`train.py`:
```python
optimizer = torch.optim.AdamW(model.parameters(), lr=PEAK_LR, weight_decay=WEIGHT_DECAY)
```
This decays *all* parameters uniformly, including LayerNorm weight/bias,
linear biases, and `WindowAttention.relative_position_bias_table`. The
Swin paper's own training recipe (and most Transformer recipes) explicitly
excludes 1-D parameters (norms, biases, position/bias tables) from weight
decay, because decaying a LayerNorm's scale toward zero or eroding the
learned relative-position bias distorts training in ways unrelated to
generalization. Fix: split `model.parameters()` into two param groups (by
`p.ndim <= 1` or by name) and set `weight_decay=0` on the norm/bias group.

Implemented as `build_param_groups()` in `train.py`, which splits on
`param.ndim <= 1` (catches all biases and LayerNorm weight/bias) plus an
explicit name check for `relative_position_bias_table` (2-D, so the ndim
rule alone would miss it). `AdamW` now takes this param-group list instead
of `model.parameters()`.

**Inconsistent masking constant (`-inf` vs `-100.0`). — FIXED**
`swin_model.py`'s shift-window mask deliberately fills disallowed
attention with `-100.0`:
```python
attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0))
```
but `decoder_model.py`'s causal mask used literal `-inf`:
```python
scores = scores.masked_fill(mask == 0, float("-inf"))
```
Now changed to `float(-100.0)` to match. The `-100.0` choice in the encoder looks like a deliberate workaround for
a known PyTorch MPS-backend issue where `softmax` over rows containing
`-inf` can produce `NaN`. Given you're training on an M-series Mac (MPS
backend), the decoder's `-inf` usage is worth changing to a large finite
negative value (e.g. `-100.0` or `-1e4`) for consistency and safety, even
though in this specific case every causal-mask row always has at least one
unmasked (diagonal) position, so it's a lower-severity risk than it would
be in code where a fully-masked row is possible.

**Dead code.**
`irdid.ipynb.py` is a byte-for-byte duplicate of `vocab.py`'s `Vocab`
class, not imported anywhere in the project. Harmless, but worth deleting
so nobody edits the wrong copy later.

**No evaluation script existed.**
Only `train.py`'s running cross-entropy loss and `generate.py`'s
single-image output existed before — no BLEU/ROUGE/CIDEr, and the test
split was never used. `eval.py` + `metrics.py` (added alongside this
report) close that gap: BLEU-1..4, ROUGE-L, and CIDEr-D implemented from
scratch (no `nltk`/`pycocoevalcap`/Java dependency), plus a "% distinct
captions generated" figure that directly flags the collapse behavior you
observed — run it against any checkpoint and split.

**Verification of the two fixes above.** `python3 -m py_compile train.py
decoder_model.py` passes on both edited files. `build_param_groups()`'s
filtering logic was additionally checked against mock parameters shaped
like the real model's (2-D conv/linear weights, 1-D biases/LayerNorm
params, and the 2-D `relative_position_bias_table` that needs the explicit
name-based exclusion since its `ndim` alone wouldn't flag it) — all land in
the correct decay/no-decay group. This confirms the code is correct and
importable; it does not by itself prove the *training outcome* improves,
since that can only be observed by actually rerunning `train.py`.

## 6. Benchmark comparison

The paper you linked (Pranay Kumar et al., *"Image Captioning Generator
Using CNN and LSTM,"* IJRASET 2022) trains a DenseNet/Xception+LSTM model
on **Flickr8k** (general photos, not biomedical) and its results section is
only presented as embedded images in the PDF, not extractable text, so
exact numbers from that specific paper aren't available. Its own related-work
section, however, points at the same standard metric set your `eval.py`
now computes (BLEU, and the field generally reports METEOR/ROUGE/CIDEr
alongside it).

For an apples-to-apples target, the more relevant baseline is the actual
**ROCOv2** dataset paper / ImageCLEFmedical Caption 2023 challenge it was
built for:

| Model | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | ROUGE | CIDEr |
|---|---|---|---|---|---|---|
| CNN-LSTM (DenseNet169) baseline on ROCOv2 | 0.586 | 0.498 | 0.347 | 0.127 | 0.453 | 0.463 |
| General ROCOv2 baseline (different run) | 0.184* | – | – | – | 0.233 | 0.203 |

\* reported as a single aggregate BLEU, not split by n-gram order.

Sources: [ROCOv2: Radiology Objects in COntext Version 2 — Scientific
Data (Nature)](https://www.nature.com/articles/s41597-024-03496-6),
[ROCOv2 — PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11208523/).

For a quick sense of how far off the *current* collapsed checkpoint is:
scoring the fixed collapsed caption against the first 20 real
`test_captions.csv` entries gives BLEU-1 ≈ 0.14, BLEU-4 ≈ 0.00,
ROUGE-L ≈ 0.13, CIDEr-D ≈ 0.17 — well below the ROCOv2 CNN-LSTM baseline
across the board, which is exactly what you'd expect from a model that
isn't actually looking at the image. Once you retrain on the current
15k/2k config — now with the weight-decay grouping fix applied — run
`eval.py --split test` for real numbers to compare against the table.

## 7. Recommended next steps, in priority order

1. ~~Fix the AdamW weight-decay grouping~~ and ~~the `-inf`/`-100.0`
   masking inconsistency~~ — **done**, see Section 5.
2. Re-run `train.py` as currently written (15k/2k, 60 epochs, early
   stopping) — the checkpoint you tested was never actually produced by
   this configuration.
3. Run `eval.py --split valid` periodically during development and
   `--split test` for the final number, and watch the "% distinct
   captions" figure specifically — if it stays low even after retraining
   on 15k samples, that's a strong signal the collapse is more structural
   (e.g. needs a pretrained encoder, or scheduled sampling / attention
   regularization) rather than just a data-scale issue.
4. Delete `irdid.ipynb.py` and the stale `swin_caption_epochN.pt` files to
   avoid confusion about which checkpoint is current.
