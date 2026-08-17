# MODEL VALIDATION REPORT
## Swin Transformer + Transformer Decoder — ROCOv2 Radiology Image Captioning

**Audit date:** 2026-08-16
**Checkpoint audited:** `swin_caption_best.pt` (uploaded), SHA256 `61a55a75…a2a2c7`, 386 MB
**Codebase audited:** `swin_model.py`, `decoder_model.py`, `caption_model.py`, `dataset.py`, `vocab.py`, `train.py`, `generate.py`, `eval.py`, `metrics.py`

---

### How this audit was performed — and why you can trust the numbers

The sandbox running this audit has **no PyTorch** (no network access to install it; the project's own `.venv` holds macOS/arm64 binaries that cannot execute on Linux). Rather than declare the dynamic phases untestable, I did the following:

1. Wrote a **torch-free unpickler** that reads the real float32 tensors out of the `.pt` zip archive.
2. Wrote a **faithful NumPy re-implementation** of `swin_model.py` and `decoder_model.py`, mirroring your code operation-for-operation (same slice orders, same mask constants, same `-100.0` fill, same guard logic in `generate()`).
3. Ran the **real checkpoint weights** on the **real images**.

**Validation that this substitute is faithful:** my NumPy pipeline reproduced *both* of your reported captions character-for-character:

| Image | Your reported output | NumPy audit output | Match |
|---|---|---|---|
| `ROCOv2_2023_test_007757` | `ct scan of the abdomen showing a mass in the right lobe of the liver` | `ct scan of the abdomen showing a mass in the right lobe of the liver` | **EXACT** |
| `ROCOv2_2023_test_000004` | `postoperative panoramic radiograph` | `postoperative panoramic radiograph` | **EXACT** |

Reproducing two different, long, non-trivial outputs exactly is strong evidence the re-implementation is correct. All numbers below are therefore **measured, not estimated**. The one approximation is GELU's `erf` (Abramowitz–Stegun 7.1.26, max abs error 1.5e-7) — far too small to affect any conclusion.

Labels used throughout: **FACT** (measured/read), **INFERENCE** (reasoned from facts), **RECOMMENDATION**.

---

# 1. Executive Summary

## Overall status: **BUGS FOUND — DO NOT TRUST RESULTS**

The implementation is, to a genuinely impressive degree, **architecturally correct**. I could not find a single wrong tensor operation: window partition/reverse round-trips exactly, the relative-position index uses all 169 offsets with zero out-of-bounds, patch merging matches official Swin's slice order, the causal mask leaks no future tokens, cross-attention wires Q/K/V correctly, and target shifting is off-by-one-free. The checkpoint loads with zero missing and zero unexpected keys.

**But the trained model does not work, and the reason is not the architecture.** Three measured findings:

1. **The Swin encoder has collapsed.** Of all variance in the 49×768 output, **only 0.225% distinguishes one spatial position from another** — 99.775% is a constant pattern shared by all 49 tokens. Mean pairwise cosine similarity between tokens: **0.9977**.

2. **The encoder is nearly blind to image content.** Cosine similarity between the real chest-CT's representation and **pure random noise** is **0.9987** — *higher* than its similarity to another real medical image (**0.9793**). Feeding an all-black image produces the fluent, confident caption *"mri of the right foot showing a mass in the left side of the left femur."* A 50%-grey image produces the **exact same caption** as a real angiogram.

3. **Training stopped after 3 epochs, in both runs, before the encoder ever trained.** Encoder weight standard deviations are still at **1.00–1.06×** their random initialization; every encoder LayerNorm weight is still ≈1.000. The 2,000-image run and the 15,000-image run *both* recorded best val_loss at epoch 3 (4.9970 and 4.9936) — scaling the data 7.5× changed nothing, which rules out "not enough data" as the primary cause.

**Measured validation performance (120 validation images, first-ever real numbers for this model):**

| Metric | This model | ROCOv2 CNN-LSTM baseline |
|---|---|---|
| BLEU-1 | **0.0774** | 0.586 |
| BLEU-4 | **0.0141** | 0.127 |
| ROUGE-L | **0.1483** | 0.453 |
| CIDEr-D | **0.1421** | 0.463 |
| Distinct captions | **14 / 120 (11.7%)** | — |

Three caption templates account for **77.5%** of all 120 outputs.

**Answer to "Can I trust this model's current results?" — NO** for any performance claim. **YES** for the architecture implementation itself, which passed every correctness test I could devise. Details in §18.

---

# 2. Architecture Summary

**FACT** — all figures below read directly from the checkpoint's stored tensors and confirmed by executed forward pass.

| Component | Configuration | Verified how |
|---|---|---|
| **Encoder** | Swin-T: patch 4, embed 96, window 7, depths (2,2,6,2), heads (3,6,12,24) | Executed forward pass + checkpoint shapes |
| Heads per stage | (3, 6, 12, 24) | `relative_position_bias_table` second dim = 3/6/12/24 |
| **Decoder** | 6 layers, d_model 768, 8 heads, **FFN 3072** | Checkpoint `ff.0.weight` = (3072, 768) |
| Vocabulary | 7,886 words (min_freq 2, from 15,000 train captions) | Rebuilt independently → exact match |
| Max caption length | 40 | `decoder.pos_enc.pe` = (1, 40, 768) |
| Input shape | (B, 3, 224, 224) | Executed |
| Final visual representation | (B, 49, 768) | Executed |
| **Total parameters** | **96,381,512** | Summed from checkpoint |
| Checkpoint contents | `model_state`, `vocab_itos`, `epoch`, `val_loss` | Unpickled |

> **[WARNING] SPEC DISCREPANCY — FFN dimension.** Your written specification says `ff_dim = 2048`. The code says otherwise: `caption_model.py` passes `ff_dim=d_model * 4 = 3072` to `CaptionDecoder`, and the checkpoint confirms `(3072, 768)`. **The code is the source of truth: your paper must say 3072, not 2048.** This is the only numerical discrepancy between your spec and the implementation — everything else (d_model 768, 6 layers, 8 heads, depths, heads, window 7, 49×768) matches exactly.

---

# 3. Code Audit

## [PASS] — verified correct

| # | Component | File | Evidence |
|---|---|---|---|
| P1 | `window_partition` / `window_reverse` | `swin_model.py:32-43` | Round-trip on all 4 stage resolutions returns the **bit-exact** original array. 56→64 windows, 28→16, 14→4, 7→1. |
| P2 | Relative-position index construction | `swin_model.py:68-78` | Index range [0, 168], table allocated 169 rows → **no out-of-bounds**; all 169 offsets used; diagonal constant (=84, the (0,0) offset); translation-invariant (offset (0,1) at position (0,0) yields same index as at (3,3)). |
| P3 | Relative-position bias application | `swin_model.py:94-96` | Matches official Swin exactly: gather → `view(N,N,-1)` → `permute(2,0,1)` → `unsqueeze(0)` broadcast add. |
| P4 | Shift-window mask | `swin_model.py:150-171` | 9 region labels; **15/64** windows masked (only edge/wrap-around windows), 49/64 interior windows fully unmasked — exactly correct. Mask is symmetric. |
| P5 | Encoder mask application | `swin_model.py:98-101` | **Additive** (`attn + mask`), which is more numerically robust than `masked_fill`. Matches official Swin. |
| P6 | `PatchMerging` slice order | `swin_model.py:220-224` | `[x0,x1,x2,x3] = [(0::2,0::2),(1::2,0::2),(0::2,1::2),(1::2,1::2)]` — identical to official Swin. All three transitions verified: 96→192, 192→384, 384→768; 56→28→14→7. |
| P7 | Head-dimension divisibility | all stages | 96/3, 192/6, 384/12, 768/24 all = 32 exactly. No silent truncation. |
| P8 | **Causal mask** | `decoder_model.py:100` | Executed test: token *t* attends to exactly {0…*t*}. **Zero future-token leakage.** |
| P9 | Cross-attention Q/K/V wiring | `decoder_model.py:76` | `cross_attn(x, memory, memory)` → Q from caption, K/V from image. Executed shapes: Q (1,8,T,96), K/V (1,8,49,96), scores (1,8,T,49). Correct. |
| P10 | **Target shifting** | `caption_model.py:29` + `train.py:102` | `tgt_in = captions[:, :-1]`, `targets = captions[:, 1:]` — complementary halves, **no off-by-one**. |
| P11 | Loss reshaping | `train.py:104` | `logits.reshape(-1, V)` → (B·39, 7886); `targets.reshape(-1)` → (B·39). Correct pairing. |
| P12 | `ignore_index` / label smoothing | `train.py:89` | `ignore_index=vocab.pad_id` correct; `label_smoothing=0.1` applied. |
| P13 | **Vocabulary built from training data only** | `train.py:71` | `build_vocab_from_csv(TRAIN_CSV, ...)` — validation/test captions **never** influence the vocabulary. No leakage vector here. |
| P14 | Image↔caption pairing | `dataset.py:56-67` | `__getitem__` reads `row[IMAGE_COL]` and `row[CAPTION_COL]` from the **same row**; DataLoader shuffles indices, not columns. **Pairing cannot desynchronize.** |
| P15 | Validation transform | `train.py:76` | `valid_ds` passes no transform → falls back to `dataset.py`'s un-augmented default. Augmentation correctly excluded from validation. |
| P16 | Preprocessing consistency | `train.py`/`dataset.py`/`generate.py`/`eval.py` | Resize (224,224), ImageNet mean/std, `.convert("RGB")`, `ToTensor` — **byte-identical across all four files**. |
| P17 | Checkpoint reconstructs architecture | — | Expected keys generated from code = 343; actual = 343. **0 missing, 0 unexpected, all shapes match.** `load_state_dict(strict=True)` would succeed. |
| P18 | Vocab restored from checkpoint at inference | `generate.py:27-36`, `eval.py:54-63` | Restores `itos` from `ckpt["vocab_itos"]`, not rebuilt from CSV. Guarantees train/inference vocab identity. |
| P19 | `model.eval()` before generation | `generate.py:45`, `eval.py:102` | Dropout correctly disabled at inference. |
| P20 | Weight-decay parameter grouping | `train.py:47-65` | Correctly excludes 1-D params **and** `relative_position_bias_table` (2-D, would be missed by ndim check alone). Matches Swin's published recipe. |
| P21 | No NaN / Inf anywhere | checkpoint | Scanned all 343 tensors: **zero** NaN, **zero** Inf. |
| P22 | Metric implementations | `metrics.py` | See §12 — all behave monotonically and correctly. |
| P23 | No missing images | dataset | 0 missing among the 15,000 train and 2,000 valid IDs actually used. |

## [BUG] — demonstrable defects

### BUG-1 — [SEVERITY: CRITICAL] No weight initialization for any layer except the position-bias table

- **File / class:** `swin_model.py`, all classes; `decoder_model.py`, all classes
- **Line:** `swin_model.py:66` is the *only* `nn.init.*` call in the entire codebase
- **Expected:** Official Swin calls `self.apply(_init_weights)`, applying `trunc_normal_(std=0.02)` to **every** `nn.Linear` and `ones_`/`zeros_` to every `nn.LayerNorm`.
- **Actual:** Only `relative_position_bias_table` is initialized. Every other Linear falls back to PyTorch's default `kaiming_uniform_` → `U(−1/√fan_in, +1/√fan_in)`, std = 1/√(3·fan_in).
- **Evidence (measured from checkpoint):** early-stage encoder Linears have std **0.0598** (fan_in 96) versus the 0.02 that Swin's recipe prescribes — **~3× too large**. Deeper layers happen to land near 0.02 only because fan_in grows.

| Tensor | fan_in | PyTorch-default std | Swin-recipe std | Actual in checkpoint |
|---|---|---|---|---|
| `layers.0.blocks.0.attn.qkv.weight` | 96 | 0.0589 | 0.02 | **0.0598** |
| `layers.0.blocks.0.mlp.fc1.weight` | 96 | 0.0589 | 0.02 | **0.0592** |
| `layers.1.blocks.0.attn.qkv.weight` | 192 | 0.0417 | 0.02 | 0.0434 |
| `layers.3.blocks.0.attn.qkv.weight` | 768 | 0.0208 | 0.02 | 0.0214 |

- **Severity:** Critical. Over-scaled early-layer initialization in a 12-block residual stack causes activation variance to compound through the residual path; combined with BUG-2 there is nothing damping it.
- **Recommended fix:** add a `_init_weights` method applying `trunc_normal_(std=0.02)` to Linear weights, `zeros_` to Linear biases, `ones_`/`zeros_` to LayerNorm, called via `self.apply(...)` in `SwinEncoder.__init__` and `CaptionDecoder.__init__`. **Does not violate from-scratch constraints** — it is ~8 lines of `nn.init` calls.

### BUG-2 — [SEVERITY: HIGH] No stochastic depth (DropPath) and no attention/projection dropout in the encoder

- **File / class:** `swin_model.py`, `SwinBlock.forward` (lines 173-201), `WindowAttention.forward`
- **Expected:** Official Swin wraps both residual branches in `DropPath(p)` with `p` increasing linearly across the 12 blocks (default max 0.1 for Swin-T), and exposes `attn_drop`/`proj_drop`.
- **Actual:** `x = shortcut + x` and `x = x + self.mlp(...)` — no stochastic depth. `WindowAttention` has no dropout at all. `Mlp` accepts a `drop` argument but `SwinBlock` passes the default `0.0`.
- **Evidence:** grep for `DropPath` / `drop_path` in `swin_model.py` → zero matches. `Mlp(dim, int(dim*mlp_ratio), drop)` is called with `drop=0.0` from `SwinBlock.__init__`'s default.
- **Severity:** High. The encoder is the randomly-initialized half and receives the weakest gradient; it is precisely the part that needs depth-wise regularization.
- **Recommended fix:** implement `DropPath` (≈10 lines, pure `torch`) with a linearly increasing schedule. **From-scratch compliant.**

### BUG-3 — [SEVERITY: MEDIUM] Masking constant inconsistency inside `generate()`

- **File:** `caption_model.py`, lines 45 and 57
- **Expected:** the codebase deliberately standardized on `-100.0` (not `-inf`) to avoid a known MPS softmax NaN — this is documented in `swin_model.py:169` and was applied as a fix in `decoder_model.py:47`.
- **Actual:** `generate()` still writes `float("-inf")` in both anti-repetition guards.
- **Evidence:** `step_logits.scatter_(1, prev_token.unsqueeze(1), float("-inf"))` and `step_logits[b, banned_next] = float("-inf")`.
- **Why it is not currently fatal:** these logits go only to `argmax`, never `softmax`, so no NaN arises **today**.
- **Why it is still a bug:** the moment anyone applies `softmax` to `step_logits` — which any probability trace, beam search, or sampling extension will do — the MPS issue you already fixed twice returns. Additionally, if trigram-blocking ever bans the entire vocabulary, `argmax` over all-`-inf` silently returns index 0 = `<pad>`.
- **Recommended fix:** replace both with `-100.0` for consistency with the rest of the codebase.

## [WARNING] — real issues, evidence-backed

### WARN-1 — [HIGH] The learning-rate schedule terminates training at its worst possible moment

**FACT.** `LambdaLR` calls `step()` once at construction, and `train.py` calls `scheduler.step()` *after* each epoch. The resulting actual LR per training epoch:

| Epoch | LR | Note |
|---|---|---|
| 1 | 1.0e-4 | |
| 2 | 2.0e-4 | |
| **3** | **3.0e-4** | **← best val_loss recorded here, in BOTH the 2k and 15k runs** |
| 4 | 4.0e-4 | |
| 5 | 5.0e-4 | warmup complete, peak reached |
| 6–9 | ~5.0e-4 | cosine decays only ~1% over these epochs |

With `EARLY_STOP_PATIENCE = 6`, training halts at roughly epoch 9 — while LR is still ≈4.96e-4, essentially peak. **The model never experiences the low-LR end of the cosine schedule at all.**

**INFERENCE.** Two independent runs at 2,000 and 15,000 images both peaking at epoch 3 (LR 3e-4) and degrading thereafter is a textbook signature that `PEAK_LR = 5e-4` is too high for this model — plausibly *because of* BUG-1 and BUG-2. This is the single most actionable finding in the audit.

### WARN-2 — [MEDIUM] `masked_fill(mask == 0, -100.0)` leaks when legitimate scores fall below ≈−95

**FACT** (measured): with 5 masked positions at −100 and one legitimate score *s*:

| legitimate score *s* | P(real token) | probability leaked to masked positions |
|---|---|---|
| 0.0 | 1.000000 | 0.000000 |
| −10.0 | 1.000000 | 0.000000 |
| −95.0 | 0.967408 | **0.032592** |
| −105.0 | 0.001346 | **0.998654** |

The encoder is immune (it *adds* the mask; the offset is always −100 relative to real scores). The **decoder's `masked_fill` is not** — it *sets* an absolute value. Currently harmless because trained scores sit in roughly [−10, 10], but it is a latent correctness cliff, not a safety margin you chose deliberately.

### WARN-3 — [MEDIUM] 10.4% of training captions are silently truncated

**FACT.** `MAX_LEN = 40`, and `Vocab.encode` keeps only `max_len - 2 = 38` word tokens.

| Split (as used) | mean length | median | max | **truncated (>38 tokens)** |
|---|---|---|---|---|
| train[:15000] | 21.2 | 17 | 181 | **1,557 (10.4%)** |
| valid[:2000] | — | — | — | **237 (11.8%)** |

Truncated captions lose their ending, so the model is trained on examples whose `<eos>` was cut — teaching it that long captions simply stop. **INFERENCE:** contributes to the measured generated-caption length of 10.0 tokens versus reference 23.4.

### WARN-4 — [LOW] Small validation caption leakage

**FACT.** 8 of 2,000 validation rows (**0.40%**) have a caption string that also appears verbatim in `train[:15000]` (e.g. `"panoramic radiograph"`, `"preoperative chest x ray"`). Across full splits: 56 train∩valid, 52 train∩test, 29 valid∩test unique captions. These are short generic captions attached to genuinely different images, so this is inherent to ROCOv2, not a bug in your splitting — but it must be disclosed in the paper.

### WARN-5 — [LOW] Encoder is Pre-LN, decoder is Post-LN

**FACT.** `swin_model.py:199-200`: `x = shortcut + x; x = x + self.mlp(self.norm2(x))` → **Pre-LN**. `decoder_model.py:75-77`: `x = self.norm1(x + ...)` → **Post-LN**. Both are individually valid published designs; mixing them in one model is an internal inconsistency worth stating explicitly in the paper rather than leaving for a reviewer to find. Post-LN decoders are known to be more sensitive to learning rate — relevant to WARN-1.

### WARN-6 — [LOW] Checkpoint cannot resume training

**FACT.** Saved keys: `model_state`, `vocab_itos`, `epoch`, `val_loss`. **No optimizer state, no scheduler state, no config.** Any interrupted run must restart from epoch 1 with fresh AdamW moments.

### WARN-7 — [LOW] CIDEr is scaled ×10 and IDF is computed on the evaluation set

**FACT.** `metrics.py:134` — `10.0 * sum(per_n) / n_max`. A perfect match scores **10.0**, not 1.0. Also, `corpus_cider_d` computes IDF from the *reference set being evaluated*, so scores depend on evaluation-set size and are not comparable across different `--n` values. Document both, or scores will be misread against the 0.463 literature baseline.

### WARN-8 — [INFO] `train.py` says 15000, your brief says 2000

`TRAIN_SAMPLES = 15000`, `VALID_SAMPLES = 2000` in the committed `train.py`. The uploaded checkpoint (vocab 7,886) is from that 15k config. The 2k pilot is not reproducible from the current code without editing constants.

## [UNVERIFIED] / NOT TESTED

| Item | Reason |
|---|---|
| **METEOR** | **NOT IMPLEMENTED** — `metrics.py` provides BLEU-1..4, ROUGE-L, CIDEr-D only. Requires WordNet synonym matching; cannot be reported. |
| Phase 15 gradient-flow statistics | **NOT TESTED — no autograd available.** NumPy re-implementation is forward-only. Weight-drift analysis (§8) is offered as a substitute measurement. |
| Phase 16 tiny 10-example overfit test | **NOT TESTED — requires backpropagation.** Script provided in §16; this is your single highest-priority next action. |
| Phase 18 per-epoch training curves | **NOT TESTED** — `train.py` prints metrics but persists no log file; only `epoch` and `val_loss` of the best checkpoint survive. |
| Encoder/decoder head counts at runtime | Confirmed indirectly and conclusively via `relative_position_bias_table` second dim = (3,6,12,24). |
| Corrupted images | 0 missing files; all 122 images opened successfully by PIL. Full-corpus scan not performed. |

---

# 4. Tensor Shape Verification — EXECUTED

**FACT.** Produced by actually running the checkpoint weights on `ROCOv2_2023_test_007757.jpg`.

| Stage | Executed shape | Expected | Match |
|---|---|---|---|
| Input | (1, 3, 224, 224) | (B,3,224,224) | ✓ |
| PatchEmbed | **(1, 3136, 96)** | B×3136×96 | ✓ |
| Stage 1 (depth 2, heads 3) | **(1, 3136, 96)** | B×3136×96 | ✓ |
| PatchMerging 1 | **(1, 784, 192)** | B×784×192 | ✓ |
| Stage 2 (depth 2, heads 6) | **(1, 784, 192)** | B×784×192 | ✓ |
| PatchMerging 2 | **(1, 196, 384)** | B×196×384 | ✓ |
| Stage 3 (depth 6, heads 12) | **(1, 196, 384)** | B×196×384 | ✓ |
| PatchMerging 3 | **(1, 49, 768)** | B×49×768 | ✓ |
| Stage 4 (depth 2, heads 24) | **(1, 49, 768)** | B×49×768 | ✓ |
| Final LayerNorm (= `memory`) | **(1, 49, 768)** | B×49×768 | ✓ |

**Every expected dimension in your specification is confirmed by execution.** No NaN, no Inf.

**Spatial mapping (code-derived, exact):** token *t* ↔ grid (row = *t*//7, col = *t*%7) ↔ pixel block rows [32·row, 32·row+32), cols [32·col, 32·col+32). Row-major order is preserved because `PatchEmbed`'s `flatten(2).transpose(1,2)`, `PatchMerging`'s strided slicing, and `window_reverse`'s inverse permutation all preserve it — verified by the bit-exact round-trip test (P1).

---

# 5. Attention Verification

## W-MSA (Phase 3) — **PASS**
**FACT.** Stage 1: 56×56 grid → **64 windows** of **49 tokens** each (round-trip exact). Head dim 96/3 = **32**. Per head per window: Q (49,32), K (49,32), V (49,32) → **QKᵀ = (49,49)**. Confirmed by executed forward pass. Scaling `head_dim ** -0.5` applied to `q` before the matmul (`swin_model.py:91`), equivalent to dividing the product.

## SW-MSA (Phase 4) — **PASS**
**FACT.** `shift_size = window_size // 2 = 3` for window 7 (`swin_model.py:239`, odd-indexed blocks). Verified: `torch.roll(-3,-3)` → partition → attention with mask → `window_reverse` → `torch.roll(+3,+3)`. Mask built from 9 region labels; **15/64 windows carry any mask**, 49/64 interior windows fully unmasked — exactly the expected pattern for a 56×56 grid with 8×8 windows (only the last row/column of windows wrap). Mask is symmetric. Masked values are added as −100.0 *before* softmax, so `e^−100 ≈ 3.7e−44` → effectively zero probability. **Masking is effective.**

## Relative position bias (Phase 5) — **PASS**
**FACT.** Table shape (169, heads) with 169 = (2·7−1)² ✓. Index range [0, 168] — **no out-of-bounds** against a 169-row table. All 169 offsets are used. Diagonal is constant (index 84 = the (0,0) offset). Translation-invariant. Bias shape after `view(N,N,-1).permute(2,0,1).unsqueeze(0)` is (1, heads, 49, 49), broadcasting correctly onto attention (B_, heads, 49, 49). Checkpoint confirms per-stage tables of (169,3), (169,6), (169,12), (169,24).

## Decoder self-attention (Phase 8) — **PASS**
**FACT.** Executed test on T=6: token 0 → {0}; token 1 → {0,1}; … token 5 → {0..5}. **Zero future-token access.** See §3 P8.

## Cross-attention (Phase 9) — **PASS on wiring, FAIL on learned behavior**
**FACT (shapes).** Executed: Q (1,8,T,96), K (1,8,49,96), V (1,8,49,96), scores **(1,8,T,49)**, output (1,8,T,96) → reshaped (1,T,768). All correct.

**FACT (behavior).** Measured cross-attention entropy over the 49 image tokens, where ln(49) = 3.8918 nats is perfectly uniform:

| Image | step 0 | steps 1+ (mean) |
|---|---|---|
| `test_007757` | 1.371 (35.2% of uniform) | **3.860 (99.2% of uniform)** |
| `test_000004` | 3.512 (90.2% of uniform) | **3.785 (97.2% of uniform)** |

**INFERENCE.** After the first generated token, cross-attention is **essentially uniform** — mathematically equivalent to adding one fixed, image-averaged bias vector to every caption position. The mechanism that is supposed to ground language in vision is contributing almost no position-specific information. Combined with §10 (the 49 tokens are near-identical anyway), even the peaked step-0 attention conveys little: attending sharply to token 17 versus averaging all 49 gives nearly the same vector when cos(token_i, token_j) = 0.9977.

---

# 6. Data Pipeline Audit (Phase 12)

**FACT** — measured on the real CSVs.

| Check | Result | Verdict |
|---|---|---|
| Split sizes | train 59,958 / valid 9,904 / test 9,927 | ✓ |
| Duplicate IDs | 0 / 0 / 0 | **PASS** |
| Null captions | 0 / 0 / 0 | **PASS** |
| Duplicate captions within split | 318 / 11 / 13 | acceptable (ROCOv2 artifact) |
| Vocabulary built from training only | yes — `build_vocab_from_csv(TRAIN_CSV)` | **PASS** |
| Image↔caption pairing | same DataFrame row, cannot desync | **PASS** |
| Missing images (15,000 train + 2,000 valid used) | **0** | **PASS** |
| train∩valid caption leakage (subsets used) | 8 rows / 2,000 = **0.40%** | WARN-4 |
| Caption truncation at 38 tokens | **10.4% train, 11.8% valid** | WARN-3 |

**Is `df.iloc[:N]` (never shuffled) a biased subsample? — NO. [PASS]**

**FACT.** Modality distribution comparison:

| Subset | CT | Ultrasound | MRI | X-ray |
|---|---|---|---|---|
| first 15,000 (used) | 28.51% | 3.88% | 10.35% | 15.51% |
| full 59,958 | 28.57% | 4.09% | 10.32% | 15.38% |
| random 15,000 (seed 0) | 27.79% | 4.25% | 10.75% | 15.81% |

The head of the CSV is distributionally representative to within ~0.5 pp. **This is a genuine PASS — a plausible bug I specifically hunted for and did not find.** Shuffling would be tidier but is not causing your problem.

**Dataset language prior (relevant to §14).** CT is mentioned in 28.5% of training captions — 7.3× more often than ultrasound (3.9%). Of the 1,496 captions mentioning "abdomen", **64.8% also mention CT** versus 4.2% ultrasound. "ct scan of" is the 2nd most common 3-word caption opening in the entire training set (193 occurrences).

---

# 7. Loss Audit (Phases 10 & 11)

**FACT.** `train.py:101-104`:
```python
logits  = model(images, captions)   # (B, 39, 7886)
targets = captions[:, 1:]            # (B, 39)
loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
```
`criterion = CrossEntropyLoss(ignore_index=vocab.pad_id, label_smoothing=0.1)`.

Reshaping: (B·39, 7886) against (B·39,) — correct one-label-per-position pairing.

**Concrete alignment, using your example caption:**

| decoder input (`captions[:, :-1]`) | `<sos>` | a | chest | radiograph | shows | pneumonia |
|---|---|---|---|---|---|---|
| **target** (`captions[:, 1:]`), same column | a | chest | radiograph | shows | pneumonia | `<eos>` |

The prediction produced at input position *t* is scored against the token at position *t+1*. **No off-by-one error.** `<sos>` is consumed only as input and never appears as a target; `<eos>` appears only as a target and is never fed as input except as a generated token during inference. Padding positions are excluded from both numerator and denominator of the loss average by `ignore_index`. **[PASS]**

**What does val_loss ≈ 4.99 actually mean? [FACT — computed]**

I computed reference baselines on the real validation targets (42,256 non-pad positions):

| Predictor | Loss |
|---|---|
| Uniform over 7,886 vocab (plain CE) | 8.9728 |
| Context-free unigram (plain CE) | 6.1907 |
| Context-free unigram, **minimum achievable label-smoothed CE** (ε=0.1) | **6.6630** |
| **Actual checkpoint val_loss (15k run)** | **4.9936** |
| Actual checkpoint val_loss (2k run) | 4.9970 |

**INFERENCE.** The model beats a context-free unigram predictor by **1.67 nats**, so it *has* learned real sequential language structure — this is not a dead model. What it has **not** learned is image conditioning (§10, §14).

---

# 8. Gradient Flow (Phase 15)

**NOT TESTED — no autograd available in this environment.** Substitute measurement: **weight drift from initialization**, which is the integral of all gradients ever applied.

**FACT.** PyTorch `nn.Linear` default init has std = 1/√(3·fan_in); `nn.LayerNorm` starts at weight exactly 1.0, bias exactly 0.0.

| Tensor | expected-at-init std | actual std | ratio |
|---|---|---|---|
| `encoder.layers.0.blocks.0.attn.qkv.weight` | 0.05893 | 0.05978 | **1.015** |
| `encoder.layers.0.blocks.0.mlp.fc1.weight` | 0.05893 | 0.05917 | **1.004** |
| `encoder.layers.2.blocks.0.attn.qkv.weight` | 0.02946 | 0.03010 | **1.022** |
| `encoder.layers.3.blocks.0.attn.qkv.weight` | 0.02083 | 0.02137 | **1.026** |
| `decoder.layers.0.self_attn.q_proj.weight` | 0.02083 | 0.02314 | **1.111** |
| `decoder.fc_out.weight` | 0.02083 | 0.02344 | **1.125** |

| LayerNorm | weight mean | weight std |
|---|---|---|
| `encoder.patch_embed.norm` | 1.00247 | 0.00911 |
| `encoder.layers.0.blocks.0.norm1` | 1.00212 | 0.00959 |
| `encoder.layers.3.blocks.1.norm2` | 1.00023 | 0.00397 |
| **`encoder.norm`** (last encoder op) | **0.95628** | **0.02278** |
| `decoder.layers.5.norm3` | **1.07641** | 0.01362 |

**INFERENCE — this is the classic gradient-starvation signature.** Encoder weights have moved 1.5–2.6% from initialization; decoder weights 9–13%. Every encoder LayerNorm is still at ≈1.000 **except `encoder.norm`** — the one layer directly adjacent to the decoder — which moved to 0.956. Gradient magnitude decays sharply with distance from the loss. Meanwhile `decoder.embed.weight` std is 0.986 (from an `nn.Embedding` N(0,1) init) and `decoder.fc_out.bias` has developed clear structure (top-biased tokens: `right`, `contrast`, `the`, `cystic`, `aortic` — a learned unigram prior).

**The decoder trained. The encoder essentially did not.** After 3 epochs this is expected; the problem is that training *stopped* at 3 epochs (WARN-1).

---

# 9. Tiny Dataset Overfit Test (Phase 16)

**NOT TESTED — requires backpropagation, unavailable here.** This remains, as you correctly identified, the mandatory gate. Runnable script:

```python
"""tiny_overfit.py -- MANDATORY sanity check. Run BEFORE any further full training.
Trains on 10 image-caption pairs. Uses your unmodified modules."""
import torch, pandas as pd, torchvision.transforms as T
from torch.utils.data import DataLoader
from dataset import ROCODataset
from vocab import Vocab
from caption_model import SwinCaptioningModel
from train import build_param_groups

ROOT="/Users/nagashiva/Downloads/rocov2"; N=10; MAX_LEN=40
DEVICE=torch.device("mps" if torch.backends.mps.is_available() else "cpu")

df=pd.read_csv(f"{ROOT}/train_captions.csv").iloc[:N]
vocab=Vocab(df["Caption"].astype(str).tolist(), min_freq=1)   # min_freq=1: no <unk> on 10 samples
ds=ROCODataset(f"{ROOT}/train_captions.csv", f"{ROOT}/train_images/train", vocab, MAX_LEN, max_samples=N)
dl=DataLoader(ds,batch_size=N,shuffle=False)

model=SwinCaptioningModel(vocab_size=len(vocab),max_len=MAX_LEN).to(DEVICE)
opt=torch.optim.AdamW(build_param_groups(model,0.0),lr=1e-4)     # NO weight decay, LOW lr
crit=torch.nn.CrossEntropyLoss(ignore_index=vocab.pad_id)        # NO label smoothing
imgs,caps=next(iter(dl)); imgs,caps=imgs.to(DEVICE),caps.to(DEVICE)

for step in range(400):
    model.train()
    logits=model(imgs,caps); loss=crit(logits.reshape(-1,logits.size(-1)),caps[:,1:].reshape(-1))
    opt.zero_grad(); loss.backward()
    gn=torch.nn.utils.clip_grad_norm_(model.parameters(),1e9)     # measure, don't clip
    opt.step()
    if step%25==0:
        enc=sum(p.grad.norm()**2 for n,p in model.named_parameters() if n.startswith("encoder") and p.grad is not None)**0.5
        dec=sum(p.grad.norm()**2 for n,p in model.named_parameters() if n.startswith("decoder") and p.grad is not None)**0.5
        print(f"step {step:4d}  loss {loss.item():.4f}  |grad| total {gn:.3f}  enc {enc:.4f}  dec {dec:.4f}  ratio {enc/dec:.4f}")

model.eval()
with torch.no_grad():
    for i in range(N):
        ids=model.generate(imgs[i:i+1],vocab,max_len=MAX_LEN,device=DEVICE)
        print(f"\nGT : {df.iloc[i]['Caption'][:90]}\nGEN: {vocab.decode(ids[0].tolist())}")
```

**PASS criterion:** loss → below ~0.5 and the 10 generated captions closely reproduce the 10 training captions.
**If it FAILS:** the defect is in gradient flow / masking / target alignment, and no amount of data or epochs will help.
**If it PASSES** (which, given every static check above passed, I expect): the architecture is sound and the problem is confined to the training recipe — BUG-1, BUG-2, WARN-1.

The `enc/dec` gradient-norm ratio printed each 25 steps directly measures the gradient starvation inferred in §8. Watch it.

---

# 10. Representation Health (Phase 17) — **THE CENTRAL FINDING**

**FACT — measured on real checkpoint weights and real images.**

## Within-image: the 49 tokens are near-identical

| Image | mean pairwise cosine (off-diag) | std | min | pairs with cos > 0.99 |
|---|---|---|---|---|
| `test_007757` (chest CT) | **0.9977** | 0.0050 | 0.9695 | **96.2%** |
| `test_000004` (angiogram) | **0.9995** | 0.0005 | 0.9960 | **100.0%** |

Feature norms are almost perfectly constant: mean 23.774, **std 0.026** (007757); mean 23.807, **std 0.010** (000004).

## Variance decomposition — the decisive number

| Quantity | Value | % of total |
|---|---|---|
| Total variance of the 49×768 tensor | 0.735965 | 100% |
| Variance **across the 49 tokens** (avg over 768 dims) | 0.001657 | **0.225%** |
| Variance **across the 768 dims** (avg over 49 tokens) | 0.735965 | 100.000% |

> **Only 0.225% of the signal in the encoder's output distinguishes one spatial location from another. The other 99.775% is a constant pattern present at every one of the 49 positions.**

## Control experiment — is the encoder responding to image *content* at all?

**FACT.** I ran the real checkpoint on real images and on synthetic controls:

| Input | within-image token cos | Generated caption |
|---|---|---|
| REAL `test_007757` (chest CT) | 0.9977 | `ct scan of the abdomen showing a mass in the right lobe of the liver` |
| REAL `test_000004` (angiogram) | 0.9995 | `postoperative panoramic radiograph` |
| **CONTROL all-black** | 1.0000 | `mri of the right foot showing a mass in the left side of the left femur` |
| **CONTROL all-white** | 1.0000 | `ct scan of the chest showing bilateral pleural effusions` |
| **CONTROL uniform noise** | 0.9999 | `mri of the brain showing a mass in the left frontal lobe` |
| **CONTROL 50% grey** | 1.0000 | `postoperative panoramic radiograph` ← **identical to the real angiogram** |

Pairwise cosine similarity of mean-pooled 49×768 representations:

|  | 007757 | 000004 | black | white | noise | grey |
|---|---|---|---|---|---|---|
| **007757** | 1.0000 | 0.9793 | 0.9936 | 0.9878 | **0.9987** | 0.9826 |
| **000004** | 0.9793 | 1.0000 | 0.9864 | 0.9883 | 0.9796 | **0.9993** |

> **The real chest CT is MORE similar to pure random noise (0.9987) than to another real medical image (0.9793). A 50%-grey rectangle is more similar to the real angiogram (0.9993) than the two real images are to each other.**

**INFERENCE.** This is quantitative, reproducible proof of **encoder representation collapse**, satisfying the bar you correctly set ("do not label it representation collapse without quantitative evidence"). The encoder produces a nearly image-independent, nearly position-independent constant vector. The model is, functionally, an unconditional language model with a very weak image-derived perturbation.

**Early vs. late training comparison: NOT TESTED** — only the epoch-3 checkpoint survives; no epoch-1 or epoch-60 checkpoint from the 15k run is available.

---

# 11. Training Behavior (Phase 18)

**NOT TESTED for per-epoch curves** — `train.py` prints but does not persist logs; no history file exists. **RECOMMENDATION:** append a CSV log line per epoch (`epoch, train_loss, val_loss, lr, grad_norm, epoch_time`), which makes all four requested plots trivial and costs 3 lines.

**FACT — what the two surviving checkpoints tell us:**

| Run | vocab | train samples | best epoch | best val_loss |
|---|---|---|---|---|
| 2k pilot | 2,682 | 2,000 | **3** | 4.9970 |
| 15k | 7,886 | 15,000 | **3** | 4.9936 |

Both vocabularies were independently rebuilt from the CSVs and matched the checkpoints **exactly** (2,682 and 7,886), confirming provenance beyond doubt.

**INFERENCE — this is the most diagnostic pair of numbers in the audit.** A 7.5× increase in training data moved validation loss by **0.0034 nats (0.07%)** and did not change the best epoch at all. If the bottleneck were data quantity, more data would help. It did not. **The bottleneck is the optimization recipe, not the dataset size.** This directly refutes the "just needs more data" hypothesis and redirects attention to BUG-1, BUG-2, and WARN-1.

---

# 12. Validation Metrics — REAL, MEASURED

**FACT.** 120 validation images (`valid_captions.csv` rows 1–120), 15k checkpoint, greedy decoding with trigram blocking, evaluated with your own `metrics.py`.

| Metric | **This model** | ROCOv2 CNN-LSTM baseline | Ratio |
|---|---|---|---|
| BLEU-1 | **0.0774** | 0.586 | 0.13× |
| BLEU-2 | **0.0431** | 0.498 | 0.09× |
| BLEU-3 | **0.0236** | 0.347 | 0.07× |
| BLEU-4 | **0.0141** | 0.127 | 0.11× |
| METEOR | **NOT IMPLEMENTED** | — | — |
| ROUGE-L | **0.1483** | 0.453 | 0.33× |
| CIDEr-D | **0.1421** (×10 scale) | 0.463 | 0.31× |
| **Validation CE** (no smoothing, PAD ignored) | **4.2598** | — | — |
| **Perplexity** = exp(CE) | **70.8** | — | — |
| **Token accuracy** (teacher-forced, PAD ignored) | **27.35%** (713/2,607) | — | — |
| **Distinct captions** | **14 / 120 (11.7%)** | — | — |
| Mean generated length | 10.0 tokens | reference 23.4 | — |

## Collapse evidence at the output level

| Count | % | Caption |
|---|---|---|
| 36 | 30.0% | `mri of the brain showing a mass in the left frontal lobe` |
| 33 | 27.5% | `postoperative radiograph` |
| 24 | 20.0% | `ct scan of the abdomen showing a mass in the right lobe of the liver` |
| 7 | 5.8% | `ct scan of the chest showing a mass in the right upper lobe of the lung` |
| 4 | 3.3% | `ct scan of the chest showing a mass in the right lung` |

**Three templates cover 77.5% of all 120 outputs.** First-token distribution: `mri` 33.3%, `ct` 32.5%, `postoperative` 29.2% — 95% of images receive one of three opening words.

## Metric sanity check (Phase 25) — **PASS**

**FACT.** Ran `metrics.py` on controlled toy cases:

| Case | BLEU-1 | BLEU-4 | ROUGE-L | CIDEr |
|---|---|---|---|---|
| prediction == reference | **1.0000** | **1.0000** | **1.0000** | **10.0000** |
| partial overlap | 0.6065 | 0.4824 | 0.7722 | 6.0795 |
| completely unrelated | 0.0899 | **0.0000** | 0.1359 | 0.6673 |
| empty prediction | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| repeated-word spam | 0.1103 | 0.0000 | 0.1164 | 0.8218 |

Corpus-level, 4 sentences: all-exact → BLEU-4 1.0000 / CIDEr 10.0; shuffled-to-wrong-reference → BLEU-4 0.0000 / CIDEr 0.0380. Brevity penalty verified: truncating a 13-token reference prediction to 7 tokens drops BLEU-1 from 1.0000 to 0.4244.

**All metrics are monotonic and correctly ordered. [PASS]** Caveats: CIDEr max is 10.0 (WARN-7); IDF is computed on the evaluation set, so absolute CIDEr values shift with `--n`.

> **Note (Phase 25 finding worth remembering):** a corpus where every prediction is the *same* generic caption still scored **BLEU-4 0.2205 / CIDEr 1.6994** in my toy test. Collapsed models can post non-trivial n-gram scores. **This is precisely why the "% distinct captions" check in `eval.py` is the more honest headline number, and why you should keep reporting it.**

---

# 13. Qualitative Results (Phase 21)

**FACT.** First 10 validation images.

| # | Reference (truncated) | Generated | Classification | Error type |
|---|---|---|---|---|
| 1 | Chest X-ray, enlarged cardiac silhouette, CT ratio 70% | `chest x ray showing bilateral pleural effusions` | **partially correct** | modality ✓, finding ✗ |
| 2 | Digital subtraction ureterogram, fistula | `ct scan of the chest showing bilateral pleural effusions` | **incorrect** | modality ✗, anatomy ✗, finding ✗ |
| 3 | DBT/US breast cancer comparison | `postoperative radiograph` | **incorrect** | modality ✗, anatomy ✗, incomplete |
| 4 | CT scan image for lung cancer | `ct scan of the abdomen showing a mass in the right lobe of the liver` | **partially correct** | modality ✓, anatomy ✗ (chest→abdomen), hallucinated finding |
| 5 | A giant retroperitoneal tumor | `mri of the brain showing a mass in the left frontal lobe` | **incorrect** | anatomy ✗, hallucination |
| 6 | Biphasic CECT abdomen, dilated CBD | `mri of the brain showing a mass in the left frontal lobe` | **incorrect** | modality ✗, anatomy ✗ |
| 7 | 18F-FDG PET-CT, periampullary thickening | `ct scan of the abdomen showing a mass in the right lobe of the liver` | **partially correct** | region ✓ (abdomen), modality ✗ (PET-CT→CT), finding ✗ |
| 8 | Early axial T2-weighted MRI | `mri of the brain showing a mass in the left frontal lobe` | **partially correct** | modality ✓, anatomy unverifiable, hallucinated finding |
| 9 | Late sagittal T2-weighted MRI | `mri of the brain showing a mass in the left frontal lobe` | **partially correct** | modality ✓, identical to #8 |
| 10 | Chest X-ray, bilateral pleural effusions, consolidation | `postoperative radiograph` | **incorrect** | modality ✗, incomplete |

**Tally:** correct 0/10, mostly correct 0/10, partially correct 5/10, incorrect 5/10.
**Error taxonomy:** modality errors 5, anatomy errors 6, finding errors 9, hallucinated findings 6 (masses/effusions absent from the reference), incomplete captions 3, intra-caption repetition **0** (the trigram guard works). Note #8 and #9 are different images (early axial vs. late sagittal MRI) receiving byte-identical captions.

---

# 14. Failure Analysis (Phase 22)

## ⚠ First: your stated ground truth for `007757` is wrong

**FACT.** I searched `test_captions.csv` directly. The caption you quoted —
> *"Operative planning ultrasound prior to ultrasound-guided subcutaneous collection drainage and appendicolith retrieval…"*

— belongs to **`ROCOv2_2023_test_000036`**, not `007757`. The actual row is:

```
ROCOv2_2023_test_007757 → "Chest computed tomography before endobronchial ultrasound-guided
                            transbronchial needle aspiration demonstrated enlargement of the
                            right paratracheal lymph node."
```

**This materially changes the diagnosis:**

| Aspect | Model output | **Actual** ground truth | Verdict |
|---|---|---|---|
| Modality | `ct scan` | "Chest computed **tomography**" | **CORRECT** |
| Region | `the abdomen` | "**Chest** … right paratracheal" | wrong |
| Finding | `a mass in the right lobe of the liver` | "enlargement of the right paratracheal **lymph node**" | wrong |

The failure is **not** "hallucinated CT when the truth was ultrasound." It is **"correct modality, wrong body region, wrong finding."** Any conclusion drawn from the incorrect ground truth would have been wrong.

## Where the failure actually occurs — measured, step by step

**FACT.** Full generation trace for `007757`:

| Step | Context | Top predictions (probability) | Chosen | Cross-attn entropy |
|---|---|---|---|---|
| 0 | `<sos>` | **ct 0.261**, abdominal 0.084, computed 0.076, mri 0.029 | `ct` | 1.371 (peaked) |
| 1 | `ct` | **scan 0.452**, of 0.090, image 0.053, showing 0.038 | `scan` | 3.859 (uniform) |
| 2 | `ct scan` | **of 0.230**, showing 0.203, demonstrating 0.041 | `of` | 3.847 |
| 3 | `ct scan of` | **the 0.679**, abdomen 0.131, chest 0.018 | `the` | 3.853 |
| **4** | `ct scan of the` | **abdomen 0.301**, **chest 0.275**, thorax 0.075, neck 0.045 | `abdomen` | 3.834 |
| 5 | `…abdomen` | **showing 0.263**, and 0.099, with 0.081 | `showing` | 3.864 |
| 7 | `…showing a` | **mass 0.091**, large 0.073, huge 0.058, liver 0.031 | `mass` | 3.886 |

> **The entire error is decided at step 4 by a margin of 0.026 probability: `abdomen` 0.301 vs `chest` 0.275.** The correct word was the model's *second* choice, and greedy decoding — which has no mechanism to reconsider — committed the remaining 11 tokens to a self-consistent abdominal narrative. `liver` at step 11 is a *consequence* of `abdomen` at step 4, not an independent error.

Note the model got the modality right (`ct`, p=0.261 at step 0, and step 0 is the one step where cross-attention is genuinely peaked, entropy 1.371 = 35% of uniform). **The visual signal is strong enough to influence the first token and essentially nothing after it.**

## Ranked causes — each with evidence

| Rank | Cause | Verdict | Evidence |
|---|---|---|---|
| **1** | **Encoder representation collapse** | **CONFIRMED** | 0.225% of variance is spatial; cos(real image, noise) = 0.9987 > cos(real, real) = 0.9793; black/grey/noise inputs all produce fluent captions (§10) |
| **2** | **Undertrained encoder / training stopped at epoch 3** | **CONFIRMED** | Encoder weights at 1.00–1.06× init; all encoder LayerNorms still ≈1.000; both runs best at epoch 3 (§8, §11) |
| **3** | **Weak cross-modal alignment** | **CONFIRMED** | Cross-attention entropy 99.2% of uniform after step 0 (§5) |
| **4** | **Decoder language prior dominates** | **CONFIRMED** | Model beats unigram baseline by 1.67 nats while being image-blind; `fc_out.bias` encodes a learned unigram prior; 3 templates = 77.5% of outputs (§7, §12) |
| **5** | **Dataset language bias** | **CONFIRMED as amplifier** | "abdomen"→CT co-occurrence 64.8%; "ct scan of" is 2nd most common caption opening (§6) |
| **6** | **Decoding strategy (greedy)** | **CONFIRMED as amplifier, not root cause** | Step-4 margin 0.301 vs 0.275; beam search would likely have kept both hypotheses alive (§14 trace) |
| **7** | **Insufficient training data** | **REFUTED** | 2,000 → 15,000 images changed val_loss by 0.0034 nats and did not move the best epoch (§11) |
| **8** | **Tokenization / vocabulary** | **LARGELY REFUTED** | All key ground-truth words (`chest`, `lymph`, `node`, `enlargement`, `ct`, `ultrasound`, `transbronchial`) are in vocabulary; only `endobronchial`, `paratracheal` map to `<unk>` |
| **9** | **Preprocessing / train-inference mismatch** | **REFUTED** | Byte-identical transforms across all four scripts; vocab restored from checkpoint; `eval()` called (§3 P15–P19) |
| **10** | **Implementation bug in the forward pass** | **REFUTED** | Every tensor operation verified correct (§3, §4, §5); NumPy re-implementation reproduced both reported captions exactly |

**Root cause, stated plainly:** the architecture is correct, but the encoder never learned to produce distinguishable features, because (a) it was never properly initialized (BUG-1), (b) it had no depth regularization (BUG-2), and (c) training was terminated at epoch 3 by a learning-rate schedule that peaks *after* the best epoch and an early-stopping patience that expires before cosine decay can help (WARN-1). The decoder, being closer to the loss, trained normally and learned to produce fluent, dataset-typical radiology sentences — which is exactly what a medically-plausible-but-visually-wrong caption is.

---

# 15. Comparison of Two Images (Phase 24)

**FACT.**

| Quantity | `test_007757` | `test_000004` |
|---|---|---|
| Generated | `ct scan of the abdomen showing a mass in the right lobe of the liver` | `postoperative panoramic radiograph` |
| Ground truth | Chest CT, right paratracheal lymph node | Digitally subtracted angiogram of the IMA |
| Within-image token cos | 0.9977 | 0.9995 |
| Feature norm (mean ± std) | 23.774 ± 0.026 | 23.807 ± 0.010 |
| Variance across 49 tokens | 0.00166 | 0.00037 |
| Cross-attn entropy, step 0 | 1.371 (35.2% of uniform) | 3.512 (90.2%) |
| Cross-attn entropy, steps 1+ | 3.860 (99.2%) | 3.785 (97.2%) |
| Caption length | 16 steps | 4 steps |
| First-token probability | ct = 0.261 | postoperative = 0.046 |

**Cross-image:** mean |A−B| per element 0.1390; relative L2 difference 0.2103; cosine similarity between matching tokens 0.9779; between mean-pooled vectors **0.9793**.

**INFERENCE — nuanced answer to "does the model respond to visual differences?"** **Partially, and only at the first token.** The two images do produce different first tokens (`ct` vs `postoperative`) and measurably different first-token distributions, so ~2% of representational difference is enough to flip a coin at step 0. But that same difference (0.9793) is *smaller* than the difference between a real image and random noise (0.9987 similarity), so the discrimination is not reliably content-driven. After step 0 the visual pathway goes uniform and the decoder's language model takes over completely.

---

# 16. Known Limitations of This Audit

1. **No backpropagation.** Phases 15 (gradient statistics) and 16 (tiny overfit) could not be executed. Weight-drift analysis (§8) is a substitute, not a replacement.
2. **NumPy substitute for PyTorch.** Validated by exact reproduction of two captions, but GELU's `erf` is approximated to 1.5e-7 and PIL's bilinear resize may differ from `torchvision`'s in the last decimal. Neither can change conclusions resting on 0.9977-vs-0.98 magnitudes.
3. **120 of 9,904 validation images** evaluated (~1.2%), limited by NumPy inference speed (58 s). Metrics have sampling error; the collapse statistics (11.7% distinct) are unambiguous at this sample size.
4. **METEOR not implemented** and therefore not reported.
5. **Only one epoch's checkpoint exists** per run, so early-vs-late representation comparison was impossible.
6. **`test_007757` and `test_000004` are test-set images.** They were analyzed only because you had already generated captions for them; all *metrics* in §12 use validation data, per Rules 11–12.

---

# 17. Recommended Fixes

## MUST FIX — before any further training run

| # | Fix | Addresses | From-scratch safe? |
|---|---|---|---|
| **M1** | **Run the tiny 10-example overfit test** (§9 script) and report the `enc/dec` gradient ratio. Do not start any full run until this passes. | Gates everything | ✓ |
| **M2** | **Add proper weight initialization**: `trunc_normal_(std=0.02)` on every Linear, `zeros_` on biases, `ones_`/`zeros_` on LayerNorm, applied via `self.apply(_init_weights)`. | BUG-1 | ✓ ~8 lines of `nn.init` |
| **M3** | **Lower `PEAK_LR`** from 5e-4 to ~1e-4–2e-4, **or** raise `EARLY_STOP_PATIENCE` well above 6 so cosine decay is reachable. Evidence: best epoch = 3 = LR 3e-4 in both runs. | WARN-1 | ✓ constant change |
| **M4** | **Implement `DropPath`** with a linearly increasing rate across the 12 encoder blocks. | BUG-2 | ✓ ~10 lines |
| **M5** | **Log every epoch to CSV** (`epoch, train_loss, val_loss, lr, grad_norm, token_acc, time`). Without this, Phase 18 is permanently unanswerable. | §11 | ✓ 3 lines |
| **M6** | **Replace `float("-inf")` with `-100.0`** in `caption_model.generate()` (both guards). | BUG-3 | ✓ |
| **M7** | **Correct `ff_dim` to 3072** everywhere in your paper/spec. | §2 | ✓ documentation |

## OPTIONAL IMPROVEMENTS

| # | Improvement | Addresses |
|---|---|---|
| O1 | Raise `MAX_LEN` from 40 to ~64, or report the 10.4% truncation rate as a known limitation | WARN-3 |
| O2 | Add an auxiliary loss on the encoder output (e.g. a concept-classification head using ROCOv2's unused `*_concepts.csv`) to give the encoder direct gradient rather than only what leaks through cross-attention | §8, §10 |
| O3 | Beam search (width 3–5) instead of greedy — the step-4 margin was 0.026 | §14 |
| O4 | Shuffle before `df.iloc[:N]` — measured as *not* currently harmful (§6), but tidier |
| O5 | Save optimizer + scheduler state to allow resumption | WARN-6 |
| O6 | Deduplicate the 8 leaked validation captions | WARN-4 |
| O7 | Add a permanent `return_attn=True` debug flag to `MultiHeadAttention` | reproducibility of §5 |
| O8 | Report "% distinct captions" alongside BLEU in every experiment table | §12 |

## WOULD VIOLATE THE FROM-SCRATCH GOAL — not recommended

Pretrained Swin/ViT weights, CLIP/BLIP, HuggingFace `transformers`, torchvision's Swin, pretrained tokenizers or BERT embeddings. **None of these are necessary** — every MUST-FIX above is implementable in plain `torch.nn`, and the audit found the architecture already correct.

---

# 18. Research Paper Recommendations

## SAFE to report

- **The architecture and its verification.** Every tensor shape (224×224×3 → 3136×96 → 784×192 → 196×384 → 49×768), the W-MSA/SW-MSA mechanism, relative position bias (169 offsets, no out-of-bounds), patch merging, causal masking, and cross-attention wiring are all verified correct by execution. **This is a genuine, defensible contribution and the strongest part of your work.**
- **Parameter count: 96,381,512.**
- **The negative result itself**, which is publishable and interesting: *a correctly-implemented Swin-T encoder trained from random initialization on 15,000 radiology image–caption pairs undergoes measurable representation collapse* — supported by the 0.225% spatial-variance figure, the cos(real, noise) = 0.9987 > cos(real, real) = 0.9793 control, and the black/grey/noise caption experiment. Few papers run that control; it is rigorous evidence.
- **The 2k vs 15k comparison** as evidence that data scale was not the bottleneck (val_loss 4.9970 → 4.9936, best epoch 3 in both).
- **The measured validation metrics**, *provided* they are labelled as an undertrained epoch-3 checkpoint on 120 validation images, and reported alongside the 11.7%-distinct figure.
- **Your own metric implementation**, with the Phase 25 sanity table and the CIDEr ×10 / eval-set-IDF caveats disclosed.

## NOT SAFE to report

- ❌ Any framing of these metrics as this architecture's achievable performance — they measure a 3-epoch checkpoint whose encoder never trained.
- ❌ Direct comparison to the ROCOv2 CNN-LSTM baseline as a like-for-like result (that baseline uses a **pretrained** DenseNet169; yours is random-init). State this asymmetry explicitly.
- ❌ `ff_dim = 2048` — the code and checkpoint say **3072**.
- ❌ METEOR — not implemented.
- ❌ Any claim of the form "the model learned to identify modality" — first-token accuracy on 120 images is 3 words covering 95% of images; that is a prior, not perception.
- ❌ The `007757` ground truth quoted in your brief (it belongs to `test_000036`).

## Framing suggestion

Position the paper as **"a verified from-scratch implementation, and a controlled study of representation collapse when training Swin without pretraining at small data scale."** The audit gives you the evidence for exactly that paper. It does not give you evidence for a performance paper — and attempting one on these numbers would not survive review.

---

# 19. FINAL VERDICT

## Can I trust this model's current results? — **NO** (with an important partial YES)

**NO — for any performance or capability claim.** Three independent, measured lines of evidence:

1. **The encoder is functionally blind.** cos(real chest CT, pure random noise) = **0.9987** exceeds cos(real chest CT, real angiogram) = **0.9793**. An all-black image yields a confident, fluent, entirely fabricated clinical caption. A 50%-grey rectangle yields the *identical* caption to a real angiogram.
2. **The spatial representation has collapsed.** Only **0.225%** of the variance in the 49×768 output distinguishes spatial positions; mean pairwise token cosine is **0.9977**. Cross-attention entropy after step 0 is **99.2% of uniform** — the vision→language bridge carries almost no information.
3. **The encoder never trained.** Encoder weights sit at **1.00–1.06×** their random initialization and every encoder LayerNorm is still ≈1.000, while decoder weights moved 9–13%. Both training runs stopped at **epoch 3**.

Consequently BLEU-1 0.0774 / BLEU-4 0.0141 / ROUGE-L 0.1483 / CIDEr-D 0.1421, with 14 distinct captions across 120 images, describe a language model with a decorative image input — not a captioning system.

**YES — for the implementation itself.** This is a real and separable result. I attacked the code with every static and dynamic test available and found **no incorrect tensor operation**: window partition/reverse round-trips bit-exactly at all four resolutions; the relative-position index uses all 169 offsets with zero out-of-bounds; the shift mask covers exactly the 15/64 wrap-around windows; patch merging matches official Swin's slice order; the causal mask leaks nothing; cross-attention is correctly wired Q←caption, K/V←image; target shifting has no off-by-one; the checkpoint loads with 0 missing and 0 unexpected keys; there are no NaN or Inf anywhere. **Three genuine defects exist (missing weight init, missing stochastic depth, `-inf`/`-100.0` inconsistency) and all three are in the *training recipe*, not the architecture.**

**What this means practically:** you do not need to rewrite your model. You need to (1) run the tiny-overfit gate, (2) add proper initialization, (3) lower the learning rate or extend patience, (4) add DropPath — then retrain. Your faculty can be told, accurately and with evidence, that the architecture is verified correct and the current results reflect an incompletely-trained encoder rather than a design error.

---
---

# VIVA SECTION — 20 Questions

## The questions

1. Why is the patch size 4, and what would break if you used 3?
2. Why exactly 3,136 tokens after patch embedding?
3. Why is the initial embedding dimension 96, and where does 48 come from?
4. Why does 96 become 192 at the first patch merge — why not 384?
5. Why does 192 become 384, and 384 become 768?
6. Why are there exactly 49 final tokens?
7. Why is the window 7×7, and what happens to windowing at stage 4?
8. What is W-MSA, and what is its computational cost versus full attention?
9. What is SW-MSA and what problem does it solve that W-MSA cannot?
10. Why shift by exactly 3?
11. Why is an attention mask needed after shifting, and why −100.0 instead of −∞?
12. What is relative position bias, why 169 entries, and why relative rather than absolute?
13. Why does QKᵀ produce 49×49 inside a window, and what is one element of that matrix?
14. Why is Q from the caption and K/V from the image in cross-attention — what breaks if reversed?
15. What exactly does one element of the T×49 cross-attention matrix mean?
16. Why is causal masking required, and what specifically would the model learn without it?
17. Why do we shift caption inputs and targets by one position?
18. Why do we ignore PAD in cross-entropy — what goes wrong if we don't?
19. Why can this model produce a medically plausible but visually incorrect caption?
20. In your own audit, how would you *prove* representation collapse is occurring rather than merely asserting it?

---

## The answers

### Q1 — Why is the patch size 4, and what would break if you used 3?

**ANSWER.** Patch size 4 is chosen so 224 divides evenly (224/4 = 56) and so the resulting 56×56 grid survives three successive halvings to exactly 7×7. With patch size 3, 224/3 = 74.67 — a non-integer. Your `PatchEmbed` uses `self.grid_size = img_size // patch_size`, which would floor to 74 and **silently discard the last 2 pixel rows and columns**; worse, 74 is not divisible by the window size 7, so `window_partition`'s reshape would fail.

**MATHEMATICAL INTUITION.** Every Swin stage requires `H % window_size == 0` and `H % 2 == 0` for merging. The chain 56 → 28 → 14 → 7 needs $224 = 4 \cdot 7 \cdot 2^3$. Patch size 4 is the unique small value making this exact.

**CODE CONNECTION.** `swin_model.py:12` — `self.grid_size = img_size // patch_size`. Note the **floor division**: it never raises an error on a bad configuration, it silently crops. Official Swin pads instead; yours does not (a `[WARNING]` in §3's less-general column).

**TENSOR SHAPES.** patch 4 → (1,3,224,224) → (1,3136,96) ✓. patch 3 → grid 74 → 74 % 7 = 4 ≠ 0 → reshape error in `window_partition`.

### Q2 — Why exactly 3,136 tokens?

**ANSWER.** $224/4 = 56$ patches per side; $56 \times 56 = 3{,}136$. Each is one token.

**MATHEMATICAL INTUITION.** Patching is the compression that makes attention affordable: attention costs $O(N^2)$, so at the pixel level $N = 50{,}176$ gives a $2.5 \times 10^9$-entry matrix. At $N = 3{,}136$ it is $9.8 \times 10^6$ — a 256× reduction, obtained purely by choosing the unit of analysis.

**CODE CONNECTION.** `swin_model.py:18-22` — `self.proj(x)` gives (B,96,56,56); `.flatten(2).transpose(1,2)` gives (B,3136,96). The flatten is **row-major**, which is why token *t* ↔ (t//56, t%56).

**TENSOR SHAPES.** (1,3,224,224) → conv → (1,96,56,56) → flatten → (1,3136,96).

### Q3 — Why is the embedding dimension 96, and where does 48 come from?

**ANSWER.** 48 = 4×4×3 is the number of **raw pixel values** in one patch (4×4 spatial × 3 channels). 96 is the **learned** output width — a design choice from the Swin-T configuration, chosen so it divides evenly by the head counts (96/3 = 32) and so three doublings reach 768.

**MATHEMATICAL INTUITION.** $\mathbf{z}_i = W\,\text{flatten}(\text{patch}_i) + b$ with $W \in \mathbb{R}^{96\times48}$ — a projection from 48 raw dimensions into a 96-dimensional *learned* space. It is an expansion (48→96), giving the network more capacity than the raw pixels carry.

**CODE CONNECTION.** `swin_model.py:15` — `nn.Conv2d(3, 96, kernel_size=4, stride=4)`. Because stride = kernel size, patches never overlap, so this convolution is **exactly** the shared linear projection above. Checkpoint confirms `(96, 3, 4, 4)`, which is 96 filters × 48 weights each.

**TENSOR SHAPES.** per patch: (48,) → (96,). Whole image: (1,3136,48) → (1,3136,96).

### Q4 — Why does 96 become 192 — why not 384?

**ANSWER.** Patch merging concatenates a 2×2 neighbourhood (4 tokens × 96 = 384 channels) and then **projects back down to 192**, i.e. 2C not 4C. Keeping 4C would preserve information exactly but leave total capacity unchanged; halving it to 2C is a deliberate compression.

**MATHEMATICAL INTUITION.** Token count drops 4× while width grows 2×, so total capacity halves at every stage:
$$\frac{H}{2}\cdot\frac{W}{2}\cdot 2C = \frac{HWC}{2}$$
This is the same trade a CNN makes with strided pooling — spend resolution to buy semantic depth.

**CODE CONNECTION.** `swin_model.py:213` — `nn.Linear(4 * dim, 2 * dim, bias=False)`. Note `bias=False`, matching official Swin. Checkpoint: `layers.0.downsample.reduction.weight` = **(192, 384)**.

**TENSOR SHAPES.** (1,3136,96) → view (1,56,56,96) → concat → (1,28,28,384) → LN → Linear → **(1,784,192)**.

### Q5 — Why does 192 become 384, and 384 become 768?

**ANSWER.** The identical rule applied twice more: concat 2×2 → 4C → project to 2C.

**MATHEMATICAL INTUITION.** Geometric progression $96 \cdot 2^k$ for $k = 0,1,2,3$, mirrored by resolution $56/2^k$. Channel width and spatial extent trade off at a fixed 2:4 rate.

**CODE CONNECTION.** `swin_model.py:264-269` — the loop `dim *= 2; res //= 2` after each `BasicLayer` whose `downsample` is not `None`. Crucially `downsample = PatchMerging if i < len(depths) - 1 else None`, so **stage 4 has no merge** — verified in the checkpoint by the absence of any `encoder.layers.3.downsample.*` key.

**TENSOR SHAPES.** Verified by execution: (1,784,192) → (1,196,384) → (1,49,768). Checkpoint reductions: (384,768) and (768,1536).

### Q6 — Why exactly 49 final tokens?

**ANSWER.** $56 \to 28 \to 14 \to 7$ after three merges, so the final grid is 7×7 = **49**. Each token traces back to a 224/7 = **32×32-pixel** region.

**MATHEMATICAL INTUITION.** $56/2^3 = 7$. Three merges is what the depths tuple `(2,2,6,2)` implies — four stages, three transitions between them.

**CODE CONNECTION.** `swin_model.py:272-273` — `self.out_dim = dim` (768), `self.out_res = res` (7). Executed forward pass confirms `(1, 49, 768)`. Spatial mapping: token *t* ↔ (t//7, t%7) ↔ pixels [32·row, 32·row+32) × [32·col, 32·col+32).

**TENSOR SHAPES.** (1,49,768) — this is the `memory` tensor consumed by every one of the 6 decoder layers.

### Q7 — Why 7×7 windows, and what happens at stage 4?

**ANSWER.** 7 divides 56, 28, and 14 evenly, so windowing works at stages 1–3 without padding. At stage 4 the grid is 7×7 — **equal to the window** — so there is exactly one window and "windowed" attention silently becomes **full global self-attention** over all 49 tokens.

**MATHEMATICAL INTUITION.** Windows per stage: $(56/7)^2 = 64$, $(28/7)^2 = 16$, $(14/7)^2 = 4$, $(7/7)^2 = 1$. Locality is progressively relaxed, and the last stage sees everything.

**CODE CONNECTION.** `swin_model.py:138-141`:
```python
if min(input_resolution) <= window_size:
    self.shift_size = 0
    self.window_size = min(input_resolution)
```
At stage 4, `min(7,7) <= 7` is True → shift disabled, window = 7. **Both stage-4 blocks are therefore W-MSA; neither is SW-MSA** — a subtlety worth knowing for your viva, since the naive reading of "alternating blocks" would suggest otherwise.

**TENSOR SHAPES.** Verified round-trip: 56→(64,7,7,96), 28→(16,7,7,192), 14→(4,7,7,384), 7→(1,7,7,768).

### Q8 — What is W-MSA and what does it cost?

**ANSWER.** Window Multi-head Self-Attention: standard scaled dot-product attention computed **independently inside each non-overlapping 7×7 window**, never across the whole image.

**MATHEMATICAL INTUITION.**
$$\text{full: } O(N^2) \qquad \longrightarrow \qquad \text{windowed: } O(N \cdot M^2), \quad M=7$$
Since $M$ is a constant, cost becomes **linear in $N$**. At stage 1 that is $3136 \times 49 = 153{,}664$ score entries per head instead of $3136^2 = 9{,}834{,}496$ — a **64× reduction**.

**CODE CONNECTION.** `swin_model.py:52-106` (`WindowAttention`), applied to the output of `window_partition` (line 32). Scaling uses `self.scale = head_dim ** -0.5` applied to `q` (line 91) rather than dividing the product — algebraically identical.

**TENSOR SHAPES.** Per window per head at stage 1: Q,K,V each (49,32); QKᵀ = **(49,49)**; output (49,32); 3 heads concatenated → (49,96).

### Q9 — What is SW-MSA and why is it needed?

**ANSWER.** Shifted-Window MSA. With fixed windows, a token at a window edge can **never** exchange information with its neighbour one position away in the adjacent window — no matter how many blocks you stack, information cannot cross window boundaries. SW-MSA cyclically shifts the whole grid before partitioning, so the window boundaries land in different places and previously-separated neighbours now share a window.

**MATHEMATICAL INTUITION.** One W-MSA block + one SW-MSA block gives every token a receptive field covering all its immediate neighbours; stacking pairs grows it further. Without the shift the receptive field would be permanently capped at one 7×7 window.

**CODE CONNECTION.** `swin_model.py:237-241` — `shift_size = 0 if (i % 2 == 0) else window_size // 2`. Even blocks W-MSA, odd blocks SW-MSA. The roll and un-roll: lines 180-183 and 193-196.

**TENSOR SHAPES.** Unchanged by shifting — (1,3136,96) in, (1,3136,96) out. The shift changes *which tokens co-occur in a window*, not any dimension.

### Q10 — Why shift by exactly 3?

**ANSWER.** `shift_size = window_size // 2 = 7 // 2 = 3`. Half a window is the displacement that maximally re-partitions: every new window straddles the corners of four old windows.

**MATHEMATICAL INTUITION.** Shifting by 0 changes nothing; shifting by 7 returns the identical partition (a full period). The midpoint, 3, puts every old boundary at a new window's interior — maximum boundary disruption.

**CODE CONNECTION.** `swin_model.py:239`, passed into `SwinBlock`, used at line 181: `torch.roll(x, shifts=(-3,-3), dims=(1,2))` and reversed at line 194 with `shifts=(+3,+3)`.

**TENSOR SHAPES.** Roll preserves shape exactly: (1,56,56,96) → (1,56,56,96).

### Q11 — Why is a mask needed after shifting, and why −100.0?

**ANSWER.** `torch.roll` is *cyclic* — it wraps content from the bottom edge to the top and from the right edge to the left. So after the shift, some 7×7 windows contain patches from **opposite sides of the image** that were never spatially adjacent. Letting those attend would be meaningless. The mask assigns a large negative bias to exactly those pairs. **−100.0 rather than −∞** because softmax over a row containing −∞ can produce NaN on PyTorch's MPS (Apple Silicon) backend, and `train.py` targets MPS.

**MATHEMATICAL INTUITION.** $e^{-100} \approx 3.7\times10^{-44}$ — numerically indistinguishable from zero after softmax, but a finite number that cannot produce NaN. **However** (audit finding WARN-2): in the *decoder*, `masked_fill` *sets* the value to −100 rather than adding it, so if a legitimate score ever falls below ≈−95 the mask leaks. Measured: at a legitimate score of −105, **99.87% of probability escapes to masked positions**. The encoder is immune because it *adds* the mask.

**CODE CONNECTION.** Encoder (safe, additive): `swin_model.py:100` — `attn = attn.view(...) + mask.unsqueeze(1).unsqueeze(0)`. Decoder (fragile, absolute): `decoder_model.py:47` — `scores.masked_fill(mask == 0, float(-100.0))`. Mask built at `swin_model.py:150-171`.

**TENSOR SHAPES.** Shift mask (64,49,49) at stage 1 — verified: **15/64 windows carry any mask**, 49/64 fully unmasked. Broadcast: `(1,64,1,49,49) + (B,64,3,49,49)`.

### Q12 — What is relative position bias, why 169, why relative?

**ANSWER.** A learned scalar added to the attention **score** for each pair of positions, depending only on their relative offset $(\Delta h, \Delta w)$ — not their absolute location. Swin uses this **instead of** any positional encoding in the encoder.

**MATHEMATICAL INTUITION.** In a 7×7 window, offsets range over $\{-6,\dots,+6\}$ on each axis: $13 \times 13 = 169$ distinct offsets. Relative encoding is **translation-invariant** — "the token directly above me" means the same thing wherever the window sits in the image. An absolute scheme would have to relearn that relationship at every location.

**CODE CONNECTION.** Table: `swin_model.py:63-66`, shape ((2·7−1)², heads) = (169, heads), the **only** explicitly initialized tensor in the codebase (`trunc_normal_(std=0.02)`). Index: lines 68-78. Applied: lines 94-96. **Audit verification:** index range [0,168] against a 169-row table → no out-of-bounds; all 169 offsets used; diagonal constant at index 84; translation-invariance confirmed.

**TENSOR SHAPES.** Checkpoint tables: (169,3), (169,6), (169,12), (169,24). Gathered → (49,49,heads) → permuted (heads,49,49) → unsqueezed (1,heads,49,49) → broadcast onto (B_,heads,49,49).

### Q13 — Why does QKᵀ give 49×49, and what is one element?

**ANSWER.** Inside one window there are 49 tokens. Each produces a query and a key of dimension $d_k = 32$. $Q \in (49,32)$, $K^\top \in (32,49)$, so $QK^\top \in (49,49)$. **Element $(i,j)$ is the dot product $Q_i \cdot K_j$** — the raw relevance of token $j$ to token $i$, before scaling, bias, and softmax.

**MATHEMATICAL INTUITION.** $Q_i \cdot K_j = \sum_{d=1}^{32} Q_{i,d}K_{j,d}$ — large when the two vectors point in similar directions in the learned 32-dimensional space. The transpose exists purely so the inner dimensions match: you need $(49,32)\times(32,49)$ to get every query against every key. Row $i$ becomes token $i$'s attention distribution over the whole window after softmax.

**CODE CONNECTION.** `swin_model.py:92` — `attn = q @ k.transpose(-2, -1)`, with `q = q * self.scale` applied one line earlier. Then `+ bias` (line 96), optional `+ mask` (line 100), then `softmax` (line 103), then `attn @ v` (line 104).

**TENSOR SHAPES.** Stage 1, 3 heads: (B_,3,49,32) @ (B_,3,32,49) → **(B_,3,49,49)** → softmax → @ (B_,3,49,32) → (B_,3,49,32) → transpose+reshape → (B_,49,96).

### Q14 — Why Q from caption and K/V from image? What breaks if reversed?

**ANSWER.** The question being asked at each decoding step is *"given the words I have written so far, which parts of the image matter for my next word?"* That question **originates in the language stream**, so the caption supplies the **Query**. The image is the fixed source of information being consulted, so it supplies **Key** (what I contain) and **Value** (what I hand over if selected).

**What breaks if reversed:** $Q \in (49,768)$ from the image and $K,V \in (T,768)$ from the caption gives an attention matrix of shape **(49, T)** and an output of shape **(49, 768)** — one vector per *image region*, not per *caption position*. You would then have nothing of length $T$ to feed to `fc_out`, so you could not produce a per-word distribution at all. The shape mismatch makes autoregressive generation structurally impossible, not merely worse.

**MATHEMATICAL INTUITION.** Attention output always inherits the **query's** sequence length. To predict $T$ words you need $T$ output vectors; therefore the query must be the caption.

**CODE CONNECTION.** `decoder_model.py:76` — `self.cross_attn(x, memory, memory)`. In `MultiHeadAttention.forward(q, k, v, mask)`, `x`→`q_proj`, first `memory`→`k_proj`, second `memory`→`v_proj`. No mask is passed, correctly: a word may always see the entire image (unlike self-attention, which must not see future words).

**TENSOR SHAPES.** Executed: Q (1,8,T,96), K (1,8,49,96), V (1,8,49,96) → scores **(1,8,T,49)** → @V → (1,8,T,96) → (1,T,768).

### Q15 — What does one element of the T×49 cross-attention matrix mean?

**ANSWER.** After scaling and softmax, element $(i,j)$ is **the probability mass caption position $i$ places on image region $j$** when computing its next-word representation. Row $i$ sums to 1 across the 49 regions. Region $j$ maps to grid cell (j//7, j%7) and thence to pixels [32·row, 32·row+32) × [32·col, 32·col+32).

**MATHEMATICAL INTUITION.** Each row is a spatial saliency map over the image, for one word. In a healthy model, generating "liver" should concentrate mass on the liver's tokens.

**AUDIT FINDING — measured, and this is the important part.** In your trained checkpoint the rows are **almost perfectly uniform after the first token**: entropy 3.860 nats versus ln(49) = 3.8918 = **99.2% of maximum**. Uniform attention means the output is simply the *average* of all 49 image tokens — a single fixed vector, identical for every word. Since the 49 tokens are themselves near-identical (cos = 0.9977), even the peaked step-0 row (entropy 1.371) conveys little. **The bridge is intact structurally and empty informationally.**

**TENSOR SHAPES.** (1,8,T,49); averaged over the 8 heads for interpretation → (T,49); one row → (49,).

### Q16 — Why is causal masking required?

**ANSWER.** During training the model is fed the **entire ground-truth caption at once** (teacher forcing). Without a causal mask, position $t$ could attend to position $t+1$ — which literally contains the word it is being asked to predict. The model would learn the trivial identity "copy my next input token," achieve near-zero training loss, and produce garbage at inference time, where no future tokens exist.

**MATHEMATICAL INTUITION.** The mask enforces the autoregressive factorization
$$p(w_1,\dots,w_T) = \prod_{t=1}^{T} p(w_t \mid w_{<t})$$
Every conditional must depend only on the past. It also lets all $T$ positions be trained in **one parallel forward pass** while still behaving as if generated sequentially.

**CODE CONNECTION.** `decoder_model.py:100` — `torch.tril(torch.ones(T,T)).view(1,1,T,T)`, applied in `MultiHeadAttention` at line 47. Passed **only** to `self_attn` (line 75), never to `cross_attn` (line 76) — correct, since restricting image access serves no purpose. **Audit result: verified by execution, token $t$ attends to exactly $\{0..t\}$, zero leakage.**

**TENSOR SHAPES.** mask (1,1,T,T) broadcasts onto scores (B,8,T,T).

### Q17 — Why shift caption inputs and targets by one?

**ANSWER.** Because the task is *next*-word prediction. The decoder receives positions $0..T{-}2$ and must predict positions $1..T{-}1$; the shift is what makes the label at each position be the following word rather than the current one.

**MATHEMATICAL INTUITION.** For the caption `<sos> a chest radiograph shows pneumonia <eos>`:

| input | `<sos>` | a | chest | radiograph | shows | pneumonia |
|---|---|---|---|---|---|---|
| **target** | a | chest | radiograph | shows | pneumonia | `<eos>` |

Every column is one training example. `<sos>` is input-only (it seeds generation); `<eos>` is target-only (the model must learn to emit it, or generation never terminates).

**CODE CONNECTION.** `caption_model.py:29` — `tgt_in = captions[:, :-1]`; `train.py:102` — `targets = captions[:, 1:]`. **Audit result: verified, no off-by-one.**

**TENSOR SHAPES.** captions (B,40) → input (B,39) → logits (B,39,7886); targets (B,39). Flattened for the loss: (B·39, 7886) vs (B·39,).

### Q18 — Why ignore PAD in cross-entropy?

**ANSWER.** Captions vary in length but must be stacked into one rectangular tensor, so short ones are right-padded with `<pad>` to 40. Those positions carry no linguistic content. Scoring them would teach the model to predict `<pad>` — which, since padding dominates short captions, would make "always emit `<pad>`" a strong loss-minimizing strategy.

**MATHEMATICAL INTUITION.** `ignore_index` removes those positions from **both** the numerator and the denominator of the averaged loss. In your data, mean caption length is 21.2 of 40 slots, so roughly **47% of all positions are padding** — without masking, nearly half the training signal would be noise pushing toward a degenerate solution.

**CODE CONNECTION.** `train.py:89` — `CrossEntropyLoss(ignore_index=vocab.pad_id, label_smoothing=0.1)`; `vocab.py:29` guarantees `pad_id = 0`. **Audit measurement:** across the 120 validation images, 2,607 non-pad positions out of 120×39 = 4,680 total → **44.3% padding**, confirming the estimate.

**TENSOR SHAPES.** targets (B·39,) with `pad_id` entries excluded; loss averaged over surviving positions only.

### Q19 — Why can this model produce a medically plausible but visually incorrect caption?

**ANSWER.** Because two capabilities are being learned by two different halves of the network at very different rates, and only one of them succeeded. The **decoder** learned genuine radiology language statistics — it beats a context-free unigram baseline by 1.67 nats and produces grammatical, idiomatic clinical sentences. The **encoder** did not learn to distinguish images. So the model generates the *most probable radiology sentence*, essentially unconditioned on what it is looking at.

**EVIDENCE FROM YOUR OWN MODEL.** cos(real chest CT, random noise) = **0.9987** > cos(real chest CT, real angiogram) = **0.9793**. An all-black image yields *"mri of the right foot showing a mass in the left side of the left femur."* Cross-attention entropy after step 0 is **99.2% of uniform**. Three caption templates cover **77.5%** of 120 validation images. Encoder weights sit at **1.00–1.06×** their random initialization.

**Dataset amplification.** CT appears in 28.5% of training captions versus ultrasound 3.9%; of captions mentioning "abdomen", **64.8%** also mention CT. `"ct scan of"` is the second most common three-word opening in the training set. A decoder with no reliable visual signal will fall back on exactly these regularities.

**The concrete mechanism, from the `007757` trace.** At step 4 the model scored `abdomen` 0.301 versus `chest` 0.275 — a **0.026 margin**, essentially a coin flip. Greedy decoding committed, and the remaining 11 tokens (`showing a mass in the right lobe of the liver`) followed coherently from that one wrong choice. The output is fluent and clinically well-formed precisely *because* the language model is good; it is wrong because the vision model is not contributing.

**CODE CONNECTION.** `caption_model.py:59` — `next_id = step_logits.argmax(-1, keepdim=True)`, no beam search, no way to revisit step 4.

### Q20 — How would you *prove* representation collapse rather than assert it?

**ANSWER.** Five independent measurements, all of which I ran on your checkpoint. Asserting collapse from "the captions look repetitive" is not proof — repetitive captions are consistent with a fine encoder and a lazy decoder. You must measure the encoder output directly.

**1. Variance decomposition.** Split the variance of the 49×768 tensor into between-token and within-token components.
> Measured: **0.225%** of total variance distinguishes spatial positions. 99.775% is a shared constant.

**2. Pairwise cosine similarity between the 49 tokens.**
> Measured: mean **0.9977** (007757), **0.9995** (000004); 96–100% of pairs exceed 0.99.

**3. Feature-norm dispersion.** Collapsed representations have near-constant magnitude.
> Measured: 23.774 ± **0.026**; a relative spread of 0.1%.

**4. THE DECISIVE TEST — synthetic controls.** Feed all-black, all-white, uniform noise, and 50% grey. If the encoder responds to content, these must produce clearly different representations from real medical images.
> Measured: cos(real CT, **noise**) = **0.9987** *exceeds* cos(real CT, real angiogram) = **0.9793**. A grey rectangle produced the **identical caption** to a real angiogram. **This is the single most convincing piece of evidence**, because it cannot be explained by any decoder-side effect.

**5. Cross-attention entropy** against the ln(49) = 3.8918 uniform ceiling.
> Measured: **3.860 = 99.2% of uniform** after step 0.

**And one control against a false positive:** repetitive output alone would *not* have proven encoder collapse — a well-trained encoder feeding a lazy decoder produces the same symptom. Test 4 is what separates the two hypotheses, because it manipulates the *input* and observes the *encoder output* directly, bypassing the decoder entirely.

**CODE CONNECTION.** All five run on `memory = model.encoder(image_tensor)` — shape (1,49,768) — requiring no architectural change whatsoever.

**RECOMMENDATION.** Add tests 1, 2, and 4 as a permanent `representation_health()` function invoked every few epochs during training. Collapse detected at epoch 2 is a hyperparameter adjustment; collapse detected after a completed run is a wasted run.
