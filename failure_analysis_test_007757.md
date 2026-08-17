# Root-Cause Analysis: Why `swin_caption_best.pt` Mis-captioned `ROCOv2_2023_test_007757.jpg`

Every number in this document was produced one of two ways, and each claim is labeled which:

- **[VERIFIED]** — computed directly, in this session, from your actual files: `swin_caption_best.pt` (the real checkpoint, torch-free-unpickled to read its stored tensors' shapes/metadata), `train_captions.csv` / `test_captions.csv` (the real ROCOv2 CSVs extracted from your uploaded `rocov2.zip`), `collapse_check.csv` (your own prior `eval.py` run), and every `.py` file in the project.
- **[REQUIRES LOCAL EXECUTION — CANNOT DETERMINE HERE]** — this sandbox has no working `torch` (no network access to `pip install` it, and the project's own `.venv` contains a macOS/arm64 build that cannot run on this Linux sandbox — confirmed by trying both). Any claim needing an actual forward pass (real attention weights, real token statistics, real per-step logits) is explicitly marked, and Section 20 gives you a ready-to-run script — using your unmodified modules plus one small, additive, external inspection wrapper — to produce those exact numbers on your own machine.

I am not going to guess these numbers. Where I don't have them, I say so.

---

## 1. Executive diagnosis

**The single most important finding in this whole investigation: the "ground truth" caption you gave me for `ROCOv2_2023_test_007757` is not that image's actual caption. [VERIFIED]**

Your prompt states the ground truth for `007757` is:
> "Operative planning ultrasound prior to ultrasound-guided subcutaneous collection drainage and appendicolith retrieval..."

I searched `test_captions.csv` directly. That sentence is the caption for a **different** image:

```
ID                          Caption
ROCOv2_2023_test_000036     Operative planning ultrasound prior to ultrasound-guided
                             subcutaneous collection drainage and appendicolith retrieval...
```

The **actual** row for `ROCOv2_2023_test_007757` in `test_captions.csv` is:

```
ID                          Caption
ROCOv2_2023_test_007757     Chest computed tomography before endobronchial ultrasound-guided
                             transbronchial needle aspiration demonstrated enlargement of the
                             right paratracheal lymph node.
```

This changes the diagnosis substantially. Re-comparing the model's actual output against the **correct** ground truth:

| | Model generated | Actual ground truth (`007757`) |
|---|---|---|
| Modality | "**ct** scan" | "Chest computed **tomography**" — **the model got the modality right** |
| Region | "of the **abdomen**" | "**Chest** ... right paratracheal" — **wrong region** |
| Finding | "a **mass** in the right lobe of the **liver**" | "**enlargement** of the right paratracheal **lymph node**" — wrong finding, though both involve a right-sided, roughly-liver-adjacent/mediastinal location and an enlargement/mass-type abnormality |

So the failure is **not** "hallucinated CT when the truth was ultrasound" (your framing) — it is **"got modality right, got organ system/region wrong (chest vs. abdomen), got the specific finding wrong (lymph node enlargement vs. liver mass)."** That is a meaningfully different, and less alarming, failure mode than the one described in the prompt. I'll analyze both the corrected case and note where your original framing (using the `000036` caption) would have led to a different, incorrect conclusion.

**Second finding, independent of the ID mixup: the checkpoint under test is confirmed, byte-for-byte, to be the stale 2,000-sample checkpoint, not a 15,000-sample one. [VERIFIED]** (Section 13, Section 14.) It stopped improving at epoch 3 with `val_loss ≈ 4.997` (perplexity $e^{4.997}\approx 148$, vs. $\ln(2682)\approx 7.9$ for a uniform random guess over the vocabulary — so it is far better than random, but nowhere near converged). Your own `collapse_check.csv` (a real `eval.py` run you already did, re-analyzed here in Section 14) shows this exact checkpoint produces the **identical** caption for 13 of 20 test images. This is strong, already-collected, real evidence of partial representation/language collapse — independent of anything about image `007757` specifically.

**My ranked assessment (justified in full in Section 15):** the wrong caption on `007757` is best explained by **(1) severe undertraining of this specific checkpoint** (3 epochs on 2,000 images) compounding with **(2) a genuine, dataset-measurable language prior** ("abdomen" co-occurs with "CT" in 64.8% of training captions that mention abdomen — Section 13), rather than by an implementation bug, a tokenization defect, or a train/inference preprocessing mismatch (all three of which I checked directly against the code and found to be **correct**, Section 9 and Section 12).

---

## 2. Complete codebase architecture — dependency / data-flow map

**[VERIFIED — read directly from every file]**

```
vocab.py            Vocab class: regex tokenizer, itos/stoi, encode()/decode()
   ↑ used by
dataset.py           build_vocab_from_csv() -> Vocab   |   ROCODataset(csv, images_dir, vocab, ...)
   ↑ used by
train.py              imports ROCODataset, build_vocab_from_csv, SwinCaptioningModel
                       builds vocab from train_captions.csv (15,000 rows)
                       builds train/valid ROCODataset + DataLoader
                       builds SwinCaptioningModel, AdamW (build_param_groups), LambdaLR scheduler
                       runs the training loop -> writes swin_caption_best.pt / swin_caption_last.pt

swin_model.py         PatchEmbed, window_partition/reverse, WindowAttention, Mlp, SwinBlock,
                       PatchMerging, BasicLayer, SwinEncoder
decoder_model.py      PositionalEncoding, MultiHeadAttention, DecoderLayer, CaptionDecoder
   ↑ both imported by
caption_model.py      SwinCaptioningModel(encoder=SwinEncoder, decoder=CaptionDecoder)
                       .forward(images, captions)   -- training path, teacher forcing
                       .generate(images, vocab, ...) -- inference path, greedy autoregressive
   ↑ used by
generate.py            loads a checkpoint, rebuilds vocab from ckpt["vocab_itos"],
                       rebuilds SwinCaptioningModel, loads state_dict, calls .generate()
eval.py                same load pattern as generate.py, but loops over a whole split,
                       scores with metrics.py (BLEU/ROUGE/CIDEr), reports % distinct captions
metrics.py             corpus_bleu / corpus_rouge_l / corpus_cider_d — pure-stdlib, no deps
```

`irdid.ipynb.py` is a dead, unimported duplicate of `vocab.py` (confirmed: nothing imports it). `main.py` is empty.

**Nothing in this dependency graph is unusual or wrong.** Every file's imports resolve to exactly what its docstring/comments claim.

---

## 3. Config discrepancy check — paper description vs. actual code vs. actual checkpoint

**[VERIFIED]** — I did not assume the code matches your stated architecture; I read every default and then independently cross-checked it against the actual tensor shapes stored inside `swin_caption_best.pt` (extracted via a torch-free unpickler, since I cannot run real `torch` here — see Section 20 for why).

| Config item | Code default (`swin_model.py` / `decoder_model.py` / `caption_model.py`) | Checkpoint's actual stored shapes | Match? |
|---|---|---|---|
| Encoder depths | `(2,2,6,2)` | stage 0: 2 blocks, stage 1: 2 blocks, stage 2: 6 blocks, stage 3: 2 blocks | **Yes** |
| Encoder heads | `(3,6,12,24)` | not directly recoverable from weight *shapes* (head count is a runtime split of the same `qkv` weight, not a separate stored dimension) — only checkable by running the model | Cannot verify from checkpoint alone; matches code default |
| Encoder channel widths | `96→192→384→768` | `patch_embed.proj.weight (96,3,4,4)`, stage dims 96/192/384/768 confirmed from `qkv.weight` second dim, `encoder.norm.weight (768,)` | **Yes** |
| Window size | 7 | `relative_position_bias_table (169, heads)` → $169=(2\times7-1)^2$, confirms window_size=7 | **Yes** |
| Patch merging present after stages 0,1,2 only | `downsample=None` for last stage | `encoder.layers.{0,1,2}.downsample.reduction.weight` present, **no** `encoder.layers.3.downsample.*` key | **Yes** |
| $d_{model}$ (decoder) | 768 | `decoder.embed.weight (2682, 768)`, `decoder.layers.0.self_attn.q_proj.weight (768,768)` | **Yes** |
| Decoder FFN dim | `4×768=3072` | `decoder.layers.0.ff.0.weight (3072, 768)`, `ff.2.weight (768, 3072)` | **Yes** |
| Decoder layers | 6 | keys exist for `decoder.layers.{0..5}.*`, none beyond | **Yes** |
| Decoder heads | 8 | not recoverable from checkpoint shape (same reason as encoder heads) | Cannot verify from checkpoint alone; matches code default |
| max caption length | 40 | `decoder.pos_enc.pe` buffer shape `(1, 40, 768)` | **Yes** |
| Final visual token count | $7\times7=49$ | consistent with stage-3 resolution (56→28→14→7 after 3 merges) and `encoder.norm.weight (768,)` being the last op before the decoder receives memory | **Yes** |
| Vocabulary size | not fixed in code — determined at vocab-build time | `decoder.embed.weight` and `decoder.fc_out.weight` both `(2682, 768)` | consistent internally, but **2,682 is the 2,000-sample vocab, not the 15,000-sample one** — see Section 13 |

**Conclusion: there is no architecture mismatch between the code's declared config and what's actually saved in the checkpoint.** Every shape that *can* be checked from the checkpoint alone matches the code's defaults exactly. The only real issue found in this section is **which training run produced this checkpoint** (Section 13), not a structural bug.

---

## 4. Image → caption tensor flow (full pipeline, shapes only — architecture-determined, not data-dependent)

**[VERIFIED — pure shape arithmetic from the code, does not require execution]**

| # | Stage | Code | Input shape | Output shape |
|---|---|---|---|---|
| 1 | Resize + normalize | `generate.py`'s `transform` | raw JPEG | `(1,3,224,224)` |
| 2 | Patch embed | `PatchEmbed.forward` | `(1,3,224,224)` | `(1,3136,96)` |
| 3 | Stage 1 (2 blocks, W-MSA/SW-MSA) | `BasicLayer` | `(1,3136,96)` | `(1,3136,96)` |
| 4 | Patch merge | `PatchMerging` | `(1,3136,96)` | `(1,784,192)` |
| 5 | Stage 2 (2 blocks) | `BasicLayer` | `(1,784,192)` | `(1,784,192)` |
| 6 | Patch merge | `PatchMerging` | `(1,784,192)` | `(1,196,384)` |
| 7 | Stage 3 (6 blocks) | `BasicLayer` | `(1,196,384)` | `(1,196,384)` |
| 8 | Patch merge | `PatchMerging` | `(1,196,384)` | `(1,49,768)` |
| 9 | Stage 4 (2 blocks, no merge after) | `BasicLayer` | `(1,49,768)` | `(1,49,768)` |
| 10 | Final norm | `SwinEncoder.forward`'s `self.norm` | `(1,49,768)` | `(1,49,768)` = **`memory`** |
| 11 | Caption embed + pos enc | `CaptionDecoder` | ids `(1,T)` | `(1,T,768)` |
| 12 | 6× decoder layer (self-attn → cross-attn → FFN) | `DecoderLayer` ×6, `memory` fed to every layer | `(1,T,768)` + `(1,49,768)` | `(1,T,768)` |
| 13 | Output projection | `decoder.fc_out` | `(1,T,768)` | `(1,T,2682)` logits |
| 14 | Softmax + argmax (inference) | `SwinCaptioningModel.generate` | `(1,2682)` (last position) | 1 token id |
| 15 | Repeat 11–14, append token, until `<eos>`/`max_len=40` | `generate()` loop | — | final `ids` sequence |
| 16 | Decode | `Vocab.decode` | ids | caption string |

---

## 5. Swin encoder trace, for the actual two images

**[Shapes VERIFIED from code; pixel-dependent values REQUIRE LOCAL EXECUTION]**

The *shape* at every stage is identical for every 224×224 input — it does not depend on image content, so I can give you this table with full confidence for `ROCOv2_2023_test_007757.jpg` (and any other image) without running anything:

| Stage | Tensor shape | Operation | Meaning |
|---|---|---|---|
| Input | `224×224×3` | resize + ImageNet normalize | raw pixels |
| Patch embedding | `56×56×96` (pre-flatten) / `3136×96` (post-flatten) | `Conv2d(3,96,k=4,s=4)` + flatten + LN | one 96-dim vector per 4×4 pixel patch |
| Stage 1 (×2 blocks) | `3136×96` | W-MSA / SW-MSA, 3 heads, window 7 | local attention within 64 windows of 49 tokens |
| Patch merge 1 | `784×192` | concat 2×2 → LN → Linear(384→192) | resolution ÷2, channels ×2 |
| Stage 2 (×2 blocks) | `784×192` | 6 heads, window 7 | 16 windows of 49 tokens |
| Patch merge 2 | `196×384` | same op | resolution ÷2, channels ×2 |
| Stage 3 (×6 blocks) | `196×384` | 12 heads, window 7 | 4 windows of 49 tokens |
| Patch merge 3 | `49×768` | same op | resolution ÷2, channels ×2 |
| Stage 4 (×2 blocks) | `49×768` | 24 heads, window 7 = grid size 7 → **global** attention (all 49 tokens are one "window") | full-image self-attention at the coarsest resolution |
| Final | `49×768` | LayerNorm | = `memory`, fed to the decoder |

**What I cannot give you without running the model:** the actual numeric contents of `memory` for this specific image — i.e., whether the values at this stage already look "confused" (e.g., unusually flat/uniform across the 49 tokens) or look like a normal, differentiated Swin-T feature map. That requires executing `encoder(image_tensor)` with the real checkpoint weights, which needs `torch`. **[REQUIRES LOCAL EXECUTION]** — script in Section 20.

---

## 6. The 49×768 representation: what each token spatially represents

**[VERIFIED — this mapping is pure indexing logic in the code, does not depend on pixel values, so I can derive it exactly]**

Trace how spatial position survives every reshape:

- `PatchEmbed.forward`: `x.flatten(2).transpose(1,2)` on a `(B,96,56,56)` conv output flattens in **row-major order**: token index $i$ at the `56×56` stage corresponds to grid position $(\text{row}=i \,//\, 56,\ \text{col}=i \,\%\, 56)$.
- `PatchMerging.forward`: builds the next stage from `x.view(B,H,W,C)` then slices `x[:, 0::2, 0::2, :]` etc. — this preserves row-major ordering at the new, halved resolution; it does not shuffle tokens.
- `window_partition`/`window_reverse` inside each `SwinBlock` temporarily reshuffle tokens into windows for attention, but `window_reverse` restores the exact original row-major layout before the block returns — so ordering downstream of a block is unaffected by windowing.

Therefore, at the final stage, **token index $t \in \{0,\ldots,48\}$ maps to grid position**:

$$
\text{row} = t \,//\, 7, \qquad \text{col} = t \,\%\, 7
$$

and, since $224 / 7 = 32$, each of these 49 tokens' receptive field ultimately traces back to (approximately — attention lets information mix across regions by this depth, so this is the *token's origin*, not a hard boundary) a **$32\times32$-pixel block** of the resized input:

$$
\text{pixel rows } [\,32\cdot\text{row},\ 32\cdot\text{row}+32\,), \qquad \text{pixel cols } [\,32\cdot\text{col},\ 32\cdot\text{col}+32\,)
$$

| Token | row,col | Pixel region (rows, cols) |
|---|---|---|
| 0 | (0,0) | rows 0–32, cols 0–32 (top-left corner) |
| 1 | (0,1) | rows 0–32, cols 32–64 |
| ... | ... | ... |
| 6 | (0,6) | rows 0–32, cols 192–224 (top-right corner) |
| 7 | (1,0) | rows 32–64, cols 0–32 |
| ... | ... | ... |
| 24 | (3,3) | rows 96–128, cols 96–128 (**image center**) |
| ... | ... | ... |
| 42 | (6,0) | rows 192–224, cols 0–32 (bottom-left) |
| 48 | (6,6) | rows 192–224, cols 192–224 (bottom-right corner) |

This mapping is exact and code-verified; it holds for every image, not just `007757`.

**Diversity/collapse question — is the representation diverse, low-variance, or collapsed for this image?**
**[REQUIRES LOCAL EXECUTION for this specific image's numbers]** — computing mean, std, min/max, per-token variance, pairwise cosine similarity, and feature-norm distribution over the real 49×768 tensor requires an actual forward pass. I will not estimate these. Section 20's script computes exactly this (`token_diversity_report()`), for both `007757` and `000004`, printing the full statistics table you asked for.

**What I *can* say with real evidence, without running this image specifically:** your own `collapse_check.csv` (Section 14) shows this exact checkpoint emits the *identical* caption for 13 of 20 *different* test images — images that have visibly different ground-truth findings. That is strong indirect evidence of collapse *somewhere in the model* (most likely the decoder dominating over a weak/underfit encoder signal — see Section 15), even before looking at `007757`'s specific tensor.

---

## 7. Caption embedding + positional encoding

**[VERIFIED from code]**

`decoder.embed`: `nn.Embedding(2682, 768)` — looked up from `decoder_model.py:88`. Given a caption id sequence of length $T$ (the decoder input is `captions[:, :-1]`, so $T=39$ during training, and grows by 1 each generation step during inference), the embedding lookup produces:

$$
\text{Embedding}(\text{tgt}) \in \mathbb{R}^{1 \times T \times 768}
$$

The fixed sinusoidal table (`decoder.pos_enc.pe`, confirmed shape `(1,40,768)` in the checkpoint) is added elementwise, sliced to the current length:

```python
def forward(self, x):
    return x + self.pe[:, : x.size(1)]
```

i.e. `x + self.pos_enc.pe[:, :T]`. Output shape unchanged: `(1, T, 768)`.

---

## 8. Masked self-attention analysis, with a concrete toy example

**[Mechanism VERIFIED from `decoder_model.py:25-51,96-105`; real numeric weights REQUIRE LOCAL EXECUTION]**

Code (`MultiHeadAttention.forward`, called from `DecoderLayer.forward` as `self.self_attn(x, x, x, tgt_mask)`):

```python
Q = self.q_proj(q).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
K = self.k_proj(k).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
V = self.v_proj(v).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)
if mask is not None:
    scores = scores.masked_fill(mask == 0, float(-100.0))
attn = torch.softmax(scores, dim=-1)
out = attn @ V
```

Since `self.self_attn(x, x, x, ...)` — $q=k=v=x$ — this is genuine self-attention: every caption token queries every other caption token in its own sequence.

**Toy numeric walkthrough** (a 4-token caption prefix `<sos> ct scan of`, $d_{model}=768$, 8 heads, $d_k=96$; using a single head and tiny illustrative numbers to make the mechanics concrete — these are not real trained weights):

$$
x = \begin{bmatrix}x_1\\x_2\\x_3\\x_4\end{bmatrix} \in \mathbb{R}^{4\times768} \quad\text{(4 tokens: <sos>, ct, scan, of)}
$$

$Q = xW_Q,\ K=xW_K,\ V=xW_V$, each $(4\times96)$ for one head. $QK^\top \in (4\times4)$: entry $(i,j)$ is "how much does token $i$'s query match token $j$'s key," before masking. The **causal mask** (`torch.tril(torch.ones(T,T))`) zeros out — via `masked_fill(mask==0, -100.0)` — every entry where $j>i$:

$$
\text{mask} = \begin{bmatrix}1&0&0&0\\1&1&0&0\\1&1&1&0\\1&1&1&1\end{bmatrix}
\quad\Rightarrow\quad
\text{scores after masking (row 2, i.e. token "ct")} = [\,s_{21},\ s_{22},\ -100,\ -100\,]
$$

After softmax, row 2 ("ct") gets nonzero weight *only* on positions 1 (`<sos>`) and 2 (`ct` itself) — it is mathematically forbidden from seeing "scan" or "of," which haven't been generated yet at that point in a real autoregressive rollout. This is **why** the mask exists: without it, position 2 could trivially attend to position 3/4 and "cheat" by copying the very word it's supposed to be predicting, since during training the full ground-truth sequence is available in the input tensor (teacher forcing) — the mask is what forces the model to actually learn "predict the next word from context," not "copy the next input token."

**Real, this-image numbers:** what the actual masked self-attention weights look like for `007757`'s generated sequence — **[REQUIRES LOCAL EXECUTION]**, script in Section 20.

---

## 9. Cross-attention analysis — the one place image and text meet

**[Mechanism VERIFIED from `decoder_model.py:74-78`; real attention weights REQUIRE LOCAL EXECUTION]**

```python
x = self.norm2(x + self.dropout(self.cross_attn(x, memory, memory)))
```

$$
Q \leftarrow x \text{ (caption)} \qquad K,V \leftarrow \text{memory (image)}
$$

**Why Q from caption, K/V from image, and not the reverse:** the *question being asked* at every decoding step is "given the words I've written so far, which parts of the image are relevant to choosing my next word?" — that question originates from the language side, so the caption supplies the **Query**. The image supplies the **Key/Value** pair because the image is the fixed, already-computed source of *information to draw from* — the same 49×768 `memory` tensor is reused, unchanged, as $K$ and $V$ at every one of the 6 decoder layers and at every one of the (up to 40) generation steps; only the caption-side $Q$ grows as the sequence gets longer.

For a caption of length $T$, with `num_heads=8`, `d_model=768`, `d_k=96`:

$$
Q \in \mathbb{R}^{T\times768} \ \xrightarrow{\text{split heads}}\ \mathbb{R}^{8\times T\times 96}, \qquad
K, V \in \mathbb{R}^{49\times768} \ \xrightarrow{\text{split heads}}\ \mathbb{R}^{8\times 49\times 96}
$$

$$
QK^\top \in \mathbb{R}^{T\times 49} \text{ per head}
$$

**What one element $QK^\top[i,j]$ means:** the raw (pre-softmax, pre-scale) relevance of image token $j$ (one of the 49 spatial regions mapped in Section 6) to caption position $i$'s current query. After scaling by $1/\sqrt{96}$ and softmax (row-wise, over the 49 image tokens — **no causal mask here**, since a word is always allowed to look at the entire image, unlike self-attention which cannot look at future words), row $i$ becomes a probability distribution over the 49 image regions: "when producing the word at position $i$, this is how much attention is being paid to each of the 49 spatial regions of the image."

**No causal mask on cross-attention, by design** — `self.cross_attn(x, memory, memory)` is called with `mask=None` implicitly (the `DecoderLayer.forward` signature only threads `tgt_mask` into `self_attn`, not `cross_attn`). This is correct: restricting *which words* a position can see (self-attention) is about not leaking the future *answer*; restricting *which image regions* a position can see would serve no such purpose — the whole image is always legitimate context.

**"Which of the 49 tokens does the model attend to when generating 'ct'? 'scan'? 'abdomen'? 'mass'? 'liver'?" — [REQUIRES LOCAL EXECUTION].** This cannot be answered from static code reading; it depends on the actual trained $W_Q, W_K, W_V$ weights and the actual image's `memory` tensor. `decoder_model.py`'s `MultiHeadAttention.forward` currently computes `attn` internally and discards it (only returns the projected output) — it does not expose attention weights. Section 20 gives you a **minimal, additive** inspection wrapper (not a rewrite of `decoder_model.py`) that captures exactly this per your instruction not to rearchitect anything.

---

## 10. Q/K/V mathematical trace — one element, fully explained

**[VERIFIED — general mechanism from code; this section is the "teach me" requirement applied to your literal `cross_attn` line]**

```python
scores = Q @ K.transpose(-2, -1)
```

- **Exact dimensions:** for one head, one image, $Q \in (T, 96)$, $K \in (49, 96)$. `K.transpose(-2,-1)` gives $(96, 49)$. `Q @ K^\top` is a $(T,96)\times(96,49) \to (T,49)$ matrix multiply.
- **What the transpose is for:** matrix multiplication requires the inner dimensions to match. $Q$'s rows are "query vectors, one per caption position, each 96-dim"; $K$'s rows are "key vectors, one per image token, each 96-dim." To get *every query against every key* as a dot product, you need $(T,96) \times (96,49)$ — hence transposing $K$ from $(49,96)$ to $(96,49)$.
- **What one output element means:** entry $(i,j)$ of the result is $Q_i \cdot K_j = \sum_{d=1}^{96} Q_{i,d} K_{j,d}$ — the dot product of caption position $i$'s query vector and image token $j$'s key vector. A large positive value means the two vectors point in a similar direction in the learned 96-dim space, which the softmax will convert into "pay more attention to image token $j$ when producing word $i$."
- **Why scale by $\sqrt{d_k}=\sqrt{96}$:** as vector dimension grows, dot products of two random vectors grow in expected magnitude (variance scales with $d_k$ for random unit-scale vectors). Without scaling, scores can become large enough that softmax saturates — the largest score gets weight $\approx 1$ and everything else $\approx 0$, which produces vanishing gradients that make the query/key projections hard to train early on. Dividing by $\sqrt{d_k}$ keeps the score distribution at a roughly constant scale regardless of head dimension.
- **Why softmax:** turns arbitrary real-valued scores into a proper probability distribution (all $\ge 0$, sums to 1) over the 49 image tokens, so the next step ("weighted average of $V$") is a genuine convex combination — an interpretable "attention distribution" — rather than an arbitrary weighted sum that could blow up in scale.
- **Why multiply the softmax result by $V$:** the softmax output is *only weights* — it says "how much" to attend to each image token, but not *what information* to pull from it. $V$ is where the actual content lives. `attn @ V`, shape $(T,49)\times(49,96)\to(T,96)$, computes, for every caption position, a weighted average of all 49 image tokens' Value vectors — this is the actual mechanism by which visual information enters the caption representation.

```python
self.cross_attn(x, memory, memory)
```
maps directly onto this: `x`→ becomes $Q$ (via `q_proj` inside `MultiHeadAttention`), the first `memory` → becomes $K$ (via `k_proj`), the second `memory` → becomes $V$ (via `v_proj`). Semantically: *the caption asks a question of the image; the image answers with what it actually contains.* This is not just syntax — it is the one operation in the entire model where pixel-derived information and word-derived information are combined into a single tensor.

---

## 11. Autoregressive generation trace

**[Mechanism VERIFIED from `caption_model.py:33-63`; the actual step-by-step token/logit sequence for `007757` REQUIRES LOCAL EXECUTION]**

```python
ids = torch.full((B, 1), vocab.sos_id, ...)          # step 0: [<sos>]
for _ in range(max_len - 1):
    logits = self.decoder(ids, memory)                # decoder re-run on the WHOLE sequence so far
    step_logits = logits[:, -1, :].clone()             # only the last position's logits are used
    step_logits.scatter_(1, prev_token.unsqueeze(1), -inf)   # guard 1: never repeat previous token
    ... trigram-blocking loop ...                       # guard 2: never complete a seen trigram
    next_id = step_logits.argmax(-1, keepdim=True)      # greedy pick
    ids = torch.cat([ids, next_id], dim=1)
    if (next_id == vocab.eos_id).all(): break
```

Note precisely what happens each step: the **entire** `ids` sequence built so far is re-fed through all 6 decoder layers every single step (this is not a KV-cache-incremental implementation — it recomputes self-attention over the whole growing prefix each time), and only the *last* position's output logits are used to pick the next token. Shapes at step $t$ (sequence length $t{+}1$ including `<sos>`):

| | shape |
|---|---|
| decoder input `ids` | `(1, t+1)` |
| $Q$ (self-attn, 8 heads) | `(8, t+1, 96)` |
| $K,V$ (self-attn) | `(8, t+1, 96)` |
| $Q$ (cross-attn) | `(8, t+1, 96)` |
| $K,V$ (cross-attn, = `memory`) | `(8, 49, 96)` — fixed size, doesn't grow |
| cross-attention matrix | `(8, t+1, 49)` |
| logits used | last row only: `(1, 2682)` |

**Was the model "already biased toward CT before seeing meaningful image information"?** This is answerable in principle by comparing step-1 logits (context = `<sos>` only, i.e. *before* cross-attention has anything caption-side to work with beyond the seed token) against the eventual argmax choice — if "ct" or a close synonym is already the top-1 prediction at step 1 for *most/all* images regardless of content, that's direct evidence of an encoder-independent language prior dominating the very first token. **[REQUIRES LOCAL EXECUTION]** to get this image's actual step-1 top-10 — Section 20's script prints exactly this table (step, current sequence, top-10 tokens + probabilities, selected token) for the full generation trace on `007757`.

---

## 12. Vocabulary probability analysis

**[REQUIRES LOCAL EXECUTION — no way to obtain real softmax probabilities without running the model]**

Section 20's script (`generate_with_trace()`) prints, at every step: the current token sequence, and the top-10 `(word, probability)` pairs from `torch.softmax(step_logits, dim=-1)`. I am not fabricating example numbers here, since your prompt explicitly asked me not to invent results — run the script and this table falls out directly.

What this *would* tell you, once you have it: whether "ct" (or synonyms) already dominates the very first prediction (→ points to decoder language prior, category C in your Part 6 framing) versus only becoming dominant after a few tokens of caption context have accumulated (→ points more toward cross-attention/encoder signal being weak specifically for this image, category A/B).

---

## 13. Cross-entropy / training analysis — and why this checkpoint is the wrong one to be testing

**[VERIFIED, including a checkpoint fact not previously nailed down exactly]**

`train.py`'s loss:

```python
criterion = torch.nn.CrossEntropyLoss(ignore_index=vocab.pad_id, label_smoothing=LABEL_SMOOTHING)
...
logits = model(images, captions)          # (B, T-1, V) where T-1 = 39 (captions[:, :-1] fed in)
targets = captions[:, 1:]                  # (B, T-1) = (B, 39)
loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
```

**Shapes at the loss call:** `logits.reshape(-1, V)` → `(B·39, 2682)`; `targets.reshape(-1)` → `(B·39,)`. `CrossEntropyLoss` computes, for every one of the `B·39` positions independently, $-\log \text{softmax}(\text{logits})[\text{target}]$ (with label smoothing spreading a small amount of the target mass to other vocabulary entries, per `methodology.tex`'s Step 23 equation), then averages — **except** positions where `targets == vocab.pad_id`, which `ignore_index` excludes entirely from both the numerator and the denominator of that average.

**Concrete worked example**, ground-truth caption `<sos> ct scan of the abdomen <eos> <pad> <pad> ...`:

| decoder input (`captions[:, :-1]`) | `<sos>` | ct | scan | of | the | abdomen |
|---|---|---|---|---|---|---|
| **target** (`captions[:, 1:]`), same column | ct | scan | of | the | abdomen | `<eos>` |

At input position 0 (`<sos>`), the model's logits are compared against target `ct`; at input position 1 (`ct`), logits are compared against target `scan`; and so on — this is the concrete meaning of "shift right, predict next." I confirmed this by re-reading `train.py` and `caption_model.py.forward()` together (Section 4, rows 11–13) — there is no off-by-one error: `tgt_in = captions[:, :-1]` and `targets = captions[:, 1:]` are complementary halves of the same 40-length sequence, correctly aligned.

**Now, the checkpoint fact, verified exactly (not estimated):** I rebuilt the vocabulary exactly as `build_vocab_from_csv(min_freq=2, max_samples=2000)` would (tokenizing the first 2,000 rows of `train_captions.csv` with the same regex as `vocab.py`) and it produces **exactly 2,682 words** — an exact match to `swin_caption_best.pt`'s stored `vocab_itos` length. Combined with the checkpoint's stored `epoch: 3` and `val_loss: 4.996981...`, this is conclusive: **`swin_caption_best.pt` is the artifact of an old run using the 2,000-sample pilot config, not the current 15,000-sample `train.py`.** A cross-entropy loss of ~5.0 nats corresponds to a perplexity of $e^{4.997}\approx 148$ — i.e., on average the model assigns real probability mass as if choosing fairly confidently among ~148 plausible next words, out of a 2,682-word vocabulary. That is a partially-trained model, well short of convergence (three epochs, stopped by early-stopping's patience counter after no further improvement — meaning the model plateaued this early and never got better even with 60 epochs of budget available).

**Are the ground-truth words for `007757` even representable in this checkpoint's vocabulary?** Checked directly, word by word, against the checkpoint's actual 2,682-word `vocab_itos`:

| word | in checkpoint vocab? |
|---|---|
| ultrasound | **yes** |
| transbronchial | **yes** |
| aspiration | **yes** |
| lymph | **yes** |
| node | **yes** |
| enlargement | **yes** |
| chest | **yes** |
| ct | **yes** |
| endobronchial | **no** → would encode as `<unk>` |
| paratracheal | **no** → would encode as `<unk>` |

**This is an important, evidence-based finding that argues *against* "vocabulary coverage" as the primary cause:** almost every clinically important word in the correct ground truth was actually representable by this checkpoint's vocabulary. Only two rare technical terms (`endobronchial`, `paratracheal`) would have been forced to `<unk>`. The model had the *words* available; it simply didn't select them — which points toward the decoder's learned distribution / undertrained cross-modal alignment (Section 15), not tokenization, as the dominant factor.

---

## 14. Dataset bias analysis

**[VERIFIED — computed directly from the actual first 15,000 rows of `train_captions.csv`, i.e. exactly the rows `train.py`'s `TRAIN_SAMPLES=15000` uses]**

Modality-word frequency across the 15,000 training captions:

| modality | # captions mentioning it | % of 15,000 |
|---|---|---|
| CT / "computed tomography" | 4,277 | **28.5%** |
| radiograph / X-ray | 2,326 | 15.5% |
| MRI | 1,552 | 10.3% |
| angiogram | 633 | 4.2% |
| ultrasound / sonograph | 582 | 3.9% |
| PET | 156 | 1.0% |

**CT is, by a wide margin, the single most frequently mentioned modality in the training data** — more than 7× more frequent than ultrasound. This alone creates a strong prior toward the decoder defaulting to "CT" whenever visual evidence is weak or ambiguous (as it would be from an undertrained encoder — Section 15).

Region → modality co-occurrence, specifically relevant to this failure case:

| Given the caption mentions... | # captions | % that also mention CT | % that also mention ultrasound |
|---|---|---|---|
| "abdomen" / "abdominal" | 1,496 | **64.8%** | 4.2% |
| "chest" | 1,718 | 34.1% (radiograph 50.3%, higher than CT for chest specifically) | 0.3% |

This is a real, measurable pattern: **if a caption mentions the abdomen, it mentions CT roughly 65% of the time** in the training distribution the model actually learned from. Combined with an undertrained visual encoder that may not be reliably distinguishing "chest" from "abdomen" content yet (Section 6's diversity question — unresolved without execution), a decoder leaning on this exact statistical regularity is a plausible, dataset-grounded explanation for why "ct" and "abdomen" co-occurred in the output even though the correct region was "chest."

Additional context: only 395 of 15,000 captions mention "liver" at all, and of those only 15.2% also mention "mass" — "liver mass" is not a dominant template, so its appearance in the output is more idiosyncratic than "ct"/"abdomen," consistent with greedy decoding + trigram blocking pushing the sentence toward *some* concrete finding once the modality/region words are already locked in, rather than "liver mass" being a strongly memorized association.

Most frequent 3-word caption openings in the training set (a proxy for template strength):

```
1. "chest x ray"                372×  (2.5%)
2. "ct scan of"                 193×  (1.3%)
3. "magnetic resonance imaging" 134×  (0.9%)
4. "computed tomography scan"   133×  (0.9%)
5. "computed tomography of"     104×  (0.7%)
```

"ct scan of" alone is the **second most common caption opening in the entire training set.** A decoder that has learned this as a high-probability opening trigram, independent of image content, is architecturally exactly what your `generate()`'s trigram-blocking guard (Section 11) is designed to fight *repetition* within one caption — but it does nothing to prevent the *same* high-frequency opening being selected *across different images*, which is a distinct failure mode from the one those guards target.

---

## 15. Comparison of multiple test images

**[Ground truths VERIFIED from `test_captions.csv`; `collapse_check.csv` results VERIFIED as your own prior real `eval.py` run; representation/attention comparisons REQUIRE LOCAL EXECUTION]**

**Image A — `ROCOv2_2023_test_000004.jpg`.** Actual ground truth (verified, not the one implied by your prompt's placeholder):
> "Digitally subtracted angiogram of the IMA demonstrated cessation of flow through the proximal superior rectal artery in the region of the intersection between the artery and ureter with retained perfusion of the rectosigmoid region and resolution of active extravasation"

You reported the model's output for this image (in this conversation) as `"postoperative panoramic radiograph"`. Ground truth is a **digitally subtracted angiogram of a rectal/pelvic vascular structure**; the model output describes a **dental/oral panoramic radiograph** — modality, body region, and clinical context are all wrong here, more severely than the `007757` case. Note this doesn't match either of the two dominant templates in `collapse_check.csv` — that file only covers test images 1–20, and `000004` is in that range (row 4): `collapse_check.csv` actually recorded its generated caption as `"chest x ray image of the chest x rays after the chest"`, not `"postoperative panoramic radiograph"` — **these two outputs for the same image, from what should be the same checkpoint, disagree with each other**, which tells you the exact output you quoted for Image A was very likely generated by a *different* checkpoint or a different `generate.py` invocation than the one that produced `collapse_check.csv`. **[Cannot determine from available evidence]** which specific checkpoint file (`_best`, `_last`, or one of the `_epochN` files) produced the `"postoperative panoramic radiograph"` output — this is worth resolving on your end by checking which `.pt` path you passed to `generate.py` for that run.

**Image B — `ROCOv2_2023_test_007757.jpg`.** Covered in full in Section 1.

**Direct real evidence from `collapse_check.csv`** (your own executed `eval.py --n 20` run, re-analyzed here): out of 20 test images (IDs `test_000001` through `test_000020`), the model produced only **5 distinct captions total**, and **13 of the 20 (65%)** produced the exact identical string:

> "axial computed tomography image of the abdomen shows the small bowel loops of the small air fluid levels"

regardless of each image's real, very different ground truth (an aortic aneurysm, a thrombosis, a digitally-subtracted angiogram, a T2 MRI, an ovarian ultrasound, a neck CT, etc. — all genuinely different modalities and regions, per the ground-truth column of that same file). **This is real, already-collected, quantitative evidence that this specific checkpoint (`swin_caption_best.pt`) is not meaningfully conditioning on the image for the majority of the test set** — independent of, and stronger than, anything specific to `007757`.

**What I cannot do without execution:** compare `007757`'s actual `memory` tensor statistics, attention maps, or per-token logit trajectories against another image's, since that requires two real forward passes. Section 20's script runs `token_diversity_report()` and `generate_with_trace()` on both `007757` and `000004` so you can make this comparison directly.

---

## 16. Check for training/inference mismatch

**[VERIFIED — direct code comparison, no execution needed]**

| Check | `train.py` (validation path, since that's what's comparable to inference) | `generate.py` | Match? |
|---|---|---|---|
| Resize | `T.Resize((224,224))` | `T.Resize((224,224))` | **Yes** |
| Normalization | `mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]` | identical | **Yes** |
| Train-only augmentation (`RandomRotation`, `ColorJitter`) | present in `train_transform`, used **only** for the training DataLoader | absent | **Correct by design** — augmentation should not run at inference; `valid_ds` in `train.py` also uses the un-augmented default transform from `dataset.py`, consistent with `generate.py` |
| Tokenizer | `vocab.py`'s `Vocab.tokenize`, used identically everywhere | same | **Yes** |
| Vocabulary source at inference | `generate.py`'s `load_vocab_from_checkpoint()` restores `vocab.itos` **from the checkpoint itself** (`ckpt["vocab_itos"]`), not rebuilt from the current `train_captions.csv` | — | **Correct** — this is exactly what guarantees train/inference vocab consistency regardless of which checkpoint (old 2k-vocab or a future 15k-vocab one) is loaded |
| Special token ids | derived from `v.stoi[...]` after restoring `itos`, same 4 reserved tokens in the same order (`<pad>,<sos>,<eos>,<unk>`) as `vocab.py.__init__` | same | **Yes** |
| Checkpoint → architecture match | `SwinCaptioningModel(vocab_size=len(vocab), max_len=MAX_LEN)` then `load_state_dict(ckpt["model_state"])` — this will raise a hard error on any shape mismatch (`strict=True` is the default) | — | Confirmed no mismatch exists, since Section 3's shape audit found none |
| `eval()` mode | `model.eval()` called before generation in `generate.py:45` | — | **Yes** — dropout disabled correctly |
| Device handling | same `cuda → mps → cpu` fallback logic in both `train.py` and `generate.py` | — | **Yes**, consistent |
| Causal mask | built fresh inside `CaptionDecoder.forward` every call (`torch.tril(...)`), not cached/stale | — | **Yes**, correct at both train and inference time |
| Padding handling | `ignore_index=vocab.pad_id` at training; at inference, `<pad>` is never fed as decoder input (sequence only grows one real token at a time) | — | **No issue found** |

**No training/inference mismatch was found.** This is a genuinely clean result — it rules out a whole category of subtle bugs (Section 15 ranks this appropriately low as a result).

---

## 17. Root-cause ranking

**[Each item's confidence level stated explicitly — no generic ranking]**

1. **Checkpoint severely undertrained (HIGH confidence — directly verified).** `epoch=3`, `val_loss≈4.997` (perplexity ≈148), trained on only 2,000 images, vocabulary exactly matching the 2k-sample config. This alone would produce exactly the symptom seen: plausible-sounding, generic medical language that isn't well-conditioned on the specific image. Directly supported by `collapse_check.csv`'s 13/20 identical-caption result (Section 15).

2. **Measurable language prior toward "CT" + "abdomen" (HIGH confidence — directly verified from real training data).** CT is the most frequent modality (28.5% of captions), "abdomen"-mentioning captions co-occur with CT 64.8% of the time (Section 14). A decoder that hasn't yet learned strong cross-modal grounding (consistent with finding #1) has every statistical incentive to default toward this high-frequency combination when visual signal is weak or ambiguous.

3. **Weak/ambiguous visual representation for this specific image — UNRESOLVED, requires execution.** Cannot be confirmed or ruled out without running the real encoder forward pass and computing the diversity statistics from Section 6. This is likely a contributor given #1, but I am not claiming it as verified.

4. **Cross-attention not yet meaningfully grounding language in vision — UNRESOLVED, requires execution.** Same caveat; Section 9's script would show whether attention to the 49 image tokens is diffuse/uniform (weak grounding) or sharply localized but on the wrong region (different problem) when generating "ct"/"abdomen"/"mass"/"liver."

5. **Vocabulary/tokenization defect — RULED OUT (verified).** Nearly every ground-truth word for `007757` is representable in the checkpoint's vocabulary (Section 13); the two OOV terms are rare technical words unlikely to be selected by *any* undertrained model regardless of vocabulary design.

6. **Training/inference preprocessing or checkpoint-loading bug — RULED OUT (verified).** Section 16 found no discrepancy in resize, normalization, tokenizer, vocab restoration, special-token ids, `eval()` mode, or device handling.

7. **Architecture/config mismatch between paper description, code, and checkpoint — RULED OUT (verified).** Section 3's shape audit found the checkpoint's stored tensors match the code's declared architecture exactly.

8. **Decoding strategy (greedy + trigram blocking) — LOW confidence as a *root* cause, but a genuine amplifier.** Greedy decoding with no beam search means one wrong early token (e.g., "ct" instead of "chest") commits the whole rest of the sentence to a consistent-but-wrong narrative — there's no mechanism to reconsider. This wouldn't *cause* the initial wrong token, but it prevents recovery from it.

9. **An actual implementation bug independent of the above — RULED OUT for the two bugs already found and fixed this project (AdamW weight-decay grouping, `-inf`/`-100.0` masking inconsistency — both fixed and verified in `FINDINGS.md`/`train.py`/`decoder_model.py` already).** No *new* implementation bug was found in this investigation; everything checked in Sections 3, 13, and 16 matched expectations.

**Bottom line: the most defensible, fully evidence-backed explanation is #1 + #2 acting together** — an undertrained model falling back on the strongest statistical regularity in its training data (CT+abdomen) when its (also likely still-weak, though unconfirmed) visual grounding doesn't yet override that prior. #3 and #4 are very plausible but explicitly unconfirmed without running the actual forward pass.

---

## 18. Minimal possible improvements — and which would violate your from-scratch learning goal

*(Not proposing these as "the fix" — per your instructions, this is offered only as the diagnosis-driven menu, ranked by how much they respect the from-scratch constraint.)*

| Change | Addresses | Violates "from scratch"? |
|---|---|---|
| Re-run `eval.py --split test --n 200` against a checkpoint actually produced by the current `TRAIN_SAMPLES=15000` config (not `swin_caption_best.pt`) | Finding #1 directly | **No** — this is just finishing the experiment your own code already targets |
| Add the attention-weight-exposing inspection wrapper from Section 20 as a permanent, opt-in debug flag in `caption_model.generate()` | Lets you directly verify #3/#4 going forward | **No** — purely additive, doesn't change any computation |
| Increase training epochs / lower early-stopping patience threshold, or verify the 15k run actually got far enough to plateau meaningfully higher than epoch 3 | #1 | **No** |
| Add a small held-out "hard negative" sanity check (e.g., a handful of manually chosen images spanning very different modalities) evaluated every few epochs during training, not just aggregate val loss | Would catch collapse (finding #1/#2's symptom) earlier than waiting for a full `eval.py` run | **No** |
| Use a pretrained visual backbone (CLIP/BLIP/timm) | Finding #3 | **Yes — explicitly against your stated goal, not recommended by me** |
| Use a pretrained tokenizer / subword vocabulary | Finding #5 (already ruled out as primary cause) | **Yes, and not clearly justified by the evidence here anyway** |
| Beam search instead of greedy decoding | Finding #8 | **No** — still entirely hand-implementable, a reasonable from-scratch extension |

---

## 19. What to learn / do next

1. Run Section 20's script locally and fill in Sections 6, 9, 11, 12's "requires execution" gaps with real numbers — this converts the current ranked-but-partially-unconfirmed diagnosis (Section 17) into a fully evidence-closed one.
2. Confirm which checkpoint file actually produced the `"postoperative panoramic radiograph"` output for Image A (Section 15) — there's an unresolved inconsistency between that and `collapse_check.csv`'s recorded output for the same image ID.
3. Locate or retrain a checkpoint that reflects the current 15,000-sample `train.py` config, and re-run this entire analysis against *that* checkpoint — everything in this report about "undertrained" is specific to `swin_caption_best.pt`, not necessarily a permanent property of the architecture.
4. When you do get real attention maps (Section 20), specifically check whether they're **diffuse across all 49 tokens** (near-uniform weights) versus **peaked but on the wrong tokens** — these imply different next steps (former suggests cross-attention/encoder training hasn't converged yet; latter suggests the encoder's spatial features may be systematically confusable between chest/abdomen regions).

---

## 20. Script to run locally — fills every "REQUIRES LOCAL EXECUTION" gap above

This is additive only — it does not modify `swin_model.py`, `decoder_model.py`, or `caption_model.py`. It imports your real modules unchanged and wraps the loaded model's existing sub-layers to capture what they already compute internally (attention weights) but don't currently return, exactly as you asked ("modify the code minimally... do NOT rewrite the architecture").

Save as `inspect_007757.py` in your project folder and run with the same Python environment `train.py` uses (the one with `torch`/`torchvision` — e.g. `.venv/bin/python3`, or plain `python3` if that's what you normally use):

```python
"""
Additive inspection script -- does not modify swin_model.py / decoder_model.py /
caption_model.py. Wraps the already-loaded model's MultiHeadAttention modules to
capture the attention weights they compute internally but don't return, and
traces the generate() loop step by step.
"""
import math
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T

from vocab import Vocab
from caption_model import SwinCaptioningModel
from decoder_model import MultiHeadAttention

ROOT = "/Users/nagashiva/Downloads/rocov2"
CKPT = f"{ROOT}/swin_caption_best.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                       "mps" if torch.backends.mps.is_available() else "cpu")

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_vocab_from_checkpoint(ckpt):
    v = Vocab.__new__(Vocab)
    v.itos = ckpt["vocab_itos"]
    v.stoi = {w: i for i, w in enumerate(v.itos)}
    v.pad_token, v.sos_token, v.eos_token, v.unk_token = "<pad>", "<sos>", "<eos>", "<unk>"
    v.pad_id, v.sos_id, v.eos_id, v.unk_id = (v.stoi[v.pad_token], v.stoi[v.sos_token],
                                               v.stoi[v.eos_token], v.stoi[v.unk_token])
    return v


# ---- capture attention weights without touching decoder_model.py ----
_captured = {}

def patched_forward(self, q, k, v, mask=None):
    B = q.size(0)
    Q = self.q_proj(q).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
    K = self.k_proj(k).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
    V = self.v_proj(v).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
    scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float(-100.0))
    attn = torch.softmax(scores, dim=-1)
    _captured["last_attn"] = attn.detach()   # (B, heads, T_q, T_kv) -- this is the new part
    out = (attn @ V).transpose(1, 2).contiguous().view(B, -1, self.num_heads * self.d_k)
    return self.out_proj(out)

MultiHeadAttention.forward = patched_forward   # monkey-patch: same math, now also stores attn


def token_diversity_report(memory, label):
    # memory: (1, 49, 768)
    m = memory.squeeze(0)  # (49, 768)
    norms = m.norm(dim=-1)
    print(f"\n=== token diversity report: {label} ===")
    print("mean:", m.mean().item(), " std:", m.std().item())
    print("min:", m.min().item(), " max:", m.max().item())
    print("per-token feature norm: mean", norms.mean().item(), "std", norms.std().item(),
          "min", norms.min().item(), "max", norms.max().item())
    mn = F.normalize(m, dim=-1)
    cos = mn @ mn.T   # (49,49) pairwise cosine similarity
    off_diag = cos[~torch.eye(49, dtype=torch.bool)]
    print("pairwise cosine similarity (off-diagonal): mean", off_diag.mean().item(),
          "std", off_diag.std().item(), "min", off_diag.min().item(), "max", off_diag.max().item())
    print("  (near 1.0 mean/low std across the board => tokens nearly identical => collapse)")


def generate_with_trace(model, image_t, vocab, max_len=40, topk=10):
    model.eval()
    with torch.no_grad():
        memory = model.encoder(image_t.to(DEVICE))
        ids = torch.full((1, 1), vocab.sos_id, dtype=torch.long, device=DEVICE)
        step = 0
        while ids.size(1) < max_len:
            logits = model.decoder(ids, memory)
            step_logits = logits[:, -1, :].clone()
            probs = torch.softmax(step_logits, dim=-1)[0]
            top_p, top_i = probs.topk(topk)
            print(f"\nstep {step}: current sequence = {vocab.decode(ids[0].tolist())!r}")
            print(f"  decoder input shape: {tuple(ids.shape)}   cross-attn Q shape (last layer): "
                  f"(8, {ids.size(1)}, 96)   K/V shape: (8, 49, 96)")
            for p, i in zip(top_p.tolist(), top_i.tolist()):
                print(f"    {vocab.itos[i]:20s} {p:.4f}")

            prev_token = ids[:, -1]
            step_logits.scatter_(1, prev_token.unsqueeze(1), float("-inf"))
            if ids.size(1) >= 2:
                seq = ids[0].tolist()
                seen = {(seq[i], seq[i+1], seq[i+2]) for i in range(len(seq) - 2)}
                prefix = (seq[-2], seq[-1])
                banned = {t[2] for t in seen if (t[0], t[1]) == prefix}
                for b in banned:
                    step_logits[0, b] = float("-inf")

            next_id = step_logits.argmax(-1, keepdim=True)
            ids = torch.cat([ids, next_id], dim=1)
            # last decoder layer's cross-attention weights for the token just generated:
            attn = _captured["last_attn"]          # (1, 8, T, 49) after the self.cross_attn call
            per_image_token = attn[0].mean(0)[-1]  # average over heads, last query position -> (49,)
            top_img = per_image_token.topk(5)
            print(f"  selected token: {vocab.itos[next_id.item()]!r}   top-5 attended image tokens "
                  f"(row,col): " +
                  ", ".join(f"{i.item()}=({i.item()//7},{i.item()%7}):{v.item():.3f}"
                             for v, i in zip(*top_img)))
            step += 1
            if next_id.item() == vocab.eos_id:
                break
    return ids, memory


if __name__ == "__main__":
    ckpt = torch.load(CKPT, map_location=DEVICE)
    vocab = load_vocab_from_checkpoint(ckpt)
    model = SwinCaptioningModel(vocab_size=len(vocab), max_len=40).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    for image_id in ["ROCOv2_2023_test_007757", "ROCOv2_2023_test_000004"]:
        path = f"{ROOT}/test_images/test/{image_id}.jpg"
        image_t = transform(Image.open(path).convert("RGB")).unsqueeze(0)
        ids, memory = generate_with_trace(model, image_t, vocab)
        token_diversity_report(memory, image_id)
        print(f"\nFINAL CAPTION ({image_id}):", vocab.decode(ids[0].tolist()))
```

This gives you, directly: Section 6's real diversity numbers for both images, Section 9/11's real cross-attention token (mapped to row/col via Section 6's exact formula) for every generated word, and Section 12's real top-10 vocabulary probabilities at every step — the exact evidence needed to close out Sections 3/4/A/B in your Part 6 framing with certainty rather than inference.

---

## VIVA UNDERSTANDING CHECK

1. Your Section 9 cross-attention uses $Q$ from the caption and $K,V$ from the image. If you swapped this — $Q$ from the image, $K,V$ from the caption — what would the resulting attention matrix's shape and meaning become, and why would it no longer make sense for generating one word at a time?
2. Why is $\sqrt{d_k}=\sqrt{96}$ specifically, and not $\sqrt{768}$? What would change if you scaled by $\sqrt{768}$ instead?
3. Walk through, in your own words, why token 24 in the final 49-token grid corresponds to the image center — what three code operations (across `PatchEmbed`, `PatchMerging`, and `window_reverse`) had to each preserve row-major order for that claim to hold?
4. This report found the checkpoint's `vocab_itos` length (2,682) exactly matches a vocabulary rebuilt from only the first 2,000 training rows. Why is an *exact* integer match here much stronger evidence than the epoch/val_loss numbers alone?
5. `collapse_check.csv` shows 13/20 images getting an identical caption. Does this fact, by itself, prove the *encoder* has collapsed (i.e., produces near-identical `memory` tensors for different images)? What's a second, distinct explanation that would produce the same symptom without the encoder being at fault?
6. Section 14 found "abdomen" co-occurs with "CT" in 64.8% of training captions that mention abdomen. Suppose the *real* encoder representation for `007757` is highly distinctive and clearly chest-like. Could the decoder still output "abdomen" anyway? What does that imply about how strong a dataset prior would have to be for cross-attention to reliably override it?
7. In `MultiHeadAttention.forward`, why does masking use `masked_fill(mask == 0, -100.0)` rather than, say, multiplying the scores by the mask directly (`scores * mask`)?
8. The trigram-blocking guard in `generate()` prevents repeating a 3-token sequence *within one caption*. Explain precisely why this guard cannot prevent the *same* caption from being generated for two *different* images — what would a guard that could prevent that even look like?
9. `generate()` recomputes the full decoder forward pass over the entire `ids` sequence at every single generation step, rather than caching previous positions' key/value vectors. What is the computational cost implication of this as caption length grows, and where exactly in the code would you add KV-caching if you wanted to?
10. Suppose you re-run this exact analysis against a properly-trained 15,000-sample checkpoint and `007757` is *still* captioned as "CT of the abdomen." Given everything in Section 14–17, what would that outcome tell you that the current (undertrained, 2,000-sample) checkpoint's failure does *not* yet tell you?
