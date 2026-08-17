# From a 224×224 Image to a Radiology Caption — A Complete, Step-by-Step Math Walkthrough

This report traces **every transformation** the model applies, in order, from the raw pixel
tensor of a chest X-ray or CT slice all the way to a generated English sentence. Each step
gives: (1) what's happening in plain English, (2) the exact math, (3) the tensor shape before
and after, and (4) where it lives in the code (`swin_model.py`, `decoder_model.py`,
`caption_model.py`, `vocab.py`, `train.py`).

No prior transformer knowledge is assumed — every symbol is defined the first time it's used.
A running numeric example (a toy 5-caption vocabulary, matching the "axial CT scan showing a
large pleural effusion..." example) is threaded through the caption side so the abstract math
always has a concrete number attached to it.

---

## 0. Notation used throughout

| Symbol | Meaning |
|---|---|
| $B$ | batch size — how many images/captions are processed at once |
| $H, W$ | height / width of the current token grid (in *tokens*, not pixels) |
| $C$ | channel width / embedding dimension of a token at the current stage |
| $N$ | number of tokens = $H \times W$ |
| $d_{model}$ | the embedding size used everywhere in the decoder (768) |
| $d_k$ | dimension of one attention head = $d_{model} / \text{num\_heads}$ |
| $\mathbf{x}_i \in \mathbb{R}^C$ | the vector (token) representing position $i$ |
| $W, b$ | a learned weight matrix and bias vector of a linear layer |
| $\text{LN}(\cdot)$ | LayerNorm — rescales a vector to zero mean / unit variance, then applies a learned scale+shift |
| $\odot$ | elementwise (Hadamard) product |
| $\|\cdot\|$ | vector concatenation |

Every "token" in this model, whether it represents a 4×4 patch of pixels or an English word,
is just a vector of real numbers. The entire network is a sequence of operations that turn
one set of vectors into another, better, set of vectors.

---

## PART A — The Image Side (Swin Encoder): pixels → 49 feature tokens

### Step 1 — The raw input

A radiology image is resized to **224×224** pixels with 3 color channels (RGB — even a
grayscale X-ray is stored as 3 identical or near-identical channels once loaded by
`torchvision`). As a tensor:

$$
X \in \mathbb{R}^{B \times 3 \times 224 \times 224}
$$

Before entering the network it is normalized per-channel using ImageNet statistics
(`train.py`'s `train_transform`):

$$
x'_{c,h,w} = \frac{x_{c,h,w} - \mu_c}{\sigma_c}, \quad \mu = (0.485, 0.456, 0.406),\ \sigma = (0.229, 0.224, 0.225)
$$

This just rescales pixel intensities to roughly the range the network's random initialization
expects — it doesn't change what's in the image, only its numeric scale.

**Shape:** `(B, 3, 224, 224)` in → `(B, 3, 224, 224)` out (unchanged, just rescaled).

---

### Step 2 — Splitting the image into patches (`PatchEmbed`, `swin_model.py:9-22`)

A transformer operates on a *sequence of vectors*, not a 2-D grid of pixels. So the very
first job is deciding how a picture becomes a sequence.

**The idea:** cut the 224×224 image into a grid of non-overlapping **4×4** squares. Each
square (a "patch") contains $4 \times 4 \times 3 = 48$ raw pixel values. Flatten those 48
numbers into one vector, then apply one shared linear layer to project every patch into a
96-dimensional vector.

$$
\text{grid size} = \frac{224}{4} = 56 \quad\Rightarrow\quad 56 \times 56 = 3{,}136 \text{ patches}
$$

For patch $i$ (there are 3,136 of them):

$$
\mathbf{z}_i = W_{patch} \cdot \text{flatten}(\text{patch}_i) + b_{patch}, \qquad
W_{patch} \in \mathbb{R}^{96 \times 48},\ \ \mathbf{z}_i \in \mathbb{R}^{96}
$$

The *same* $W_{patch}$ is reused for all 3,136 patches — this is what "shared" means. In
code this entire operation — slice into non-overlapping patches, flatten, linear-project —
is done in a single line by exploiting a trick: a convolution whose **stride equals its
kernel size** can never let patches overlap, so `nn.Conv2d(3, 96, kernel_size=4, stride=4)`
computes exactly the equation above without ever writing an explicit "reshape into patches"
step.

After the convolution, the output is flattened from a 2-D grid into a 1-D sequence and passed
through LayerNorm:

$$
\text{PatchEmbed}(X) = \text{LN}\big(\text{flatten}(\text{Conv2d}_{4\times4,\,s=4}(X))\big)
$$

**Shape:** `(B, 3, 224, 224)` → `(B, 3136, 96)`. Each of the 3,136 tokens is a 96-number
summary of one small patch of the original image.

**Why patches, not raw pixels?** Self-attention (Step 4) compares *every* token to *every
other* token, which costs $O(N^2)$. At the pixel level $N = 224 \times 224 = 50{,}176$, and a
$50{,}176^2$ attention matrix per head is computationally impossible. Patching first cuts $N$
down to 3,136 before attention even starts.

---

### Step 3 — What "attention" means, in plain terms

Before going further, here's the core idea reused everywhere in this model (both encoder and
decoder): **attention lets every token look at other tokens and pull in a weighted average of
their information, where the weights are learned to depend on how relevant each other token
is.**

Concretely, every token produces three separate vectors from itself via three learned linear
layers:

- a **Query** $Q$ — "what am I looking for?"
- a **Key** $K$ — "what do I contain, that others might want?"
- a **Value** $V$ — "what information do I actually offer, if picked?"

The relevance of token $j$ to token $i$ is the dot product $Q_i \cdot K_j$ (large if the
vectors point in similar directions). These relevance scores are scaled and passed through a
softmax so they form a probability distribution that sums to 1 across all tokens $j$, and that
distribution is used to compute a weighted sum of every $V_j$:

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

The $\sqrt{d_k}$ divisor exists purely to keep the dot products from growing too large as the
vector dimension $d_k$ grows (large dot products push softmax into a near-one-hot regime with
vanishing gradients).

**Multi-head** attention just runs several independent copies of this ($Q,K,V$ each with their
own smaller weight matrices) in parallel, then concatenates the results — this lets different
heads specialize in different kinds of relationships (e.g., one head might learn to track
"nearby anatomical structure," another "similar tissue texture").

$$
\text{MultiHead}(Q,K,V) = \big[\,\text{head}_1 \| \text{head}_2 \| \cdots \| \text{head}_h\,\big] \, W_O
$$

Everything below is a variation on this one equation.

---

### Step 4 — Windows: why Swin doesn't run attention over all 3,136 tokens at once

Running full attention over $N = 3{,}136$ tokens means a $3{,}136 \times 3{,}136 \approx 9.8$
million-entry attention matrix *per head, per image* — expensive, and it only gets worse at
higher resolution. Swin's central trick: **only let tokens attend within a small local
window**, not across the whole image.

The $56 \times 56$ token grid is cut into non-overlapping **7×7** windows
(`window_size = 7`):

$$
\text{num. windows} = \frac{56}{7} \times \frac{56}{7} = 8 \times 8 = 64 \text{ windows}, \qquad
\text{each window has } 7 \times 7 = 49 \text{ tokens}
$$

`window_partition()` (`swin_model.py:32-36`) performs this reshape:
`(B, 56, 56, 96)` → `(B·64, 7, 7, 96)` → flattened to `(B·64, 49, 96)`.

Cost comparison:

$$
\text{full attention: } O(N^2) \qquad\longrightarrow\qquad \text{windowed attention: } O(N \cdot M^2),\ M=7
$$

This turns a quadratic cost into a *linear* one in the number of tokens — the change that
makes Swin practical at high resolution.

---

### Step 5 — Attention inside one window, with a *learned relative position bias* (`WindowAttention`, `swin_model.py:52-106`)

Inside each 49-token window, ordinary multi-head self-attention (Step 3) is applied — with one
important addition. A plain transformer normally adds a positional encoding to every token so
the network knows token order. Swin does something different: instead of an absolute position
signal, it adds a **learned bias term to the attention score itself**, based only on the
*relative* offset between the two tokens being compared, not their absolute location in the
image.

**Why relative, not absolute?** A window's content should be interpreted the same way no
matter where in the image that window happens to sit — "the token directly above me" should
mean the same thing whether the window is in the top-left or bottom-right of the image. A
relative bias is naturally translation-invariant; an absolute one is not.

For a $7\times7$ window, two positions can differ by at most $\pm 6$ along each axis, giving
$2 \times 7 - 1 = 13$ possible offsets per axis, and $13 \times 13 = 169$ possible
$(\Delta h, \Delta w)$ pairs overall. A table of 169 learned scalars (one per attention head)
stores a bias for every possible offset:

$$
\texttt{relative\_position\_bias\_table} \in \mathbb{R}^{169 \times \text{heads}}
$$

A precomputed index array (`relative_position_index`, `swin_model.py:68-78`) maps every one of
the $49 \times 49 = 2{,}401$ position pairs inside a window to the correct row of that table:

$$
B[i,j,h] = \text{table}\big[\,\text{index}(\Delta h_{ij}, \Delta w_{ij}),\ h\,\big]
$$

The full windowed attention equation becomes:

$$
\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + B\right) V,
\qquad Q,K,V \in \mathbb{R}^{49 \times d_k} \text{ per window, per head}
$$

At stage 1: $C = 96$, `num_heads = 3`, so $d_k = 96 / 3 = 32$ per head. The **same** 169-entry
bias table is reused for every one of the 64 windows in the image — it is a property of
*relative offset*, not of any specific window's location.

**Shape inside a window:** `(B·64, 49, 96)` in → `(B·64, 49, 96)` out (unchanged shape;
content is now attention-mixed).

---

### Step 6 — The blind spot, and shifted-window attention (SW-MSA) (`swin_model.py:150-171, 173-201`)

Fixed windows never talk to each other — a token at the right edge of one window has no way
to attend to a token one pixel away in the neighboring window. Left uncorrected, information
could never flow across window boundaries no matter how many layers are stacked.

**The fix:** alternate between two kinds of block down each stage. Even-indexed blocks use
ordinary windows as in Step 5 (**W-MSA**). Odd-indexed blocks first *cyclically shift* the
entire token grid by `shift_size = window_size // 2 = 3` positions using `torch.roll`, **then**
partition into windows and run attention, then roll back afterward (**SW-MSA**):

$$
\tilde{X} = \text{roll}(X,\ \text{shift}=(-3,-3))
$$

Because the grid wrapped around, some of the new 7×7 windows now contain patches that were
*not* spatially adjacent before the shift (image content from opposite edges of the grid gets
mixed together purely as a side effect of the wrap-around). An attention mask is precomputed
once per shifted block (`_build_shift_mask`) that labels every token with a region id (9
regions, from the 3×3 combination of `h_slices` × `w_slices`), then for every pair of tokens
inside a window:

$$
\text{attn\_mask}_{ij} =
\begin{cases}
0 & \text{if region}(i) = \text{region}(j) \text{ (were truly adjacent before the shift)} \\
-100.0 & \text{otherwise (accidentally grouped by the wrap-around)}
\end{cases}
$$

This mask is *added* to the attention scores before the softmax in Step 5's equation, so the
softmax assigns those accidental pairs essentially zero probability without ever needing a
separate branch of code.

**Why $-100.0$ and not $-\infty$?** Mathematically $-\infty$ is the "correct" value — it drives
the softmax weight to exactly zero. But a softmax over a row that contains $-\infty$ can
produce `NaN` on PyTorch's Apple-Silicon (MPS) backend, which matters here because `train.py`
targets `mps` when no CUDA GPU is available. A large finite negative number reaches the same
practical outcome (a softmax weight of essentially $e^{-100} \approx 0$) while sidestepping
that numerical edge case. The decoder's causal mask (Part B) was originally written with a
literal `-inf` and has since been changed to the same `-100.0` convention for consistency.

Running one **W-MSA block** followed by one **SW-MSA block** lets every token exchange
information with all of its immediate neighbors, even across the original window boundaries,
over the course of just two blocks — repeated across the depth of a stage, this lets
information propagate arbitrarily far.

---

### Step 7 — The feed-forward sub-layer (MLP) (`Mlp`, `swin_model.py:116-127`)

After attention mixes information *between* tokens, a small two-layer network processes each
token *independently*, giving the model extra capacity to transform the representation:

$$
\text{MLP}(x) = W_2 \cdot \text{GELU}(W_1 x + b_1) + b_2, \qquad
W_1 \in \mathbb{R}^{4C \times C},\ W_2 \in \mathbb{R}^{C \times 4C}
$$

The hidden dimension is $4C$ (e.g. $4 \times 96 = 384$ at stage 1) — a standard transformer
convention of expanding then contracting. GELU is a smooth, differentiable activation function
(a softened version of ReLU) applied elementwise.

**Shape:** unchanged — `(·, 96)` in → `(·, 96)` out at stage 1.

---

### Step 8 — Assembling one Swin block: LayerNorm + residual connections (`SwinBlock.forward`, `swin_model.py:173-201`)

`LayerNorm` normalizes each token vector independently to zero mean and unit variance (then
applies a learned per-channel scale $\gamma$ and shift $\beta$):

$$
\text{LN}(x) = \gamma \odot \frac{x - \mathbb{E}[x]}{\sqrt{\text{Var}[x] + \varepsilon}} + \beta
$$

This keeps activations in a numerically stable range as they pass through many stacked layers.
A **residual connection** (adding the *input* of a sub-layer back to its *output*) is applied
around both the attention and the MLP sub-layer, which gives gradients a direct path backward
through the network during training (critical for training deep networks at all) and lets each
sub-layer learn a *correction* to its input rather than having to reconstruct it from scratch.

This implementation uses **Pre-LN** — LayerNorm is applied *before* each sub-layer, not after:

$$
x \leftarrow x + \text{(Shifted-)WindowAttention}\big(\text{LN}_1(x)\big)
$$
$$
x \leftarrow x + \text{MLP}\big(\text{LN}_2(x)\big)
$$

One full `SwinBlock` = these two residual sub-layers, in order. A `BasicLayer`
(`swin_model.py:234-249`) stacks several `SwinBlock`s (alternating W-MSA / SW-MSA as in Step
6) to form one *stage*.

---

### Step 9 — Patch merging: building a feature pyramid (`PatchMerging`, `swin_model.py:209-228`)

After a stage finishes processing tokens at one resolution, `PatchMerging` shrinks the grid and
widens the channel dimension — Swin's equivalent of a CNN's pooling/downsampling layer, except
implemented as a *learned* linear projection rather than a fixed rule (so nothing is discarded
by a hard-coded max/average — the network decides what to keep).

Every $2\times2$ neighborhood of tokens (each of width $C$) is concatenated along the channel
axis into one token of width $4C$:

$$
x_{\text{cat}} = \big[\,x_{0,0} \,\|\, x_{1,0} \,\|\, x_{0,1} \,\|\, x_{1,1}\,\big] \in \mathbb{R}^{4C}
$$

then normalized and linearly projected back down to $2C$:

$$
x' = W_{merge} \cdot \text{LN}(x_{\text{cat}}), \qquad W_{merge} \in \mathbb{R}^{2C \times 4C} \text{ (no bias)}
$$

$$
(H, W, C) \ \longrightarrow\ \left(\frac{H}{2}, \frac{W}{2}, 2C\right)
$$

Note the compression this represents: area drops by $4\times$ but channel width only doubles,
so overall token *capacity* is halved at every merge:

$$
\frac{H}{2}\cdot\frac{W}{2}\cdot 2C = \frac{H\,W\,C}{2}
$$

---

### Step 10 — Repeating the pattern across four hierarchical stages (`SwinEncoder`, `swin_model.py:255-280`)

The encoder is nothing more than **Steps 5–9 repeated four times**, with patch merging between
consecutive stages (but *not* after the last one), each stage using more attention heads as the
channel width grows:

| Stage | Resolution ($H\times W$) | Tokens $N$ | Channels $C$ | Depth (blocks) | Heads |
|---|---|---|---|---|---|
| Input (after `PatchEmbed`) | $56\times56$ | 3,136 | 96 | — | — |
| **Stage 1** | $56\times56$ | 3,136 | 96 | 2 | 3 |
| → merge → | $28\times28$ | 784 | 192 | — | — |
| **Stage 2** | $28\times28$ | 784 | 192 | 2 | 6 |
| → merge → | $14\times14$ | 196 | 384 | — | — |
| **Stage 3** | $14\times14$ | 196 | 384 | 6 | 12 |
| → merge → | $7\times7$ | 49 | 768 | — | — |
| **Stage 4** | $7\times7$ | 49 | 768 | 2 | 24 |

These are exactly the published **Swin-T ("tiny")** defaults: `depths=(2,2,6,2)`,
`heads=(3,6,12,24)`, `window_size=7` throughout, `embed_dim=96`. Note that at stage 4 the
window size (7) equals the entire grid size (7×7), so "windowed" attention there is
automatically full attention over the whole 7×7 map — there is nothing left to shift or miss.

Each stage internally repeats Step 8's block structure `depth` times (alternating W-MSA/SW-MSA
per Step 6), and every block within a stage uses the *same* channel width and head count shown
in the table.

---

### Step 11 — The finished visual representation

After stage 4, one final LayerNorm is applied, producing the encoder's output — the "memory"
that the caption decoder will read from:

$$
\text{memory} = \text{LN}\big(\text{Stage}_4(\cdots\text{Stage}_1(\text{PatchEmbed}(X))\cdots)\big) \in \mathbb{R}^{B \times 49 \times 768}
$$

**Shape trace, start to finish:**
`(B,3,224,224)` → `(B,3136,96)` → `(B,784,192)` → `(B,196,384)` → `(B,49,768)`.

These 49 vectors (one per $32\times32$-pixel region of the original image, since
$224/7=32$) play the same conceptual role that a CNN backbone's final feature map plays in a
classic "Show and Tell"-style captioner — except every one of them was built purely from
attention and merging, never a convolution beyond the very first patch-embedding step, and
every one of them has, by stage 4, aggregated information from across the *entire* image, just
at progressively coarser resolution as depth increases.

---

## PART B — The Caption Side: text → vocabulary → vectors, and back to text

This part mirrors the diagrams you attached (D1–D6, E1–E3). It's threaded with a worked
example using the toy caption *"Axial CT scan showing a large pleural effusion in the right
lung."*

### Step 12 — Captions are not "in" the image; they live in a separate CSV (D1)

ROCOv2 ships `train_captions.csv` / `train_images/` as two *separate* parallel resources tied
together only by an image ID:

| ID | Caption |
|---|---|
| ROCOv2\_2023\_train\_000001 | Axial CT scan showing a large pleural effusion in the right lung. |
| ROCOv2\_2023\_train\_000002 | Chest X-ray showing consolidation ... |

`dataset.py`'s `ROCODataset.__getitem__` is what actually pairs them at training time: it
opens the `.jpg` for a given `ID` (→ a `(3,224,224)` tensor, Part A) and independently looks up
that row's `Caption` string, encodes it (Step 13), and returns `(image_tensor, caption_ids)` as
one training example. The image tensor and the caption tensor are computed by two completely
separate pipelines that only meet inside the model at cross-attention (Step 18).

---

### Step 13 — Tokenizing text into words (`Vocab.tokenize`, `vocab.py:34-39`) (D2)

A caption string is turned into a list of word tokens by a plain regex tokenizer — no
subword/BPE tokenization, no external NLP library:

1. lowercase the string,
2. replace every character that is **not** `a`–`z`, `0`–`9`, or whitespace with a space
   (this strips all punctuation — and also destroys numbers-with-units and hyphenated terms,
   e.g. "X-ray" becomes two separate tokens "x" and "ray"),

3. collapse any run of whitespace to a single space and split on it.

$$
\text{"Axial CT scan showing a large pleural effusion in the right lung."}
$$
$$
\downarrow
$$
$$
[\texttt{axial, ct, scan, showing, a, large, pleural, effusion, in, the, right, lung}]
$$

12 word tokens from this one caption.

---

### Step 14 — Building the vocabulary, once, over the whole training set (`Vocab.__init__`, `vocab.py:12-32`, `build_vocab_from_csv`) (D3)

Before training starts, every caption in the training split (up to `TRAIN_SAMPLES`, i.e.
15,000 captions in the real run) is tokenized and a frequency count is built across *all* of
them with a `Counter`. Only words seen **at least `min_freq=2` times** in the whole training
set are kept as real vocabulary entries — anything rarer is later replaced by a single
`<unk>` ("unknown") token at encode time. This keeps the vocabulary a manageable, well-trained
size instead of ballooning with one-off typos or rare terms.

Four **special tokens** are reserved first, at fixed indices:

| id | token | purpose |
|---|---|---|
| 0 | `<pad>` | filler so every caption in a batch has the same length |
| 1 | `<sos>` | "start of sentence" — the seed the decoder is given before it has written anything |
| 2 | `<eos>` | "end of sentence" — the model's learned stop signal |
| 3 | `<unk>` | stand-in for any word rarer than `min_freq` |

Then every surviving word is appended, **sorted alphabetically**, giving a fixed, reproducible
`id ↔ word` mapping (`itos`, "index to string") and its reverse (`stoi`, "string to index").
On the real 15,000-sample training run this produces a **7,886-word vocabulary**; the toy
example below uses a 19-word vocabulary for illustration:

```
0 <pad>   4 a       8 effusion  12 pleural  16 showing
1 <sos>   5 axial   9 large     13 ray      17 the
2 <eos>   6 chest  10 lung     14 right    18 x
3 <unk>   7 ct     11 of       15 scan
```

---

### Step 15 — Encoding a caption into a fixed-length integer vector (`Vocab.encode`, `vocab.py:41-49`) (D4)

Given `max_len = 40`, a caption is turned into an id sequence: `<sos>`, then up to
`max_len - 2 = 38` word ids (any word not in the vocabulary becomes `<unk>`'s id), then
`<eos>`, then `<pad>` repeated out to exactly 40 positions:

$$
\text{ids} = \big[\,\text{sos}, w_1, w_2, \ldots, w_k, \text{eos}, \underbrace{\text{pad}, \ldots, \text{pad}}_{40-k-2}\,\big], \qquad \text{len} = 40
$$

Worked example (first 18 of 40 slots shown, "in" replaced by `<unk>` since it appeared only
once in this toy corpus and `min_freq=2`):

| t=0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13–39 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `<sos>`(1) | axial(5) | ct(7) | scan(15) | showing(16) | a(4) | large(9) | pleural(12) | effusion(8) | `<unk>`(3) | the(17) | right(14) | lung(10) | `<eos>`(2), then `<pad>`(0)×26 |

**Why each special token exists:**

- `<sos>` is the very first thing fed to the decoder at generation time, before it has produced
  any words of its own — without a fixed starting point, generation could never begin.

- `<eos>` is what the model must learn to output when the sentence is complete; without it,
  greedy generation would never know when to stop and would run to `max_len` every time.

- `<pad>` exists purely so every caption in a batch can be stacked into one rectangular tensor
  despite having different real lengths; it is explicitly excluded from the loss via
  `ignore_index=vocab.pad_id` so the model is never penalized (or rewarded) for predicting
  filler.

- `<unk>` is a real, visible information loss — any sufficiently rare clinical term the
  tokenizer sees fewer than twice in training becomes indistinguishable from every other rare
  term. This is a genuine limitation of a word-level (not subword) vocabulary on long-tailed
  medical text (see Step 21).

---

### Step 16 — From integer ids to vectors: the embedding table (`CaptionDecoder.embed`, `decoder_model.py:88`) (D5)

An integer id by itself carries no learnable meaning — id 12 for "pleural" being numerically
close to id 13 for "ray" is a coincidence of alphabetical sorting, not a semantic relationship.
`nn.Embedding(vocab_size, 768)` is simply a lookup table: a matrix
$E \in \mathbb{R}^{\text{vocab\_size}\times 768}$ whose row $E_{[v]}$ is the learned 768-dimensional
vector for word id $v$. Looking up an id is just indexing that row:

$$
\mathbf{e}(v) = E_{[v]}, \qquad \mathbf{e}(12) = E_{[12]} = \text{"pleural"'s 768-number vector, learned by backprop}
$$

Because this embedding matrix is *trained end-to-end together with the rest of the model* (not
pretrained), the network is free to place semantically or clinically related words near each
other in this 768-dimensional space purely as a side effect of minimizing the captioning loss
— nothing forces this, but it's a common emergent behavior of trained embedding tables.

---

### Step 17 — Giving the decoder a sense of word order: sinusoidal positional encoding (`PositionalEncoding`, `decoder_model.py:7-18`) (D5)

Attention (Step 3) has no inherent notion of sequence order — swap two tokens and, absent
extra information, the attention computation treats them identically. The encoder handled this
with a *relative*, learned bias local to each attention window (Step 5). The decoder instead
adds a **fixed** (not learned), *absolute*, sinusoidal signal directly to each word's embedding,
following the original 2017 Transformer design:

$$
PE_{(pos,\,2i)} = \sin\!\left(\frac{pos}{10000^{2i/d}}\right), \qquad
PE_{(pos,\,2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d}}\right), \qquad d=768
$$

Every even dimension gets a sine, every odd dimension a cosine, at a frequency that varies
smoothly across the 768 dimensions — giving every position $0,1,2,\ldots$ a unique, fixed
fingerprint that the network can learn to read as "how far into the sentence am I." The same
fixed table is reused for every caption:

$$
x = \text{Embedding}(\text{tgt}) + PE[:, :T]
$$

**A genuine asymmetry worth noting:** the image side encodes position *relatively* and
*learns* it (Step 5); the caption side encodes position *absolutely* and *fixes* it
mathematically (this step). Neither is wrong — patches inside a local window benefit from
translation-invariant relative position, while a sentence benefits from knowing its absolute
position from the start — but the two halves of this model do not use a unified positional
scheme, which is worth stating plainly rather than glossing over.

**Shape after this step:** caption ids `(B, 39)` → embedded + positioned `(B, 39, 768)`
(39, not 40, because of the shift described next).

---

### Step 18 — Teacher forcing: how the model is trained to predict "the next word" (`SwinCaptioningModel.forward`, `caption_model.py`; target construction in `train.py`) (D6)

During training the decoder is **never** allowed to see its own guesses — it is always fed the
true ground-truth caption, one position shifted, so that every position simultaneously learns
"given everything correct up to here, predict the next real word," regardless of how good or
bad the model's own predictions would currently be. This lets an entire 39-token caption be
trained in **one parallel forward pass**, rather than 39 sequential steps.

Concretely, the 40-length id sequence from Step 15 is split into two overlapping halves:

```
captions[:, :-1]   → decoder INPUT   (positions 0..38, i.e. <sos> ... second-to-last word)
captions[:, 1:]    → target          (positions 1..39, i.e. first real word ... <eos>)
```

| decoder INPUT (t=0..7 shown) | `<sos>` | axial | ct | scan | showing | a | large | pleural |
|---|---|---|---|---|---|---|---|---|
| **TARGET** (same column) | axial | ct | scan | showing | a | large | pleural | effusion |

At input position $t$, the model has seen tokens $0 \ldots t$ and must predict the token at
$t+1$. A **causal mask** enforces that position $t$ is mathematically forbidden from looking
at position $t+1$ or later (Step 19) — without this, the model could simply "cheat" by copying
the next token straight out of its own input, since the true next word is sitting right there
in the shifted sequence, and it would learn nothing useful for actual generation.

---

### Step 19 — Masked self-attention: looking only at words already written (`MultiHeadAttention` + causal mask, `decoder_model.py:25-51, 96-105`)

The first sub-layer of each decoder block is exactly the multi-head attention of Step 3, with
$Q, K, V$ all computed from the caption tokens themselves ("self"-attention), **plus** a causal
mask that blocks any position from attending to a later one:

$$
\text{mask}[i,j] = \begin{cases} 1 & j \le i \\ 0 & j > i \end{cases}
\qquad\qquad
\text{scores} = \frac{QK^\top}{\sqrt{d_k}}, \quad \text{scores} \leftarrow \text{scores.masked\_fill}(\text{mask}=0,\ -100.0)
$$

`torch.tril` (lower-triangular ones) builds exactly this mask in one line
(`decoder_model.py:100`). After the softmax, position $i$'s attention weights for every $j>i$
are effectively zero, so the output at position $i$ is a weighted combination *only* of tokens
$0,\ldots,i$ — precisely the "predict the next word using only what's been written so far"
constraint the task requires. This is the same $-100.0$ masking convention used for the shifted
windows in Step 6, now applied for a different reason (causality instead of accidental
wrap-around neighbors).

$d_{model}=768$, `num_heads=8` here, so $d_k = 768/8 = 96$ per head.

---

### Step 20 — Cross-attention: the one place image and text actually meet (`DecoderLayer.forward`, `decoder_model.py:74-78`) (E1)

Everything up to this point in Part B has processed the caption in complete isolation from the
image — no visual information has entered the language stream yet. **Cross-attention is the
single bridge between the two halves of the model.** It reuses the exact same attention
equation from Step 3, but now the three roles are filled from two *different* sources:

$$
Q \leftarrow \text{caption tokens} \in \mathbb{R}^{39 \times 768} \qquad\qquad K, V \leftarrow \text{encoder memory} \in \mathbb{R}^{49 \times 768}
$$

$$
\text{CrossAttn}(x, \text{memory}) = \text{softmax}\!\left(\frac{Q_{\text{caption}}\,K_{\text{memory}}^\top}{\sqrt{d_k}}\right) V_{\text{memory}}
$$

The resulting attention matrix has shape $39 \times 49$: **every caption position asks a
question of every one of the 49 image regions**, and the softmax over each row tells that word
position which image regions are relevant to what it's about to predict. The image supplies
the keys and values (the *content*); the caption supplies the query (the *question*). No causal
mask is used here — a word is always allowed to look at the *entire* image, at every layer,
since nothing about "seeing the whole picture" should be restricted the way "seeing future
words" is.

---

### Step 21 — Completing one decoder layer, and stacking six of them (`DecoderLayer`, `decoder_model.py:61-78`)

One full `DecoderLayer` chains three sub-layers, each wrapped in a residual connection
**followed by** LayerNorm (**Post-LN** — note this is the opposite order from the encoder's
Pre-LN in Step 8; the decoder follows the original 2017 Transformer paper's convention instead
of Swin's):

$$
x \leftarrow \text{LN}_1\big(x + \text{Dropout}(\text{MaskedSelfAttn}(x))\big)
$$
$$
x \leftarrow \text{LN}_2\big(x + \text{Dropout}(\text{CrossAttn}(x, \text{memory}))\big)
$$
$$
x \leftarrow \text{LN}_3\big(x + \text{Dropout}(\text{FFN}(x))\big)
$$

where `FFN` is the same style of two-layer GELU feed-forward block as Step 7, here expanding
$768 \to 3{,}072 \to 768$ (`ff_dim = 4 \times d_{model}`). Six of these layers are stacked in
sequence (`num_layers=6`); the encoder's memory (49×768) is fed unchanged into the
cross-attention of *every* one of the six layers.

---

### Step 22 — Turning the final vectors into word probabilities (`CaptionDecoder.fc_out`, `decoder_model.py:93,105`)

After the sixth decoder layer, every position's 768-dimensional vector is projected by one
final linear layer up to the size of the vocabulary:

$$
\text{logits} = W_{out}\,x + b_{out} \in \mathbb{R}^{B \times 39 \times \text{vocab\_size}}, \qquad W_{out} \in \mathbb{R}^{\text{vocab\_size}\times 768}
$$

Each 768-vector becomes one raw score per vocabulary word — position $t$'s logits describe how
strongly the model favors each of the ~7,886 possible words as the prediction for position
$t+1$. These are *logits*, not probabilities yet; softmax (implicit inside the loss function
below, and explicit at generation time) turns them into a proper probability distribution.

---

### Step 23 — The training loss: cross-entropy with label smoothing (`train.py`)

For every non-padding position, the model's predicted distribution is compared against the
true next word using cross-entropy. With **label smoothing** ($\varepsilon = 0.1$), the "true"
target distribution is softened: instead of putting 100% of the probability mass on the single
correct word, a small amount is spread over every other word in the vocabulary:

$$
y_v^{LS} = \begin{cases} 1-\varepsilon & v = \text{target word} \\ \dfrac{\varepsilon}{V-1} & \text{otherwise} \end{cases}
\qquad\qquad
\mathcal{L} = -\sum_v y_v^{LS}\, \log p_v, \qquad p = \text{softmax}(\text{logits})
$$

Padding positions are excluded entirely via `ignore_index=vocab.pad_id` — the model is never
scored on the meaningless `<pad>` filler. Label smoothing keeps the model from becoming
overconfident (driving logits to extreme magnitude chasing an unreachable "perfect" one-hot
target), which in practice tends to produce better-calibrated, more stable training.

This loss is what `optimizer.step()` (AdamW, with the weight-decay parameter grouping and
cosine warmup/decay schedule described in `methodology.tex`) minimizes over 60 epochs.

---

## PART C — Turning a trained model back into a sentence

### Step 24 — Training vs. inference: parallel vs. sequential (E2)

**Training (teacher forcing, Step 18):** the full ground-truth caption is available up front,
so all 39 next-word predictions for one caption can be computed in a **single forward pass** —
the causal mask alone is what stops each position from "seeing" the answer to a later position.

**Inference (`SwinCaptioningModel.generate`, `caption_model.py:33-63`):** at generation time
there is no ground-truth caption to feed in — the model must build the sentence one word at a
time, feeding each new guess back in as input for the next step:

```
ids = [<sos>]                                    → decoder(ids, memory) → predict "axial"
ids = [<sos>, axial]                              → decoder(ids, memory) → predict "ct"
ids = [<sos>, axial, ct]                          → decoder(ids, memory) → predict "scan"
...                                                                          ...
ids = [<sos>, axial, ..., lung]                   → decoder(ids, memory) → predict <eos> → STOP
```

At each step, only the **last** position's logits are used (everything before it was already
finalized in a previous step); `argmax` picks the single highest-scoring next word (**greedy**
decoding — no beam search, no sampling).

---

### Step 25 — Two anti-repetition guards bolted onto greedy decoding (`caption_model.py:44-58`)

Plain greedy decoding is prone to a specific failure: once the model starts repeating a word or
short phrase, repeating it again often looks like the highest-probability choice at every
subsequent step too, so the model can get stuck in a loop. Two heuristics guard against this,
applied directly to the logits before `argmax`, each step:

1. **Never repeat the immediately preceding token:**
   $$
   \text{logits}[\,\cdot,\ \text{prev\_token}\,] \leftarrow -\infty
   $$

2. **Never complete a trigram already seen earlier in this caption.** Every 3-token window
   already generated is recorded; if the last two tokens generated match the first two tokens
   of some earlier trigram, whatever third word completed that trigram before is banned this
   time:
   $$
   \text{seen} = \{(w_i, w_{i+1}, w_{i+2})\}_{i}, \qquad
   \text{banned} = \{\,w_{i+2} : (w_i,w_{i+1}) = (\text{ids}_{-2}, \text{ids}_{-1})\,\}
   $$
   $$
   \text{logits}[\,\cdot,\ w\,] \leftarrow -\infty \quad \forall\, w \in \text{banned}
   $$

Neither guard changes the model's weights or what it "knows" — they only restrict which
choices greedy decoding is allowed to make at each step. They exist as a direct, practical
patch for a real, previously observed failure mode: `FINDINGS.md` documents a checkpoint that,
without these guards, converged to emitting the same generic caption for nearly every input
image (documented as "encoder representation collapse"). Generation stops the moment `<eos>` is
produced, or after `max_len=40` steps if it never is.

---

### Step 26 — From ids back to a readable string (`Vocab.decode`, `vocab.py:51-59`)

The reverse of Step 15: walk the generated id sequence, stop at (and drop) the first `<eos>`,
skip `<sos>`/`<pad>` if present, and look up every remaining id in `itos` to rebuild the word
sequence, joined with spaces:

$$
\text{ids} = [1,\, 5,\, 7,\, 15,\, 16,\, 4,\, 9,\, 12,\, 8,\, 2,\, 0,\, 0,\, \ldots] \quad\longrightarrow\quad \text{"axial ct scan showing a large pleural effusion"}
$$

---

## PART D — What's specifically harder about *biomedical* image captioning (E3)

The general pipeline above (Parts A–C) is a domain-agnostic image-captioning architecture. A
few properties of radiology captions specifically make this a harder setting than natural-image
captioning, worth calling out for anyone reading this as background before the results section:

- **Long-tailed, technical vocabulary.** Terms like "pneumoperitoneum" or "osteolytic" may
  appear only a handful of times across the whole training set. The word-level tokenizer with
  `min_freq=2` (Step 14) turns most of these into `<unk>`, discarding exactly the information
  most clinically important to distinguish one report from another. A subword tokenizer
  (BPE/WordPiece) or a domain-pretrained tokenizer would preserve more of this vocabulary.

- **No pretrained biomedical text representations.** The embedding table (Step 16) is learned
  entirely from scratch on however many thousand captions are used for training; models that
  initialize from BioBERT/PubMedBERT/SciBERT/BioGPT embeddings start with a much stronger prior
  over clinical language.

- **No concept/CUI supervision.** ROCOv2 ships a parallel concepts CSV (UMLS Concept Unique
  Identifiers per image) that this project's scripts load but never use (see `README.md`'s
  dataset-structure table) — many published medical captioning systems add a multi-label
  concept-classification auxiliary loss alongside the caption loss, which this model does not.

- **Captions are long, formulaic, and templated.** Clinical phrasing repeats stock words
  ("shows", "demonstrates", "axial view") across very different underlying findings, which
  means n-gram-overlap metrics like BLEU can score a caption highly even when the described
  diagnosis is wrong — this is exactly the failure `collapse_check.csv` and the "% distinct
  captions" check in `eval.py` were built to catch, since surface-level metrics alone would
  have missed it.

- **Evaluation ideally needs clinical, not just linguistic, metrics.** BLEU/ROUGE-L/CIDEr-D
  (all implemented in `metrics.py`) measure textual overlap with a reference caption, not
  factual/diagnostic correctness. Metrics such as CheXbert-F1 or RadGraph-F1, which score
  whether the generated text asserts the same clinical findings as the reference (not just the
  same words), are a natural next addition for a clinically meaningful evaluation.

---

## Summary: the full pipeline in one table

| # | Step | Operation | Shape in → out |
|---|---|---|---|
| 1 | Input | resize + normalize | `(B,3,224,224)` → same |
| 2 | Patch embed | Conv2d(k=4,s=4) + flatten + LN | `(B,3,224,224)` → `(B,3136,96)` |
| 4–6 | Window / shifted-window attn | local MHSA + relative pos. bias, alternating shift | `(B,3136,96)` → same |
| 7–8 | MLP + LN + residual | per-token feed-forward, Pre-LN | `(B,3136,96)` → same |
| 9 | Patch merge (×3, between stages) | concat 2×2 → LN → Linear(4C→2C) | halves tokens, doubles channels |
| 10 | Repeat stages 2–4 | same block structure, growing $C$, shrinking grid | `(B,3136,96)`→`(B,784,192)`→`(B,196,384)`→`(B,49,768)` |
| 11 | Final encoder output | LayerNorm | `(B,49,768)` = "memory" |
| 12–15 | Caption → ids | regex tokenize → vocab lookup → pad/truncate | text → `(B,40)` |
| 16–17 | Ids → vectors | embedding lookup + sinusoidal position | `(B,39)` → `(B,39,768)` |
| 18 | Teacher forcing | shift input/target by one position | `(B,40)` → two `(B,39)` |
| 19 | Masked self-attention | causal MHSA over caption only | `(B,39,768)` → same |
| 20 | Cross-attention | caption queries, image keys/values | `(B,39,768)` + `(B,49,768)` → `(B,39,768)` |
| 21 | Full decoder layer ×6 | self-attn → cross-attn → FFN, Post-LN | `(B,39,768)` → same |
| 22 | Output projection | Linear(768→vocab) | `(B,39,768)` → `(B,39,V)` logits |
| 23 | Loss (training only) | label-smoothed cross-entropy, pad ignored | `(B,39,V)` → scalar |
| 24–25 | Greedy generation (inference only) | autoregressive argmax + repeat-blocking | 1 token/step, up to 40 steps |
| 26 | Decode | ids → words, stop at `<eos>` | `(B,≤40)` → string |

Every arrow in this table is one of the equations above — there is no hidden step, no external
library call, and no pretrained component anywhere in the chain.
