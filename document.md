# Swin Encoder — Full Worked Derivation

This document formalizes, step by step, the exact derivation you worked through by hand:
image → patches → linear projection → windows → attention (scores → softmax → weighted
sum → multi-head concat) → MLP → LayerNorm + residual → patch merging into the next stage.
It uses your own notation ($T_i$ for a token, per-token $Q,K,V$, explicit score matrices,
percentage-style softmax weights) and fills in the a few steps your notes shorthanded, so it
reads as a complete, self-contained proof of how one Swin block works. It picks up exactly
where `swin_captioning_full_walkthrough.md` (Part A) covers the same ground in prose — this
version is the numeric derivation behind those equations.

---

## 1. Overall pipeline

$$
\text{image } 224\times224\times3 \ \text{(RGB)} \quad\longrightarrow\quad \text{should be converted into tokens}
$$

The image is cut into a grid of non-overlapping patches. Each patch spans **$4\times4$ pixels
across all 3 channels**:

$$
\text{patch size} = 4\times4\times3
$$

Since the image is $224\times224$ and each patch is $4\times4$:

$$
\frac{224}{4} = 56 \quad\Rightarrow\quad \text{a } 56 \times 56 \text{ grid of patches (56 rows} \times \text{56 columns)}
$$

$$
56 \times 56 = 3{,}136 \text{ patches total}
$$

$$
224\times224\times3 \ \longrightarrow\ 4\times4 \text{ patches} \ \longrightarrow\ 56\times56 \text{ patch grid} \ \longrightarrow\ 3{,}136 \text{ patches}
$$

---

## 2. Each patch → linear projection

### 2.1 What's inside one patch

A single patch is $4\times4$ pixels × 3 channels:

$$
4 \times 4 \times 3 = 48 \text{ raw pixel values}
$$

$$
\text{patch}_1 \rightarrow 48 \text{ values}, \quad \text{patch}_2 \rightarrow 48 \text{ values}, \quad \ldots, \quad \text{patch}_{3136} \rightarrow 48 \text{ values}
$$

Across all patches that's $3{,}136 \times 48 = 150{,}528$ raw numbers total — still just pixels,
nothing learned yet.

### 2.2 Turning 48 raw values into a 96-dimensional token

Each patch's 48 values are flattened into one vector $x \in \mathbb{R}^{48}$, and a single
**linear transformation** (the same one, reused for every patch) projects it up to 96
dimensions:

$$
y = Wx + b
$$

$$
W \in \mathbb{R}^{96\times48}, \qquad x \in \mathbb{R}^{48\times1} \quad\Rightarrow\quad y \in \mathbb{R}^{96\times1}
$$

Think of $W$ as **96 separate filters**, one per output dimension. Each filter looks at the
same 48 input values but has its own 48 learned weights, and performs a weighted sum:

$$
y_i = \sum_{j=1}^{48} w_{ij}\, x_j + b_i, \qquad i = 1, \ldots, 96
$$

$$
\text{filter}_1 \rightarrow 48 \text{ learned weights}, \quad \text{filter}_2 \rightarrow 48 \text{ learned weights}, \quad \ldots, \quad \text{filter}_{96} \rightarrow 48 \text{ learned weights}
$$

So: **48 raw pixel values in → 96 learned filter responses out**, per patch. This is exactly
what `PatchEmbed`'s `Conv2d(3, 96, kernel_size=4, stride=4)` computes, one patch at a time,
using the same 96 filters everywhere.

### 2.3 How $W$ and $b$ get learned: backprop

$W$ and $b$ start random and are updated by gradient descent, using the gradient of the
training loss with respect to every weight:

$$
\frac{\partial \mathcal{L}}{\partial w}
$$

$$
w_{\text{new}} = w_{\text{old}} - \eta \, \frac{\partial \mathcal{L}}{\partial w}
$$

where $\eta$ is the learning rate. This is the same update rule that trains every weight
matrix in the network — the patch-projection filters, every attention $W_Q/W_K/W_V$, every
MLP layer — the loss (Step 23 in the full walkthrough) is what ultimately drives all of it.

---

## 3. From patches to a token sequence

After the projection in Section 2, every one of the 3,136 patches has become a **96-dimensional
vector**:

$$
3{,}136 \text{ patches} \ \longrightarrow\ 3{,}136 \text{ vectors} \ \longrightarrow\ \text{each vector has 96 dimensions}
$$

$$
X \in \mathbb{R}^{3136 \times 96}
$$

Call the $i$-th one $T_i \in \mathbb{R}^{1\times96}$ — a **token**. There are 3,136 of them,
arranged in the same $56\times56$ grid the patches came from.

---

## 4. Patches → windows

Running attention across all 3,136 tokens at once is too expensive, so the $56\times56$ grid
is cut into non-overlapping **$7\times7$ windows**:

$$
\frac{56}{7} = 8 \quad\Rightarrow\quad 8 \times 8 = 64 \text{ windows}
$$

$$
\text{each window} = 7\times7 = 49 \text{ tokens}
$$

Checking the total is conserved:

$$
\text{before windows} = 3{,}136 \text{ tokens} \qquad\qquad 64 \text{ windows} \times 49 \text{ tokens/window} = 3{,}136 \ \checkmark
$$

Every window is now an independent group of 49 tokens, each one a 96-dimensional vector, e.g.
window 1 contains $T_1, T_2, \ldots, T_{49}$ (a $7\times7$ patch of the original grid):

$$
T_i = [\,0.1,\ 0.2,\ \ldots,\ 0.27\,] \in \mathbb{R}^{1\times96} \quad\text{— represents one small region of the image}
$$

---

## 5. Attention inside one window — full derivation

**Attention is only computed between tokens that share the same window** — $T_i$ and $T_j$
attend to each other only if both are in the same $7\times7$ window.

### 5.1 Stacking one window

For window 1, stack all 49 tokens into a matrix:

$$
X_{\text{window}} = \begin{bmatrix} T_1 \\ T_2 \\ \vdots \\ T_{49} \end{bmatrix} \in \mathbb{R}^{49\times96}
$$

### 5.2 Splitting into heads

With **3 attention heads** at this stage, the 96 dimensions split evenly:

$$
\frac{96}{3} = 32 \text{ dimensions per head}
$$

Every token produces a Query, Key, and Value by three learned linear projections (shown here
for one head; the other two heads have their own separate $W_Q, W_K, W_V$ and run in parallel):

$$
Q_i = T_i W_Q, \qquad K_i = T_i W_K, \qquad V_i = T_i W_V
$$

$$
T_1 \rightarrow Q_1, K_1, V_1 \qquad T_2 \rightarrow Q_2, K_2, V_2 \qquad \cdots \qquad T_{49} \rightarrow Q_{49}, K_{49}, V_{49}
$$

For the whole window at once (one head):

$$
Q \in \mathbb{R}^{49\times32}, \qquad K \in \mathbb{R}^{49\times32}, \qquad V \in \mathbb{R}^{49\times32}
$$

### 5.3 Why attention: one token asking a question

Say token $T_1$ wants to know which other tokens in the window are relevant to it. It offers
its **Query** $Q_1$; every token in the window (itself included) offers a **Key**. The
relevance of token $j$ to token 1 is their dot product:

$$
Q_1 \cdot K_1 = \text{Score}(1,1) \qquad Q_1 \cdot K_2 = \text{Score}(1,2) \qquad \cdots \qquad Q_1 \cdot K_{49} = \text{Score}(1,49)
$$

So $T_1$ ends up with **49 scores**, one against every token in its window (including itself).
The same happens for $T_2, T_3, \ldots, T_{49}$ simultaneously — every token computes a score
against every other token. In matrix form, this is a single matrix multiplication:

$$
Q \in (49\times32), \qquad K^\top \in (32\times49) \quad\Rightarrow\quad QK^\top \in (49\times49)
$$

Row $i$ of $QK^\top$ holds all 49 of token $i$'s relevance scores against the window.

### 5.4 Scaling

Divide by $\sqrt{d}$, where $d=32$ is this head's dimension, to keep the scores from growing
too large before the softmax:

$$
\frac{QK^\top}{\sqrt{d}} = \frac{QK^\top}{\sqrt{32}}
$$

### 5.5 Adding the learned relative position bias

A learned bias $B \in \mathbb{R}^{49\times49}$ (looked up from the 169-entry relative-position
table described in the full walkthrough, Step 5) is added elementwise, giving the final
pre-softmax attention matrix:

$$
A = \frac{QK^\top}{\sqrt{d}} + B
$$

### 5.6 Softmax: turning scores into weights that sum to 1

Take row 1 of $A$ — token 1's raw scores against all 49 tokens in the window:

$$
T_1\text{'s row} = [\,x_1, x_2, \ldots, x_{49}\,] \in \mathbb{R}^{1\times49}
$$

Softmax converts these into positive weights that sum to exactly 1:

$$
\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}
$$

$$
T_1 \rightarrow [\,0.11,\ 0.22,\ 0.33,\ \ldots\,] \in \mathbb{R}^{1\times49}, \qquad \sum = 1
$$

Read as a relevance breakdown: **token 1 relates to token 1 by 11%, to token 2 by 22%, to
token 3 by 33%**, and so on across the rest of the window.

### 5.7 Weighted sum of values

Token 1's new, context-aware representation is a weighted blend of every token's **Value**
vector, using exactly the percentages just computed:

$$
T_1^{\text{new}} = 0.11\, V_1 + 0.22\, V_2 + 0.33\, V_3 + \cdots
$$

With $V \in (49\times32)$ per head and the weight row $(1\times49)$, this weighted sum produces
one $(1\times32)$ output vector for $T_1$ — and the same computation, run for every row of the
softmax matrix at once, produces all 49 output vectors together.

### 5.8 Full matrix form, one head

$$
\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d}} + B\right) V
$$

Shape trace for one window, one head:

$$
\underbrace{Q}_{49\times32}\ \underbrace{K^\top}_{32\times49} = \underbrace{QK^\top}_{49\times49}
\quad\xrightarrow{\text{softmax}}\quad
\underbrace{\text{weights}}_{49\times49}
\quad\times\quad
\underbrace{V}_{49\times32}
\quad=\quad
\underbrace{\text{output}}_{49\times32}
$$

### 5.9 Multi-head: doing this 3 times and concatenating

The exact same computation (Sections 5.2–5.8) runs independently for **head 1, head 2, head
3** — three separate $(49\times32)$ outputs, from three separate, independently-learned
$W_Q, W_K, W_V$ triples:

$$
\text{head}_1 \in (49\times32), \qquad \text{head}_2 \in (49\times32), \qquad \text{head}_3 \in (49\times32)
$$

Concatenating them side by side along the feature dimension restores the original width:

$$
[\,\text{head}_1 \,\|\, \text{head}_2 \,\|\, \text{head}_3\,] \in (49\times96)
$$

### 5.10 What actually changed

$$
T_1^{\text{input}} \in (1\times96) \qquad\longrightarrow\qquad T_1' = T_1^{\text{output}} \in (1\times96)
$$

Same shape in and out — but $T_1'$ is no longer just "patch 1's own pixels." It now carries a
**contextual transformation**: a learned mixture of every other token's information in its
window, weighted by learned relevance. This is true for every one of the 49 tokens
simultaneously — each becomes a blend of its window, not just itself.

---

## 6. MLP (feed-forward)

After attention mixes information *between* tokens, a small two-layer network transforms each
token *independently* (no mixing across tokens here):

$$
96 \xrightarrow{\text{expand} \times 4} 384 \xrightarrow{\text{GELU}} 384 \xrightarrow{\text{project}} 96
$$

$$
96 \times \text{mlp\_ratio} = 96 \times 4 = 384
$$

$$
\text{MLP}(x) = W_2 \cdot \text{GELU}(W_1 x + b_1) + b_2
$$

GELU ("**G**aussian **E**rror **L**inear **U**nit"):

$$
\text{GELU}(x) = x \cdot \Phi(x)
$$

where $\Phi$ is the standard normal **cumulative distribution function** — GELU is a smooth,
differentiable activation, softer than ReLU, that lightly damps small/negative inputs instead
of hard-clipping them to zero.

$$
\text{attention: } T_1 \rightarrow T_1' \qquad\qquad \text{MLP: } T_1' \rightarrow T_1''
$$

$T_1''$ is still $(1\times96)$ — the MLP doesn't change the shape, only refines the content of
each token independently, after attention has already let tokens share information.

---

## 7. LayerNorm + residual — assembling one full block

### 7.1 LayerNorm

Every token vector is independently rescaled to zero mean, unit variance, then given a learned
per-channel scale $\gamma$ and shift $\beta$:

$$
\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \varepsilon}}
$$

$$
y_i = \gamma_i\, \hat{x}_i + \beta_i, \qquad \gamma = \text{scale (learned)}, \ \ \beta = \text{shift (learned)}
$$

### 7.2 The full block, in order (Pre-LN)

$$
x \ \rightarrow\ \text{LN} \ \rightarrow\ \text{W-MSA (Section 5)} \ \rightarrow\ {+}\,x \ \rightarrow\ \text{LN} \ \rightarrow\ \text{MLP (Section 6)} \ \rightarrow\ {+}
$$

Written as two residual (skip) connections:

$$
x' = x + \text{Attention}\big(\text{LN}(x)\big)
$$
$$
x'' = x' + \text{MLP}\big(\text{LN}(x')\big)
$$

The `+x` at each step is what lets each sub-layer learn a *correction* to its input rather
than having to rebuild it from nothing, and gives gradients a direct path backward through the
network during training. This is one complete `SwinBlock` — everything above (Sections 5–7)
happens once per block, and several blocks are stacked per stage.

---

## 8. Patch merging — moving to the next stage

Once a stage's blocks are done, groups of $2\times2$ neighboring tokens (each still 96-dim)
are merged to build the next, coarser stage:

$$
T_1, T_2, T_3, T_4 \ \ (\text{a } 2\times2 \text{ neighborhood, each } 96\text{-dim})
$$

$$
\text{concatenate: } 96 + 96 + 96 + 96 = 384
$$

$$
\text{project (Linear, } 384 \rightarrow 192\text{)}
$$

$$
56\times56\times96 \quad\longrightarrow\quad 28\times28\times192
$$

Resolution halves, channel width doubles — conceptually similar to a CNN's strided
downsampling, except the "pooling rule" here is a **learned linear projection** over
concatenated channels, not a fixed max/average.

This $28\times28\times192$ grid is exactly the input to **stage 2**, which repeats Sections 4–8
in full (new windows, new $Q/K/V$ projections, new relative-position bias table, same $\times4$
MLP ratio) — and stages 3 and 4 repeat the same pattern again, ending at the $7\times7\times768$,
49-token output described in `swin_captioning_full_walkthrough.md`.

---

## Summary: one block's shapes, start to finish

| Step | Operation | Shape |
|---|---|---|
| Input image | — | $224\times224\times3$ |
| Patchify | $4\times4\times3$ patches | $3{,}136$ patches $\times\,48$ values |
| Linear projection | $y=Wx+b$, $W\in\mathbb{R}^{96\times48}$ | $3{,}136 \times 96$ tokens |
| Window partition | $7\times7$ windows | $64$ windows $\times\,49$ tokens |
| $Q,K,V$ (per head) | $T_iW_Q$, $T_iW_K$, $T_iW_V$ | $49\times32$ each (3 heads) |
| Scores | $QK^\top/\sqrt{d}+B$ | $49\times49$ |
| Softmax | row-wise, sums to 1 | $49\times49$ |
| Weighted sum | softmax $\times\,V$ | $49\times32$ per head |
| Concat heads | $[\text{head}_1\|\text{head}_2\|\text{head}_3]$ | $49\times96$ |
| Residual + LN | $x' = x+\text{Attn}(\text{LN}(x))$ | $49\times96$ |
| MLP | $96\to384\to96$, GELU | $49\times96$ |
| Residual + LN | $x'' = x'+\text{MLP}(\text{LN}(x'))$ | $49\times96$ |
| Patch merge (end of stage) | concat $2\times2\to384$, project$\to192$ | $28\times28\times192$ |

This is one full stage-1 block plus the transition into stage 2, worked entirely in the
notation and numeric steps of the original derivation — stages 2–4 and the caption decoder are
covered in `swin_captioning_full_walkthrough.md`.
