# Comprehensive Technical Report: CNN-LSTM Architecture for Medical Image Captioning on ROCO v2

**Author / Workspace:** ROCO v2 Deep Learning Pipeline (`CNN+LSTM`)

**Date:** August 2026

**Primary Framework:** PyTorch 2.7.1, Torchvision 0.22.1, NLTK 3.9.1, ROUGE-Score 0.1.2

**Dataset:** Radiology Objects in COntext Version 2 (ROCO v2)

---

## Abstract

Automated caption generation for medical imaging bridges the gap between computer vision and clinical text generation. This report provides a complete theoretical, architectural, and empirical analysis of a Deep Convolutional Neural Network (CNN) combined with a Long Short-Term Memory (LSTM) recurrent neural network for medical image captioning. The architecture utilizes a pretrained **ResNet-50** feature extractor coupled via a 512-dimensional linear projection to a single-layer **LSTM** sequence decoder. The system was trained on **59,958 radiology images** from the ROCO v2 dataset, validated on **9,904 images**, and evaluated on an independent test set of **9,927 images**.

Empirical evaluation yields a final test Cross-Entropy Loss of **3.4785**, **BLEU-1** of **0.2105**, **BLEU-2** of **0.1098**, **BLEU-3** of **0.0624**, **BLEU-4** of **0.0377**, **METEOR** of **0.1721**, and **ROUGE-L** of **0.1953**. Furthermore, a comprehensive comparative study is conducted against the benchmark paper *"Image Captioning Generator Using CNN and LSTM"* (IJRASET, June 2022), analyzing architectural differences, domain complexity (general open-domain vs. specialized radiology domain), and metric variance.

---

## 1. System Architecture & Mathematical Formulation

The image captioning pipeline follows an encoder-decoder architecture: a Convolutional Neural Network (CNN) extracts visual feature representations from input medical images, and a Recurrent Neural Network (LSTM) autoregressively generates textual captions token by token.

```
       +-------------------+
       | Input Medical Img |  (3 x 224 x 224)
       +---------+---------+
                 |
                 v
       +-------------------+
       | ResNet-50 Encoder |  (Pretrained ImageNet, AvgPool)
       +---------+---------+
                 |  f_raw in R^(2048)
                 v
       +-------------------+
       | Linear Projection |  v = W_p * f_raw + b_p  in R^(512)
       +---------+---------+
                 |
        +--------+--------+
        |                 |
        v                 v
   +----------+      +----------+
   | h_0 Init |      | c_0 Init |  Initial Hidden & Cell States
   +----+-----+      +----+-----+
        |                 |
        +--------+--------+
                 |
                 v
       +-------------------+
       |   LSTM Decoder    |  <-- Token Embeddings e_t in R^(512)
       +---------+---------+
                 |  h_t in R^(512)
                 v
       +-------------------+
       |   Linear Head &   |  z_t in R^(8000)
       +---------+---------+
                 |
                 v
       +-------------------+
       | Softmax / Greedy  |  Generated Token y_t
       +-------------------+

```

### 1.1 CNN Feature Encoder

Given an input radiology image $I \in \mathbb{R}^{3 \times 224 \times 224}$, the image is normalized using standard ImageNet mean $\boldsymbol{\mu} = [0.485, 0.456, 0.406]$ and standard deviation $\boldsymbol{\sigma} = [0.229, 0.224, 0.225]$.

The image is processed through the ResNet-50 backbone $\phi_{\text{ResNet}}(\cdot)$ with the final classification layer removed:

$$\mathbf{f}_{\text{raw}} = \text{AvgPool}(\phi_{\text{ResNet}}(I)) \in \mathbb{R}^{2048}$$

To match the hidden dimensionality of the language model ($D = 512$), a linear projection layer maps $\mathbf{f}_{\text{raw}}$ into the feature embedding space:

$$\mathbf{v} = W_p \mathbf{f}_{\text{raw}} + b_p \in \mathbb{R}^{512}$$

where $W_p \in \mathbb{R}^{512 \times 2048}$ and $b_p \in \mathbb{R}^{512}$.

### 1.2 LSTM Decoder State Initialization

Rather than feeding the visual vector at every time step, the initial hidden state $h_0$ and cell state $c_0$ of the LSTM are explicitly initialized using non-linear projections of the visual feature vector $\mathbf{v}$:

$$h_0 = \tanh(W_h \mathbf{v} + b_h) \in \mathbb{R}^{1 \times 512}$$

$$c_0 = \tanh(W_c \mathbf{v} + b_c) \in \mathbb{R}^{1 \times 512}$$

where $W_h, W_c \in \mathbb{R}^{512 \times 512}$ and $b_h, b_c \in \mathbb{R}^{512}$.

### 1.3 Recurrent LSTM Decoding & Transition Equations

Let $y_1, y_2, \dots, y_T$ be a sequence of target word token IDs, where $y_1 = \langle\text{bos}\rangle$. Each token $y_{t-1}$ is mapped to a continuous dense vector representation via an embedding matrix $E \in \mathbb{R}^{V \times 512}$:

$$\mathbf{e}_t = E(y_{t-1}) \in \mathbb{R}^{512}$$

At time step $t \in \{1, \dots, T\}$, the single-layer LSTM updates its memory cell $c_t$ and hidden state $h_t$ according to standard gating formulations:

$$\mathbf{i}_t = \sigma(W_{ii} \mathbf{e}_t + b_{ii} + W_{hi} h_{t-1} + b_{hi}) \quad \text{(Input Gate)}$$

$$\mathbf{f}_t = \sigma(W_{if} \mathbf{e}_t + b_{if} + W_{hf} h_{t-1} + b_{hf}) \quad \text{(Forget Gate)}$$

$$\mathbf{g}_t = \tanh(W_{ig} \mathbf{e}_t + b_{ig} + W_{hg} h_{t-1} + b_{hg}) \quad \text{(Cell Candidate)}$$

$$\mathbf{o}_t = \sigma(W_{io} \mathbf{e}_t + b_{io} + W_{ho} h_{t-1} + b_{ho}) \quad \text{(Output Gate)}$$

$$c_t = \mathbf{f}_t \odot c_{t-1} + \mathbf{i}_t \odot \mathbf{g}_t \quad \text{(Cell State Update)}$$

$$h_t = \mathbf{o}_t \odot \tanh(c_t) \quad \text{(Hidden State Update)}$$

where $\sigma(\cdot)$ denotes the sigmoid activation function and $\odot$ represents element-wise Hadamard multiplication.

### 1.4 Vocabulary Logits & Softmax Probability

The hidden state $h_t$ is projected across the vocabulary dimension $V = 8,000$:

$$\mathbf{z}_t = W_o h_t + b_o \in \mathbb{R}^{V}$$

The conditional probability of predicting word token $v \in \{1, \dots, V\}$ at step $t$ is:

$$P(y_t = v \mid y_{<t}, I) = \text{Softmax}(\mathbf{z}_t)_v = \frac{\exp(z_{t, v})}{\sum_{k=1}^V \exp(z_{t, k})}$$

### 1.5 Loss Function Formulation

The model is trained end-to-end (with the CNN backbone frozen) using Masked Cross-Entropy Loss. For a minibatch of $B$ sequences, padding tokens ($y_{i,t} = \langle\text{pad}\rangle$, $\text{pad\_id} = 0$) are masked from gradient computation:

$$\mathcal{L}_{\text{CE}}(\theta) = -\frac{1}{\sum_{i=1}^B \sum_{t=1}^{T_i} \mathbb{I}(y_{i,t} \neq \text{pad\_id})} \sum_{i=1}^B \sum_{t=1}^{T_i} \mathbb{I}(y_{i,t} \neq \text{pad\_id}) \log P(y_{i,t} \mid y_{i,<t}, I_i)$$

where $\mathbb{I}(\cdot)$ is the indicator function.

---

## 2. Quantitative Evaluation Metrics Formulation

To rigorously measure generation performance against ground truth reference captions, six quantitative NLP metrics are evaluated on the test set:

### 2.1 BLEU-N (Bilingual Evaluation Understudy)

BLEU computes modified n-gram precision with a brevity penalty ($\text{BP}$) to penalize short predictions:

$$\text{BLEU-N} = \text{BP} \cdot \exp \left( \sum_{n=1}^N w_n \log p_n \right)$$

$$\text{BP} = \begin{cases} 1 & \text{if } c > r \\ \exp\left(1 - \frac{r}{c}\right) & \text{if } c \le r \end{cases}$$

where $p_n$ is the modified n-gram precision, $c$ is the hypothesis length, $r$ is the reference length, and $w_n = \frac{1}{N}$ (with uniform weights).

### 2.2 METEOR (Metric for Evaluation of Translation with Explicit ORdering)

METEOR computes unigram precision $P$ and recall $R$ combined with stemming and word order penalty:

$$F_{\text{mean}} = \frac{10 \cdot P \cdot R}{R + 9 \cdot P}$$

$$\text{Penalty} = 0.5 \cdot \left( \frac{\text{Number of Chunks}}{\text{Number of Matched Unigrams}} \right)^3$$

$$\text{METEOR} = F_{\text{mean}} \cdot (1 - \text{Penalty})$$

### 2.3 ROUGE-L (Longest Common Subsequence)

ROUGE-L measures sentence-level structure similarity based on the Longest Common Subsequence ($\text{LCS}$):

$$R_{\text{LCS}} = \frac{\text{LCS}(R, C)}{m}, \quad P_{\text{LCS}} = \frac{\text{LCS}(R, C)}{n}$$

$$\text{ROUGE-L} = \frac{(1 + \beta^2) R_{\text{LCS}} P_{\text{LCS}}}{R_{\text{LCS}} + \beta^2 P_{\text{LCS}}}$$

where $m$ and $n$ are reference and candidate lengths respectively ($\beta = 1.2$).

---

## 3. Experimental Setup & Workspace Architecture

### 3.1 Repository Structure

* `src/cnn_lstm.py`: Implements `CNNEncoder`, `LSTMDecoder`, and `CNNLSTMCaptioner`.
* `src/dataset.py`: PyTorch `RoCoCaptionDataset` loader with image transformations and sequence tokenization.
* `src/vocabulary.py`: Word-frequency vocabulary builder, JSON serializer, and greedy tokenizer.
* `src/losses.py`: Masked cross-entropy loss function wrapper ignoring `<pad>` tokens.
* `train.py`: End-to-end training loop with step-wise checkpointing and metrics logging.
* `evaluate.py`: Test set evaluation script for BLEU-1..4, METEOR, ROUGE-L, and test loss.
* `plot_results.py`: Publication-grade visualization generator.

### 3.2 Hyperparameter Configuration

| Parameter | Value | Description |
| --- | --- | --- |
| **Image Resolution** | $224 \times 224$ | ResNet-50 input spatial size |
| **CNN Backbone** | ResNet-50 (Frozen) | Pretrained ImageNet weights |
| **Feature Dimension ($D$)** | $512$ | Linear projection dimension ($2048 \rightarrow 512$) |
| **Embedding Dimension** | $512$ | Token embedding layer dimension |
| **Hidden Dimension** | $512$ | LSTM hidden state dimension |
| **LSTM Layers** | $1$ | Single-layer LSTM cell |
| **Vocabulary Size ($V$)** | $8,000$ | Top frequent words in ROCO v2 train set |
| **Max Sequence Length** | $48$ | Maximum generated caption length |
| **Optimizer** | Adam | Learning rate $\eta = 10^{-4}$ |
| **Gradient Clipping** | $5.0$ | Maximum norm for gradient clipping |
| **Batch Size** | $16$ | Minibatch size during training |
| **Epochs** | $8$ | Total training epochs |

---

## 4. Empirical Results & Figures

### 4.1 Training and Validation Loss Dynamics

Training was performed over 8 full epochs (59,958 training samples per epoch).

#### Epoch-by-Epoch Progress:

| Epoch | Global Step | Training Loss | Validation Loss | Best Validation Loss |
| --- | --- | --- | --- | --- |
| **1** | 3,748 | 4.8351 | 4.2607 | 4.2607 |
| **2** | 7,496 | 4.0250 | 3.9073 | 3.9073 |
| **3** | 11,244 | 3.7319 | 3.7358 | 3.7358 |
| **4** | 14,992 | 3.5450 | 3.6283 | 3.6283 |
| **5** | 18,740 | 3.4078 | 3.5592 | 3.5592 |
| **6** | 22,488 | 3.2950 | 3.5099 | 3.5099 |
| **7** | 26,236 | 3.2001 | 3.4770 | 3.4770 |
| **8** | 29,984 | **3.1161** | **3.4544** | **3.4544** |

---

### 4.2 Full Test Set Quantitative Results

The best model checkpoint (`best.pt`, Epoch 8) was evaluated on the complete ROCO v2 test split (**9,927 test images**).

#### Summary Table:

| Metric | Full Test Score (9,927 samples) | Interpretation |
| --- | --- | --- |
| **Test Loss** | **3.4785** | Masked Cross-Entropy loss |
| **BLEU-1** | **0.2105** | Unigram keyword precision |
| **BLEU-2** | **0.1098** | Bigram phrase precision |
| **BLEU-3** | **0.0624** | Trigram sequence precision |
| **BLEU-4** | **0.0377** | 4-gram exact phrase match precision |
| **METEOR** | **0.1721** | Unigram precision/recall with stemming penalty |
| **ROUGE-L** | **0.1953** | Longest Common Subsequence recall |

---

### 4.3 Qualitative Model Predictions

Below is a visual grid displaying sample radiology test images alongside Ground Truth (GT) captions and model-predicted captions.

#### Key Visual Observations:

1. **Modality Identification**: The model correctly predicts visual scanning modalities (e.g., *CT axial view*, *MRI*, *Angiogram*).
2. **Clinical Terminology Utilization**: The 8,000-word medical vocabulary enables the model to output clinical terms such as *aortic aneurysm*, *thrombosis*, and *extravasation*.

---

## 5. Benchmark Comparison with IJRASET 2022 Reference Paper

To contextualize these results, we perform a side-by-side comparison against the published paper:

> **"Image Captioning Generator Using CNN and LSTM"**
> *Authors:* M. Pranay Kumar, V. Snigdha, R. Nandini, Dr. B. Indira Reddy
> *Publication:* International Journal for Research in Applied Science & Engineering Technology (IJRASET), Vol. 10, Issue VI, June 2022.

### 5.1 Comparative Summary Table

| Dimension / Metric | IJRASET 2022 Paper (Kumar et al.) | Our Model Implementation |
| --- | --- | --- |
| **Target Task Domain** | General Open-Domain Images | Medical Radiology Imaging (ROCO v2) |
| **Dataset Used** | Flickr8k Dataset | ROCO v2 Dataset |
| **Dataset Volume** | 8,000 images (5 captions / image) | 60,000 Train / 9,904 Valid / 9,927 Test |
| **CNN Feature Extractor** | Xception (Deep Convolutional) | ResNet-50 ($2048 \rightarrow 512$ Projection) |
| **Decoder Architecture** | LSTM Sequence Decoder | Embedding + Single-Layer LSTM |
| **Vocabulary Size** | ~8,000 general words | 8,000 medical/radiology terms |
| **Reference Captions / Img** | **5 captions** per image | **1 caption** per image |
| **BLEU-1** | **~0.5300** | **0.2105** |
| **BLEU-2** | **~0.3200** | **0.1098** |
| **BLEU-3** | **~0.2100** | **0.0624** |
| **BLEU-4** | **~0.1400** | **0.0377** |
| **METEOR** | **~0.2400** | **0.1721** |
| **ROUGE-L** | **~0.3500** | **0.1953** |
| **Test Loss** | **~3.12** | **3.4785** |

---

### 5.2 Deep Technical Discussion: Why Do Metric Scores Differ?

1. **Domain Complexity & Linguistic Entropy**:
* **General Domain (Flickr8k)**: Captions describe common daily visual items ("a dog running in green grass"). Standard English words repeat across images, yielding higher n-gram overlap scores (BLEU-1 ~ 0.53).
* **Medical Domain (ROCO v2)**: Captions contain dense medical findings ("Digitally subtracted angiogram of the IMA demonstrated cessation of flow..."). Visual differences in radiology scans (e.g., subtle tissue contrast in X-rays or CT slices) are far harder for standard CNN backbones to discriminate than distinct color objects.


2. **Reference Density**:
* Flickr8k provides **5 distinct ground-truth reference captions** for every image, significantly increasing the probability that a generated n-gram matches at least one reference.
* ROCO v2 provides only **1 reference caption** per image, creating a strict evaluation environment where valid alternative clinical phrases receive zero n-gram credit.


3. **Encoder Feature Projection**:
* The IJRASET paper uses an **Xception** backbone. Our model employs **ResNet-50** with a linear bottleneck projection ($2048 \rightarrow 512$), reducing decoder parameter count (~237MB total weight size) while preserving fast inference latency.



---

## 6. Conclusion & Future Roadmap

This report demonstrated a functional CNN-LSTM medical image captioning pipeline on ROCO v2. The model achieved a validation loss of **3.4544** and demonstrated clinical vocabulary understanding on unseen radiology test images.

### Recommended Future Enhancements:

1. **Encoder Unfreezing**: Unfreeze higher residual blocks of ResNet-50 (`--unfreeze-encoder`) during later epochs to fine-tune visual representations on medical domain features.
2. **Spatial Attention Mechanisms**: Incorporate additive soft attention (e.g., *Show, Attend and Tell*) to allow the LSTM to dynamically attend to specific spatial sub-regions of radiology scans.
3. **Medical Pretrained Backbones**: Replace ImageNet-pretrained ResNet-50 with domain-specific backbones such as **RadImageNet** or **MedCLIP**.
