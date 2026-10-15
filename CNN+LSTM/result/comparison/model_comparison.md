# Swin vs CNN+LSTM on ROCO v2

Test samples: 9927 per model. Greedy decoding is used for both.
Hardware: NVIDIA GeForce RTX 4060 Laptop GPU · CUDA 11.8 · 7.99560546875 GiB total VRAM.

| Model | Parameters (M) | BLEU-1 | BLEU-4 | ROUGE-L | CIDEr-D | Model ms/image | Images/s incl. load | Peak GPU GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Swin Transformer | 90.94 | 0.1716 | 0.0327 | 0.2033 | 0.2171 | 40.52 | 20.30 | 0.91 |
| CNN+LSTM | 35.38 | 0.1571 | 0.0273 | 0.1831 | 0.1726 | 7.98 | 63.70 | 0.53 |

Both models were evaluated using the same test rows, image preprocessing, reference text, and metric implementation. Captioning is not image classification: the per-model confusion matrices classify next-token words under teacher forcing and are diagnostic only.

Swin-only outputs are in `result/swin/`. CNN+LSTM-only outputs are kept in `CNN+LSTM/result/`. This combined report and comparison dashboard are in `CNN+LSTM/result/comparison/`.
