# Swin Transformer results

## Evaluation settings

- Checkpoint epoch: 5
- Device: cuda
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU (8.00 GiB total)
- Samples: 9927 of 9927
- Decoding: greedy; autocast: True
- Confusion matrix: teacher-forced token classification; same frequent word classes for both models, with all other words grouped as OTHER.
- `*_token_class_report.csv` contains per-token precision/recall/F1 for all observed lexical token classes.

## Caption metrics

| Metric | Score |
|---|---:|
| bleu1 | 0.1716 |
| bleu2 | 0.0984 |
| bleu3 | 0.0561 |
| bleu4 | 0.0327 |
| rougeL | 0.2033 |
| ciderD | 0.2171 |
| distinct_caption_rate | 0.5227 |

## Runtime

| Measurement | Result |
|---|---:|
| Loop time, includes image loading | 488.999 |
| Teacher-forced forward | 35.348 |
| Greedy generation | 366.860 |
| Combined model compute | 402.208 |
| Feature analysis (extra forward passes) | 4.489 |
| Images/second including loading | 20.301 |
| Model compute ms/image | 40.517 |
| Peak allocated GPU memory (GiB) | 0.909 |

Model compute times exclude checkpoint loading and metric computation. GPU synchronization is used around timed model calls.
The confusion matrix is over next-word token classes, not image diagnosis classes. It is teacher-forced and should be interpreted as a language/token diagnostic, not clinical classification accuracy.
