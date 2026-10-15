# CNN+LSTM results

## Evaluation settings

- Checkpoint epoch: 16
- Device: cuda
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU (8.00 GiB total)
- Samples: 9927 of 9927
- Decoding: greedy; autocast: True
- Confusion matrix: teacher-forced token classification; same frequent word classes for both models, with all other words grouped as OTHER.
- `*_token_class_report.csv` contains per-token precision/recall/F1 for all observed lexical token classes.

## Caption metrics

| Metric | Score |
|---|---:|
| bleu1 | 0.1571 |
| bleu2 | 0.0871 |
| bleu3 | 0.0482 |
| bleu4 | 0.0273 |
| rougeL | 0.1831 |
| ciderD | 0.1726 |
| distinct_caption_rate | 0.4398 |

## Runtime

| Measurement | Result |
|---|---:|
| Loop time, includes image loading | 155.834 |
| Teacher-forced forward | 20.634 |
| Greedy generation | 58.549 |
| Combined model compute | 79.183 |
| Feature analysis (extra forward passes) | 4.515 |
| Images/second including loading | 63.702 |
| Model compute ms/image | 7.976 |
| Peak allocated GPU memory (GiB) | 0.531 |

Model compute times exclude checkpoint loading and metric computation. GPU synchronization is used around timed model calls.
The confusion matrix is over next-word token classes, not image diagnosis classes. It is teacher-forced and should be interpreted as a language/token diagnostic, not clinical classification accuracy.
