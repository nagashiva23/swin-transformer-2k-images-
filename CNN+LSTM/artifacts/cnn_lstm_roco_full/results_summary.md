# CNN+LSTM test results

| Metric | Value |
|---|---:|
| Test images | 9927 |
| Device | cuda |
| Evaluation loop time (s) | 57.131 |
| Teacher-forced forward time (s) | 11.644 |
| Model compute time (s) | 38.593 |
| Caption generation time (s) | 26.948 |
| Evaluation throughput (images/s) | 173.760 |
| Evaluation time per image (s) | 0.0058 |
| Token cross-entropy | 3.4033 |
| Teacher-forced token accuracy | 0.3640 |
| Distinct caption rate | 0.4400 |
| BLEU-1 | 0.2199 |
| BLEU-2 | 0.1162 |
| BLEU-3 | 0.0672 |
| BLEU-4 | 0.0412 |
| METEOR | 0.1820 |
| ROUGE-L | 0.2043 |

Evaluation loop time includes test data loading, model calls, and batch decoding; it excludes checkpoint loading and corpus metric calculation. Model compute and caption generation times are measured with CUDA synchronization when running on GPU. Results vary with GPU load, batch size, data storage, and worker count.

RMSE is not applicable to generated captions: token IDs are categorical labels, so numeric distances between IDs have no meaning.
Teacher-forced token accuracy is included as a supplementary diagnostic; BLEU/METEOR/ROUGE and the example grid are more informative for caption quality.
