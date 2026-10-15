# Swin Transformer results

Run `python compare_models_results.py --limit 0 --batch-size 8` from the repository root. This folder contains Swin-only reports, image examples, architecture flow diagram, interactive feature analysis, GPU/CPU timings, and token-level confusion matrices. CNN+LSTM results are stored only in `CNN+LSTM/result/`; the cross-model comparison is stored in `CNN+LSTM/result/comparison/`.

The confusion matrix is for next-token prediction under teacher forcing. This project generates captions; it does not have image diagnosis class labels, so this matrix is not a disease-classification report.

Interactive HTML includes the Swin training history and image-feature PCA/norm plots. Caption-keyword colors are exploratory proxies, not diagnosis classes. Activation maps show feature magnitude, not clinical saliency explanations.
