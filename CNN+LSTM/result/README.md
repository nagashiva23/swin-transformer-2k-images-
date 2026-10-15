# CNN+LSTM results

Run `python ..\compare_models_results.py --limit 0 --batch-size 8` from this folder. The script writes this model's report, image examples, architecture flow diagram, interactive feature analysis, compute timing, and token-level confusion matrices here. Cross-model dashboards are saved under `comparison/` inside this folder.

The confusion matrix is for next-token prediction under teacher forcing. This project generates captions; it does not have image diagnosis class labels, so this matrix is not a disease-classification report.

Interactive HTML files include per-epoch training history, the joint metrics/runtime dashboard, and feature PCA/norm distributions. Feature labels are caption-keyword proxies. Activation maps show feature magnitude and are not clinical saliency explanations. Historical epoch durations were not stored; the updated trainer records GPU name, memory, and duration for future epochs.
