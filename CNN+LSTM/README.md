# CNN + LSTM Image Captioning (ROCO v2)

This workspace implements a baseline medical image captioning model using:

- **Encoder**: pretrained `ResNet-50` (classification head removed)
- **Projection**: `2048 -> 512`
- **Decoder**: `Embedding + LSTM + Linear vocab head`
- **Decoding**: greedy autoregressive generation

It follows the same ROCO v2 split/vocabulary pipeline style as your Swin project.

## Folder structure

- `src/vocabulary.py`: tokenizer, vocabulary build/load, caption CSV reader
- `src/dataset.py`: ROCO dataset loader + ImageNet normalization
- `src/cnn_lstm.py`: CNN encoder + LSTM decoder + integrated model
- `src/losses.py`: cross-entropy with `<pad>` ignore
- `train.py`: training + checkpointing + resume
- `evaluate.py`: test loss + BLEU/METEOR/ROUGE-L
- `inference.py`: single image or split-index inference

## Install

```bash
cd "C:\AIE Files\Projects\S5\Dl\CNN+LSTM"
python -m pip install -r requirements.txt
```

## Train

Auto-detects ROCO root from common paths including:
`C:\AIE Files\Projects\S5\Dl\DATA\rocov2`

```bash
python train.py --epochs 8 --batch-size 16 --learning-rate 1e-4 --max-vocab 8000 --max-length 48
```

Quick sanity run:

```bash
python train.py --epochs 1 --train-limit 2000 --valid-limit 500 --batch-size 8
```

Outputs are saved in:

- `artifacts/cnn_lstm_pipeline/last.pt`
- `artifacts/cnn_lstm_pipeline/best.pt`
- `artifacts/cnn_lstm_pipeline/step_XXXXXXX.pt`
- `artifacts/cnn_lstm_pipeline/vocabulary.json`

## Resume training

```bash
python train.py --resume artifacts/cnn_lstm_pipeline/last.pt
```

## Evaluate on test split

```bash
python evaluate.py --checkpoint artifacts/cnn_lstm_pipeline/best.pt
```

Optional JSON export:

```bash
python evaluate.py --checkpoint artifacts/cnn_lstm_pipeline/best.pt --output-json artifacts/cnn_lstm_pipeline/test_metrics.json
```

## Inference

Use dataset sample by split/index:

```bash
python inference.py --checkpoint artifacts/cnn_lstm_pipeline/best.pt --split test --index 0
```

Use a specific image:

```bash
python inference.py --checkpoint artifacts/cnn_lstm_pipeline/best.pt --image-path "C:\AIE Files\Projects\S5\Dl\DATA\rocov2\test_images\test\ROCOv2_2023_test_000001.jpg"
```

## Plotting & Visualization

Generate publication-ready figures for training loss curves, NLP evaluation metrics, qualitative sample captions, and benchmark paper comparison:

```bash
python plot_results.py
```

Generated plots are saved into `figures/`:
- `figures/training_validation_loss.png`: Training & validation loss across epochs.
- `figures/evaluation_metrics_bar.png`: BLEU-1..4, METEOR, ROUGE-L bar chart on test split.
- `figures/paper_vs_our_model.png`: Comparative analysis against IJRASET 2022 paper (Xception+LSTM on Flickr8k vs ResNet-50+LSTM on ROCO v2).
- `figures/qualitative_samples.png`: Grid of test sample images with Ground Truth vs Model Predicted captions.

## Benchmark Comparison: IJRASET 2022 Paper

Reference Paper: **"Image Captioning Generator Using CNN and LSTM"** (*IJRASET, June 2022*)

| Metric / Dimension | IJRASET Paper (Kumar et al.) | Our Implementation |
| :--- | :--- | :--- |
| **Dataset & Domain** | Flickr8k (General Daily Domain) | ROCO v2 (Medical Radiology Domain) |
| **CNN Feature Extractor** | Xception | ResNet-50 (`2048 -> 512` Projection) |
| **Decoder Architecture** | LSTM | Embedding + Single-Layer LSTM |
| **Vocabulary Size** | ~8,000 general words | 8,000 medical/radiology tokens |
| **Test Set Size** | 1,000 test images | 9,927 test images |
| **BLEU-1** | ~0.5300 | **0.2105** |
| **BLEU-2** | ~0.3200 | **0.1098** |
| **BLEU-3** | ~0.2100 | **0.0624** |
| **BLEU-4** | ~0.1400 | **0.0377** |
| **METEOR** | ~0.2400 | **0.1721** |
| **ROUGE-L** | ~0.3500 | **0.1953** |
| **Test Loss** | ~3.12 | **3.4785** |

For a detailed comparative breakdown, visual graphs, and in-depth domain analysis, see [RESULTS_AND_COMPARISON.md](file:///C:/Users/Ravish/.gemini/antigravity/brain/08362b55-b6d2-4559-aadf-8210ab3793a4/RESULTS_AND_COMPARISON.md).

## Included sample images

I added 5 ready-to-use sample images in:

- `samples/ROCOv2_2023_test_000001.jpg`
- `samples/ROCOv2_2023_test_000002.jpg`
- `samples/ROCOv2_2023_test_000003.jpg`
- `samples/ROCOv2_2023_test_000004.jpg`
- `samples/ROCOv2_2023_test_000005.jpg`

Run inference on one of them:

```bash
python inference.py --checkpoint artifacts/cnn_lstm_pipeline/best.pt --image-path "C:\AIE Files\Projects\S5\Dl\CNN+LSTM\samples\ROCOv2_2023_test_000001.jpg"
```
