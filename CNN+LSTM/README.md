# CNN + LSTM Image Captioning (ROCO v2)

This workspace implements a baseline medical image captioning model using:

- **Encoder**: pretrained `ResNet-50` (classification head removed)
- **Projection**: `2048 -> 512`
- **Decoder**: `Embedding + LSTM + Linear vocab head`
- **Decoding**: greedy autoregressive generation

It follows the same ROCO v2 split/vocabulary pipeline style as your Swin project.

## Folder structure

- `Report.md`: Complete academic report in Markdown with LaTeX equations, loss curves, evaluation tables, qualitative grids, and paper benchmarking
- `Report.tex`: Standalone, compilation-ready LaTeX source file formatted for academic publication
- `plot_results.py`: Script to generate publication-grade figures in `figures/`
- `src/vocabulary.py`: tokenizer, vocabulary build/load, caption CSV reader
- `src/dataset.py`: ROCO dataset loader + ImageNet normalization
- `src/cnn_lstm.py`: CNN encoder + LSTM decoder + integrated model
- `src/losses.py`: cross-entropy with `<pad>` ignore
- `train.py`: training + checkpointing + resume
- `evaluate.py`: test loss + BLEU/METEOR/ROUGE-L
- `inference.py`: single image or split-index inference

## Comprehensive Academic Report

The workspace includes a complete academic report detailing system architecture, LaTeX mathematical derivations, training dynamics, evaluation metrics, and paper comparisons:
- **Markdown Version**: [Report.md](file:///c:/AIE%20Files/Projects/S5/Dl/CNN+LSTM/Report.md)
- **LaTeX Source Code**: [Report.tex](file:///c:/AIE%20Files/Projects/S5/Dl/CNN+LSTM/Report.tex)

## Install

```bash
cd "C:\AIE Files\Projects\S5\Dl\CNN+LSTM"
python -m pip install -r requirements.txt
```

## Train

Auto-detects ROCO root from common paths including:
`C:\AIE Files\Projects\S5\Dl\DATA\rocov2`

```bash
python train.py --epochs 8 --batch-size 32 --learning-rate 1e-4 --encoder-learning-rate 1e-5 --max-vocab 8000 --max-length 48 --output-dir artifacts/cnn_lstm_roco_full
```

The default run uses the full train and validation splits, and executes all 8 requested epochs. The pretrained ResNet-50 is initially frozen; its final residual stage is unfrozen after epoch 2 at a lower learning rate. CUDA mixed precision and validation-driven learning-rate reduction are enabled when CUDA is available. `best.pt` is selected by validation loss; `last.pt` records the last completed epoch. Training does not evaluate or select on the test split.

Quick sanity run:

```bash
python train.py --epochs 1 --train-limit 2000 --valid-limit 500 --batch-size 8
```

Outputs are saved in:

- `artifacts/cnn_lstm_roco_full/last.pt`
- `artifacts/cnn_lstm_roco_full/best.pt`
- `artifacts/cnn_lstm_roco_full/vocabulary.json`
- `artifacts/cnn_lstm_roco_full/metrics.jsonl`

## Resume training

```bash
python train.py --epochs 12 --resume artifacts/cnn_lstm_roco_full/last.pt --output-dir artifacts/cnn_lstm_roco_full
```

## Evaluate on test split

```bash
python evaluate.py --checkpoint artifacts/cnn_lstm_roco_full/best.pt --batch-size 32 --output-json artifacts/cnn_lstm_roco_full/test_metrics.json --output-csv artifacts/cnn_lstm_roco_full/test_predictions.csv
```

This evaluates the held-out test split and saves BLEU-1..4, METEOR, ROUGE-L, token cross-entropy, token accuracy, distinct-caption rate, evaluation-loop/model/generation time, device, and images per second. CUDA timings synchronize the GPU before reading the clock. Loop time includes test loading and generation but excludes checkpoint loading and corpus metric calculation, so timing varies with batch size, GPU load, storage, and worker count. The CSV contains image IDs with reference and generated captions. RMSE is not meaningful for text captions because token IDs are categories, not numeric measurements.

## Inference

Use dataset sample by split/index:

```bash
python inference.py --checkpoint artifacts/cnn_lstm_roco_full/best.pt --split test --index 0
```

Use a specific image:

```bash
python inference.py --checkpoint artifacts/cnn_lstm_roco_full/best.pt --image-path "C:\AIE Files\Projects\S5\Dl\DATA\rocov2\test_images\test\ROCOv2_2023_test_000001.jpg"
```

## Plotting & Visualization

After evaluating the test split, generate training curves, test-metric plots, a qualitative image/caption grid, and a Markdown summary:

```bash
python plot_results.py --artifact-dir artifacts/cnn_lstm_roco_full --data-root "C:\AIE Files\Projects\S5\Dl\DATA\rocov2" --samples 6
```

Generated files are saved under `artifacts/cnn_lstm_roco_full/`: `figures/training_validation_loss.png`, `figures/learning_rates.png` (when logged), `figures/test_caption_metrics.png`, `figures/test_computation_time.png`, `figures/qualitative_test_captions.png`, and `results_summary.md`. Run evaluation first so test plots and image examples use real held-out predictions.

## Side-by-side result package with Swin

From this folder, run the shared analysis script in the parent workspace:

```powershell
python ..\compare_models_results.py --limit 128 --batch-size 4
```

Use `--limit 0 --batch-size 8` for all test images. This evaluates both saved checkpoints on the same rows and writes CNN+LSTM-only outputs into this workspace's `result/` folder. The joint comparison is under `result/comparison/` in this workspace. It includes timing and GPU memory measurements, interactive plots, feature analyses, generated-caption examples, architecture flow, and token-level confusion matrices/reports. Caption generation has no image diagnosis classes, so the confusion matrix measures next-token predictions under teacher forcing.

The interactive epoch-time graph will include real points after training/resuming with the updated trainer. Older epoch logs did not record elapsed time, so their values cannot be reconstructed and remain blank in that graph.

## Comparison with the IJRASET 2022 paper

The cited paper describes an Xception + LSTM system trained on Flickr8k (6,000 train, 1,000 validation, 1,000 test images). This project uses a ResNet-50 + LSTM on ROCO v2, a medical-image caption dataset. Because the domains, data splits, reference captions, and evaluation implementation differ, their caption scores are not a direct performance target. The paper's published result section does not provide a verifiable BLEU/METEOR/ROUGE score table, so this README does not claim numeric paper scores. Compare this run with other models evaluated on the same ROCO v2 split using the same caption preprocessing and metric implementation.

Current run's held-out test scores are generated by `evaluate.py` and written to `artifacts/cnn_lstm_roco_full/test_metrics.json`; don't copy earlier hard-coded scores into a report. The selected checkpoint is based on validation loss, and the test split should be evaluated only after training/model selection is complete.

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
