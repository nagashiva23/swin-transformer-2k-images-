# Comparative Research Paper: Swin Transformer vs. CNN-LSTM on ROCOv2

This directory contains the updated research paper comparing the from-scratch **Swin Transformer** with the ImageNet-pretrained **CNN+LSTM** baseline on the ROCOv2 medical image captioning dataset.

## Paper Details
- **Title**: *From-Scratch Swin Transformer vs. Pretrained CNN--LSTM: An Empirical Study and Comparative Benchmark on Radiology Image Captioning*
- **Format**: IEEE Conference Proceedings (`IEEEtran.cls`)
- **Pages**: 5 Pages
- **Compiled PDF**: `swin_captioning_paper_updated.pdf`
- **LaTeX Source**: `swin_captioning_paper_updated.tex`

## Directory Contents
- `swin_captioning_paper_updated.tex`: Complete LaTeX document with mathematical formulations, full empirical benchmarks, comparative analysis, and figure references.
- `swin_captioning_paper_updated.pdf`: Compiled camera-ready PDF.
- `figures/`: High-resolution figures embedded in the paper:
  - `swin_loss_curve.png`: Full training loss curve for Swin Transformer.
  - `swin_test_metrics.png`: NLP test metric distribution for Swin Transformer.
  - `swin_sample_prediction.png`: Sample test prediction for Swin Transformer.
  - `cnn_lstm_loss.png`: Training & validation loss trajectory for ResNet-50 + LSTM.
  - `cnn_lstm_metrics.png`: Test metric breakdown for CNN + LSTM.
  - `cnn_lstm_benchmark.png`: Comparison against the IJRASET Flickr8k benchmark.
  - `cnn_lstm_qualitative.png`: 2x2 qualitative prediction grid on radiology scans.

## Compilation Command
```bash
pdflatex -interaction=nonstopmode swin_captioning_paper_updated.tex
```
