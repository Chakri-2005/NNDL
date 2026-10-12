# FreshGuard AI — Testing & Evaluation Framework Guide

This guide details the complete, standardized testing and evaluation framework for **FreshGuard AI**. The framework provides rigorous, leakage-safe quantitative benchmarking, single-image inference and explainability, dataset forensics and environmental robustness testing, automated software unit/integration tests, and conditional GAN evaluation.

---

## 1. Directory Structure

All testing artifacts are organized in dedicated subdirectories under `results/testing/`:

```
results/testing/
├── classification/               # Quantitative benchmark results
│   ├── classification_metrics.csv
│   ├── classification_metrics.json
│   ├── per_class_metrics.csv
│   ├── latency_benchmark.csv
│   ├── evaluation_summary.md
│   ├── predictions_<model>.csv
│   ├── confusion_matrix_<model>.png
│   ├── roc_curve_<model>.png
│   └── precision_recall_<model>.png
├── explainability/               # Grad-CAM heatmaps & visual explanations
│   ├── <model>_cam.png
│   └── explainability_log.csv
├── robustness/                   # Dataset diagnostics, leakage audit & perturbations
│   ├── image_integrity_report.csv
│   ├── duplicate_report.csv
│   ├── robustness_metrics.csv
│   ├── robustness_summary.json
│   └── robustness_comparison.png
├── software_tests/               # Pytest execution logs & summaries
│   ├── test_results.txt
│   └── software_test_summary.json
└── gan/                          # Conditional GAN evaluation
    ├── gan_metrics.json
    ├── gan_evaluation_report.csv
    ├── synthetic_samples_grid.png
    ├── training_loss_curves.png
    └── gan_evaluation_summary.md
```

---

## 2. Environment Setup

Install all testing dependencies:
```bash
pip install -r requirements.txt
pip install pytest pytest-cov scikit-learn opencv-python imagehash captum python-docx
```

---

## 3. Tool Usage & Commands

### Tool 1: Quantitative Classifier Evaluation (`evaluate.py`)
Evaluates all three trained architectures (**EfficientNetV2-S**, **ConvNeXt-Tiny**, **MobileViT-S**) on the held-out test split (`results/splits/test.csv`).

```bash
# Evaluate all models with default parameters
python evaluate.py

# Specify specific models and output directory
python evaluate.py --models efficientnetv2_s convnext_tiny mobilevit --output-dir results/testing/classification --latency-runs 50
```
- **Inputs**: Test set CSV (`results/splits/test.csv`), checkpoints in `results/checkpoints/`
- **Outputs**: Metrics CSV/JSON, confusion matrices, ROC/PR curves, per-model raw predictions, latency benchmark table.

---

### Tool 2A: Individual Image Prediction (`predict.py`)
Loads a trained checkpoint, applies standard preprocessing, and performs inference on any input image. Includes abstention thresholding (`confidence < 0.60` -> `UNCERTAIN`).

```bash
# Predict with EfficientNetV2-S
python predict.py --image "data/val/Banana_Fresh/item_000_frame_00.jpg" --model efficientnetv2_s

# Compare across all three models
python predict.py --image "data/val/Orange_Rotten/item_016_frame_00.jpg" --all-models
```
- **Options**: `--model {efficientnetv2_s,convnext_tiny,mobilevit}`, `--all-models`, `--threshold 0.60`

---

### Tool 2B: Explainability via Grad-CAM (`explain.py`)
Generates gradient-weighted class activation maps showing which image regions influenced the classifier's decision.

```bash
# Generate Grad-CAM for a specific image
python explain.py --image "data/val/Banana_Fresh/item_000_frame_00.jpg" --model efficientnetv2_s --output results/testing/explainability/banana_cam.png
```
- **Inputs**: Input image path, model name, target layer (defaults to backbone penultimate layer)
- **Outputs**: 3-panel visual plot (Original Image, Grad-CAM Heatmap, Blended Overlay) + CSV audit log.

---

### Tool 3: Image Integrity, Forensics & Robustness (`forensics.py`)
Performs perceptual hash deduplication, cross-split leakage detection, and evaluates model degradation under 10 environmental transformations.

```bash
python forensics.py --output-dir results/testing/robustness
```
- **Diagnostics**: File corruption, aspect ratios, Laplacian blur variance, dark/overexposed clipping, SHA-256 exact duplicates, dHash/pHash near-duplicates across train/val/test splits.
- **Perturbations Tested**: Baseline, Brightness (+30%, -30%), Contrast (1.5x, 0.6x), Gaussian blur (r=1.5), Gaussian noise (std=0.05), Small rotations (+15°, -15°), Resampling/Downscale (128x128 -> 256x256).

---

### Tool 4: Automated Software Testing (`pytest`)
Executes 21 comprehensive unit, integration, and regression tests.

```bash
# Run full test suite
pytest -v

# Run with test coverage
pytest -v --cov=.
```
- **Test Suites**:
  - `tests/test_data_pipeline.py`: Class mappings, preprocessing dimensions, tensor bounds, missing image handling, split integrity.
  - `tests/test_model_inference.py`: Model output shapes `[B, 6]`, checkpoint weight loading, probability normalization, confidence threshold abstention.
  - `tests/test_api_integration.py`: Flask `/api/health`, `/api/models`, `/api/metrics`, `/api/predict`, `/api/compare` endpoints.
  - `tests/test_regression.py`: Deterministic seed reproducibility, invariant inference weights without gradient mutation.

---

### Tool 5: Conditional GAN Evaluation (`gan_evaluate.py`)
Evaluates class-conditional fruit synthesis, measuring distribution divergence (FID/KID), perceptual diversity, and generation latency.

```bash
# Train prototype cGAN and evaluate
python gan_evaluate.py --train --epochs 5 --output-dir results/testing/gan

# Evaluate existing cGAN checkpoint
python gan_evaluate.py --checkpoint results/checkpoints/cgan_fruit_best.pth --output-dir results/testing/gan
```
- **Outputs**: `synthetic_samples_grid.png`, `training_loss_curves.png`, `gan_metrics.json`, `gan_evaluation_report.csv`, `gan_evaluation_summary.md`.
