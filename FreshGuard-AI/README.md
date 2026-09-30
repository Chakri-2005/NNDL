# FreshGuard AI: Leakage-Safe Food Freshness Classification Framework

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![Flask Web API](https://img.shields.io/badge/Flask-3.0+-green.svg)](https://flask.palletsprojects.com/)
[![Zero Data Leakage](https://img.shields.io/badge/Data--Leakage-Zero-brightgreen.svg)]()

FreshGuard AI is a leakage-safe computer vision benchmarking framework and real-time edge application for food freshness assessment and spoilage detection across commercially key fruits (**Apple**, **Banana**, **Orange**).

---

## 1. Project Title
**FreshGuard AI: Leakage-Safe Real-Time Food Freshness Detection on Edge Hardware**

## 2. Problem Statement
Manual agricultural quality grading is slow, subjective, and prone to human error, contributing significantly to post-harvest food waste and food safety hazards. Automated computer vision inspection offers non-destructive grading, but existing academic models suffer from severe data leakage during evaluation and lack out-of-distribution safety fallbacks.

## 3. Motivation
Deploying lightweight deep learning models directly on consumer edge devices (smartphones, handheld scanners) enables instant, offline food freshness assessment without cloud latency, protecting consumer health and reducing food waste.

## 4. Research Gap
1. **Frame-Level Data Leakage**: Standard random image splitting (`random_split()`) scatters multi-angle burst photos of the same physical fruit item across train and test sets, yielding artificially inflated metrics (e.g., 98–100%).
2. **Inconsistent Architectural Benchmarking**: Prior work compares single custom CNNs against obsolete baselines with non-uniform heads and arbitrary hyperparameters.
3. **Absence of Safety Abstention**: Conventional models output forced predictions on corrupted or out-of-distribution inputs.

## 5. Objectives
1. Implement a zero-leakage dataset partitioning pipeline using SHA-256 byte check and Difference Hashing (dHash) Union-Find group clustering.
2. Benchmark three modern architectural paradigms (**EfficientNetV2-S**, **ConvNeXt-Tiny**, **MobileViT**) under a standardized classifier head.
3. Integrate an active out-of-distribution abstention safety layer ($P < 0.60 \rightarrow \text{UNCERTAIN}$) into a real-time web application.

## 6. Dataset
* **Source**: Derived from Top Indian Fruits quality dataset.
* **Classes**: 6 target categories (`Apple_Fresh`, `Apple_Rotten`, `Banana_Fresh`, `Banana_Rotten`, `Orange_Fresh`, `Orange_Rotten`).
* **Total Images**: 900 images (150 images/class across 78 physical fruit group clusters).

## 7. Data Preprocessing
* Resized to $224 \times 224 \times 3$.
* Normalized using ImageNet channel stats ($\mu=[0.485, 0.456, 0.406]$, $\sigma=[0.229, 0.224, 0.225]$).

## 8. Leakage Prevention Methodology
1. **Exact SHA-256 Hashing**: Identifies file-level duplicates.
2. **Perceptual dHash Clustering**: 64-bit gradient hash links images with Hamming distance $D \le 8$.
3. **Disjoint Set Union (Union-Find)**: Groups linked photos into physical fruit item clusters.
4. **Group-Stratified Partitioning**: Partitioning (80% Train, 10% Val, 10% Test) is strictly enforced at the group level (`results/splits/train.csv`, `val.csv`, `test.csv`).

## 9. Three Benchmark Models
1. **EfficientNetV2-S**: Progressive compound-scaled CNN backbone.
2. **ConvNeXt-Tiny**: Modernized pure convolutional network with 7x7 depthwise kernels.
3. **MobileViT**: Light-weight hybrid Vision Transformer combining local Conv blocks with global self-attention.

*Standardized Classifier Head across all models*:
`LayerNorm` $\rightarrow$ `Linear(in_features, 256)` $\rightarrow$ `GELU` $\rightarrow$ `Dropout(0.3)` $\rightarrow$ `Linear(256, 6)`

## 10. Training Methodology
* **Two-Stage Fine-Tuning**:
  * *Stage 1 (3 Epochs)*: Backbone frozen, classifier head warm-up via AdamW ($\text{lr}=10^{-3}$).
  * *Stage 2 (5 Epochs)*: Unfrozen backbone, fine-tuning via AdamW ($\text{lr}=10^{-4}$) with CosineAnnealingLR.
* **Early Stopping**: Saves best checkpoint based on Validation Macro F1 score (`results/checkpoints/`).

## 11. Evaluation Metrics
Macro Accuracy, Macro Precision, Macro Recall, Macro F1-Score, Weighted F1-Score, Parameter Count, Model Size (FP32 MB), FLOPs, and Single-Image Inference Latency (ms).

## 12. Empirical Results
*Refer to `results/FINAL_RESULTS.md` for complete empirical tables and figures.*

## 13. Model Comparison
Comparison visualizations and CSV metrics are saved in `results/metrics/model_comparison.csv` and `results/plots/model_comparison.png`.

## 14. Real-Time Application
Flask web application (`app.py`) serving dynamic model selection, confidence score meters, visual freshness badges, and safety abstention warnings.

## 15. Installation
```bash
git clone https://github.com/FreshGuard-AI.git
cd FreshGuard-AI
pip install -r requirements.txt
```

## 16. Training Commands
```bash
# 1. Prepare leakage-safe dataset splits
python scripts/prepare_dataset.py

# 2. Train individual benchmark architectures
python scripts/train_efficientnetv2.py
python scripts/train_convnext.py
python scripts/train_mobilevit.py
```

## 17. Evaluation Commands
```bash
# Evaluate all 3 models on the leakage-safe test split
python scripts/evaluate_all.py

# Generate comparison charts and summary report
python scripts/compare_models.py
```

## 18. Inference Command
```bash
# Run Flask web server on http://localhost:5050
python app.py
```

## 19. Project Structure
```text
FreshGuard-AI/
├── app.py                     # Flask inference API server
├── config.py                  # Single source of truth configuration
├── dataset.py                 # PyTorch Dataset & DataLoader
├── requirements.txt           # Python dependencies
├── README.md                  # Comprehensive project documentation
├── models/                    # Model Zoo (EfficientNetV2-S, ConvNeXt-Tiny, MobileViT)
├── scripts/                   # Dataset prep, training & evaluation routines
├── results/                   # Checkpoints, splits, metrics, plots & FINAL_RESULTS.md
├── review/                    # 12 detailed review outline & slide deck markdown files
└── literature/                # Literature survey table
```

## 20. Research Contribution
* Zero-leakage group-stratified dataset partitioning methodology.
* Fair multi-paradigm benchmark (CNN vs ConvNeXt vs Conv-ViT) with standardized classifier head.
* Active safety abstention thresholding for edge production systems.

## 21. Limitations
* Scope limited to 6 fruit/freshness categories.
* RGB modality cannot detect internal decay without hyperspectral imaging.

## 22. Future Work
* INT8 ONNX & TFLite post-training quantization for mobile acceleration.
* Grad-CAM visual heatmap integration into web API.
