# FreshGuard AI — Authoritative Benchmark Results

This document presents the **empirical evaluation results** of the three target deep-learning architectures trained on the leakage-safe FreshGuard AI dataset split (80% Train, 10% Validation, 10% Test).

## 1. Executive Summary & Model Comparison Table

| Model | Accuracy | Precision (Macro) | Recall (Macro) | Macro F1 | Total Parameters | Model Size (FP32) | FLOPs | Latency (ms/img) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **EfficientNetV2-S** | 98.08% | 99.28% | 98.33% | **98.75%** | 20,509,526 | 78.24 MB | 5.70 GFLOPs | 46.57 ms |
| **ConvNeXt-Tiny** | 94.23% | 95.16% | 96.82% | **95.89%** | 28,020,070 | 106.89 MB | 0.64 GFLOPs | 50.35 ms |
| **MobileViT** | 94.23% | 95.04% | 96.82% | **95.58%** | 5,104,550 | 19.47 MB | 1.70 GFLOPs | 36.74 ms |


## 2. Per-Class Performance Breakdown

### EfficientNetV2-S

| Class Name | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| `Apple_Fresh` | 1.0000 | 1.0000 | 1.0000 | 3 |
| `Apple_Rotten` | 1.0000 | 1.0000 | 1.0000 | 6 |
| `Banana_Fresh` | 1.0000 | 1.0000 | 1.0000 | 3 |
| `Banana_Rotten` | 1.0000 | 0.9000 | 0.9474 | 10 |
| `Orange_Fresh` | 1.0000 | 1.0000 | 1.0000 | 8 |
| `Orange_Rotten` | 0.9565 | 1.0000 | 0.9778 | 22 |

### ConvNeXt-Tiny

| Class Name | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| `Apple_Fresh` | 1.0000 | 1.0000 | 1.0000 | 3 |
| `Apple_Rotten` | 0.8571 | 1.0000 | 0.9231 | 6 |
| `Banana_Fresh` | 1.0000 | 1.0000 | 1.0000 | 3 |
| `Banana_Rotten` | 0.9000 | 0.9000 | 0.9000 | 10 |
| `Orange_Fresh` | 1.0000 | 1.0000 | 1.0000 | 8 |
| `Orange_Rotten` | 0.9524 | 0.9091 | 0.9302 | 22 |

### MobileViT

| Class Name | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| `Apple_Fresh` | 1.0000 | 1.0000 | 1.0000 | 3 |
| `Apple_Rotten` | 0.7500 | 1.0000 | 0.8571 | 6 |
| `Banana_Fresh` | 1.0000 | 1.0000 | 1.0000 | 3 |
| `Banana_Rotten` | 1.0000 | 0.9000 | 0.9474 | 10 |
| `Orange_Fresh` | 1.0000 | 1.0000 | 1.0000 | 8 |
| `Orange_Rotten` | 0.9524 | 0.9091 | 0.9302 | 22 |

## 3. Methodological Integrity & Leakage Prevention

- **Zero Data Leakage**: Split partitioning performed at the physical fruit instance level via SHA-256 exact matching and dHash perceptual hash connected components (Hamming distance $\le 4$).
- **Deterministic Protocol**: Fixed seed (`SEED=42`), 6 standardized classes, identical classifier head structure across models (`LayerNorm` $\rightarrow$ `Linear(256)` $\rightarrow$ `GELU` $\rightarrow$ `Dropout(0.3)` $\rightarrow$ `Linear(6)`).
