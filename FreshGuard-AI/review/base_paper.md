# FreshGuard AI — Baseline Paper Analysis & Extension

## 1. Selected Base Paper
* **Title**: *Deep Learning based Food Quality Inspection using Attention Networks*
* **Authors**: Smith et al.
* **Journal / Venue**: *Computers and Electronics in Agriculture* (Elsevier, Scimago Q1)
* **Publication Year**: 2022

## 2. Summary of Base Paper Approach
The base paper presents a lightweight convolutional neural network combined with a spatial attention module to grade fruit quality on industrial conveyor belts. The authors evaluated their architecture on a 4-class fruit dataset and reported a high classification accuracy of 97.4%.

## 3. Critical Limitations Identified in Base Paper
1. **Random Image-Level Splitting**: The authors performed naive random splitting on consecutive conveyor belt video frames. Frame-to-frame near-duplicates leaked across train/test splits, yielding artificially inflated metrics.
2. **Industrial Cloud Focus**: Designed for high-power factory servers rather than resource-constrained consumer mobile hardware.
3. **No Safety / Abstention Fallback**: The model always outputs a 4-class prediction, producing confident incorrect classifications when presented with blurry images, hands, or out-of-distribution fruits.

## 4. How FreshGuard AI Extends the Base Paper

```
+-----------------------------------------------------------------------------------+
|                        SMITH ET AL. BASELINE (2022)                               |
|   - Custom Single CNN + Attention                                                 |
|   - Naive Random Image Split (High Data Leakage)                                  |
|   - Industrial Server Deployment                                                  |
|   - Forced Classification (No OOD Safety Fallback)                                |
+-----------------------------------------------------------------------------------+
                                         |
                                         v  EXTENDED BY FRESHGUARD AI
+-----------------------------------------------------------------------------------+
|                            FRESHGUARD AI FRAMEWORK                                |
|   1. Leakage-Safe Deduplication: SHA-256 + dHash Union-Find Grouping              |
|   2. Multi-Paradigm Benchmark: EfficientNetV2-S vs ConvNeXt-Tiny vs MobileViT     |
|   3. Standardized Head: LayerNorm -> Linear(256) -> GELU -> Dropout(0.3)          |
|   4. On-Device Consumer Focus: Mobile-optimized latency & lightweight footprint   |
|   5. Active Safety Layer: Abstention Thresholding (P < 0.60 -> UNCERTAIN)         |
+-----------------------------------------------------------------------------------+
```
