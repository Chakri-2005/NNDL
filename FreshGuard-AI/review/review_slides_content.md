# FreshGuard AI — Review Presentation Slide Content Deck (18 Slides)

---

### SLIDE 1: Title Slide
* **Title**: FreshGuard AI: Leakage-Safe Real-Time Food Freshness Detection on Edge Hardware
* **Subtitle**: A Multi-Paradigm Deep Learning Benchmark & Abstention-Aware Edge Infrastructure
* **Team**: Team Verdura · Kousik Sarma Lakkaraju, Manikanta Sai, Chakravarthy Vemula
* **Domain**: Computer Vision, Food Safety, Edge AI Deployment

---

### SLIDE 2: Introduction, Problem & Motivation
* **Problem Statement**: Manual inspection of agricultural produce is slow, subjective, and prone to human error, resulting in substantial food waste and safety hazards.
* **Motivation**: Developing a lightweight, offline-capable mobile computer vision system to instantly detect fruit spoilage without cloud reliance.
* **Research Gap**: Existing studies suffer from severe frame-level data leakage (random splitting of burst photos) and lack out-of-distribution safety fallbacks.
* **Objective**: Build a zero-leakage, reproducible benchmark comparing **EfficientNetV2-S**, **ConvNeXt-Tiny**, and **MobileViT** with active confidence abstention.

---

### SLIDE 3: Literature Survey — Deep Learning Architectures
* **EfficientNetV2** (Tan & Le, ICML 2021): Fused-MBConv layers enable faster training and high FLOP efficiency.
* **ConvNeXt** (Liu et al., CVPR 2022): Modernized pure ConvNet with 7x7 depthwise kernels competing with Vision Transformers.
* **MobileViT** (Mehta & Rastegari, ICLR 2022): Hybrid CNN-Transformer architecture optimized for mobile edge vision tasks.
* **Attention in Food Sorting** (Smith et al., Comp. Elec. Agr. 2022): Lightweight CNNs with spatial attention report high sorting accuracy.

---

### SLIDE 4: Literature Survey — Hyperparameters & Regularization
* **Optuna & BayesOpt** (Akiba et al., KDD 2019): Tree-structured Parzen Estimator (TPE) enables automated hyperparameter optimization.
* **Imbalanced Distributions** (Branco et al., ACM CS 2016): Proves Macro-F1 is essential over simple Accuracy for skewed multi-class data.
* **Decoupled Weight Decay (AdamW)** (Loshchilov & Hutter, ICLR 2019): Decouples $L_2$ regularization from gradient updates, boosting generalization.

---

### SLIDE 5: Literature Survey — Edge Deployment & Safety Rejection
* **MobileNets** (Howard et al., CVPR 2017): Depthwise separable convolutions achieve 8x speedup on mobile devices.
* **INT8 Quantization** (Jacob et al., CVPR 2018): 8-bit quantization reduces model footprint by 4x with $<1\%$ accuracy loss.
* **OOD Confidence Thresholding** (Liang et al., ICLR 2018): Temperature scaling & confidence bounds enable safe abstention on unknown inputs.

---

### SLIDE 6: Overall Architecture Pipeline
```
Dataset (Apple/Banana/Orange) -> SHA-256 & dHash Union-Find Grouping -> Group 80/10/10 Split
  -> Deterministic Preprocessing -> 3 Backbones (EfficientNetV2-S, ConvNeXt-Tiny, MobileViT)
  -> Shared Head (LayerNorm -> Linear(256) -> GELU -> Dropout -> Linear(6))
  -> Abstention Engine (P >= 0.60) -> Web Application UI
```

---

### SLIDE 7: Module Specifications
1. **Deduplication Module**: SHA-256 byte check & 64-bit dHash Union-Find clustering ($D \le 8$).
2. **Partitioning Module**: Group-isolated 80% Train, 10% Val, 10% Test split generator.
3. **Augmentation Module**: Training-only RandomFlip, Rotation ($15^\circ$), ColorJitter.
4. **Classification Head**: Standardized 256-dim GELU bottleneck projection layer.
5. **Abstention Engine**: Out-of-Distribution safety rejection for low-confidence frames ($P < 0.60$).
6. **Inference API**: Flask REST server serving real-time predictions to web frontend.

---

### SLIDE 8: Performance Metrics
* **Macro Accuracy**: Overall proportion of correctly classified fruit images.
* **Macro Precision / Recall / F1-Score**: Unweighted mean metric across 6 classes, preventing majority class bias:
  $$\text{Macro F1} = \frac{1}{6} \sum_{c=1}^6 \frac{2 \cdot P_c \cdot R_c}{P_c + R_c}$$
* **Weighted F1-Score**: Support-weighted F1 metric reflecting class distribution balance.
* **Inference Latency (ms)**: Average single-image execution time on target hardware.

---

### SLIDE 9: Deep Learning Architecture 1 — EfficientNetV2-S
* **Backbone**: Progressive compound-scaled convolutional neural network.
* **Key Components**: Fused-MBConv blocks (3x3 Conv + 1x1 Expansion) $\rightarrow$ Standardized Head.
* **Advantage**: High parameter efficiency (~21.5 M parameters) with strong FLOP economy.

---

### SLIDE 10: Deep Learning Architecture 2 — ConvNeXt-Tiny
* **Backbone**: Modernized pure convolutional architecture mimicking ViT design principles.
* **Key Components**: 7x7 Depthwise Convolutions $\rightarrow$ Inverted Bottlenecks $\rightarrow$ LayerNorm $\rightarrow$ Standardized Head.
* **Advantage**: Superior inductive bias for visual contours without Transformer patch overhead (~28.6 M params).

---

### SLIDE 11: Deep Learning Architecture 3 — MobileViT
* **Backbone**: Light-weight hybrid Vision Transformer combining Conv local features with Transformer global attention.
* **Key Components**: MV2 blocks $\rightarrow$ MobileViT blocks (Unfold $\rightarrow$ Transformer $\rightarrow$ Fold) $\rightarrow$ Standardized Head.
* **Advantage**: Compact footprint (~5.6 M params), ideal for resource-constrained mobile edge platforms.

---

### SLIDE 12: Research Contribution & Novelty
1. **Leakage-Safe Partitioning**: First food freshness benchmark implementing perceptual dHash Union-Find group isolation (0 group leakage).
2. **Standardized Multi-Paradigm Comparison**: Fair, identical classifier head comparison across CNN, Modern Conv, and Conv-ViT architectures.
3. **Active Safety Layer**: Production-ready abstention thresholding preventing false safety claims.

---

### SLIDE 13: Algorithm Procedure
1. Index images $\rightarrow$ Calculate SHA-256 & 64-bit dHash.
2. Link near-duplicates ($D \le 8$) $\rightarrow$ Build Union-Find group clusters.
3. Perform Group-Stratified Split $\rightarrow$ Export `train.csv`, `val.csv`, `test.csv`.
4. Stage 1 (3 epochs): Warm-up head with frozen backbone ($\eta_1 = 10^{-3}$).
5. Stage 2 (5 epochs): Unfreeze upper backbone, fine-tune with CosineAnnealingLR ($\eta_2 = 10^{-4}$).
6. Evaluate test set $\rightarrow$ Compute metrics, confusion matrix, FLOPs, and latency.

---

### SLIDE 14: Hyperparameter Table & Justification
* **Image Size**: $224 \times 224$ (Standard ImageNet scale).
* **Batch Size**: 32 (Stable gradient estimation).
* **Optimizer**: AdamW with weight decay $\lambda = 10^{-4}$ (Decoupled decay).
* **Dropout**: 0.3 (Head regularization).
* **Abstention Threshold**: 0.60 (Safety fallback).

---

### SLIDE 15: Empirical Results & Discussion
* Summary table comparing EfficientNetV2-S, ConvNeXt-Tiny, and MobileViT across Accuracy, Macro F1, Parameters, FLOPs, and Latency.
* Confusion matrix analysis highlighting inter-class distinction (e.g. `Banana_Fresh` vs `Banana_Rotten`).
* Discussion of trade-offs between parameter scale, FLOP count, and edge latency.

---

### SLIDE 16: Dataset Details & Scope
* **Fruits**: Apple, Banana, Orange (Fresh vs Rotten state).
* **Classes**: 6 standardized target categories.
* **Total Images**: 900 images.
* **Groups**: 78 isolated physical fruit items.
* **Split Ratios**: 81.6% Train, 9.2% Val, 9.2% Test (Group-isolated).

---

### SLIDE 17: User Interface & Web Application
* **Frontend**: Responsive single-page web UI (`slides/ui_mockups/app.html`).
* **Backend**: Flask inference server (`app.py`).
* **Features**: Dynamic model selector (EfficientNetV2-S, ConvNeXt-Tiny, MobileViT), confidence meter, visual freshness badge (FRESH / ROTTEN), and UNCERTAIN safety warning screen.

---

### SLIDE 18: Base Paper & Future Work
* **Base Paper**: Smith et al. (2022), *Computers and Electronics in Agriculture*.
* **Our Extension**: Resolved data leakage via dHash grouping, added multi-paradigm benchmarking, and introduced OOD safety abstention.
* **Future Work**: INT8 ONNX/TFLite mobile quantization, Grad-CAM explainability heatmaps, and hyperspectral time-series quality tracking.
