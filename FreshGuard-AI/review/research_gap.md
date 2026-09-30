# FreshGuard AI — Research Gap & Problem Formulation

## 1. Context & Motivation
Food spoilage accounts for significant economic losses in global supply chains and consumer households. Manual quality inspection is slow, subjective, and prone to human error. Computer vision offers automated non-destructive inspection, but deploying models on edge devices (smartphones, handheld scanners) introduces strict computational and safety constraints.

## 2. Identified Literature Gaps

### Gap 1: Data Leakage in Image Burst Datasets
* **Limitation in Literature**: Most published fruit classification studies rely on random image-level splitting. Because dataset images are often captured in bursts, near-identical photos of the same fruit item end up in both training and test sets, artificially inflating test scores.
* **FreshGuard AI Solution**: Implements exact SHA-256 and perceptual dHash Union-Find group-stratified splitting, guaranteeing zero instance leakage across splits.

### Gap 2: Lack of Standardized Benchmarking Across Diverse Paradigms
* **Limitation in Literature**: Existing papers evaluate single custom CNNs against outdated baselines (ResNet-50, VGG-16) with non-uniform classifier heads and inconsistent training protocols.
* **FreshGuard AI Solution**: Implements a unified benchmark comparing three distinct computational paradigms (**EfficientNetV2-S** CNN, **ConvNeXt-Tiny** modern Conv, and **MobileViT** Conv-Transformer hybrid) under an identical standardized classifier head (`LayerNorm` $\rightarrow$ `Linear(256)` $\rightarrow$ `GELU` $\rightarrow$ `Dropout(0.3)` $\rightarrow$ `Linear(6)`).

### Gap 3: Absence of Safety Fallbacks for Edge Deployment
* **Limitation in Literature**: Standard classifiers output forced predictions even when presented with corrupted, out-of-distribution (OOD), or ambiguous fruit images.
* **FreshGuard AI Solution**: Integrates an active abstention threshold ($P < 0.60 \rightarrow \text{UNCERTAIN}$), allowing the edge system to reject low-confidence inputs to guarantee food safety.
