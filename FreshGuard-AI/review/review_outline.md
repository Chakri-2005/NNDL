# FreshGuard AI — Comprehensive Project Review Outline

## 1. Project Overview & Scope
FreshGuard AI is an edge-compatible deep-learning infrastructure for real-time food freshness assessment and spoilage detection across commercially key fruits (Apple, Banana, Orange). The system evaluates three modern architectures (**EfficientNetV2-S**, **ConvNeXt-Tiny**, and **MobileViT**) under a strict, leakage-safe dataset partitioning framework.

## 2. Executive Summary of Improvements & Innovations
* **Zero-Leakage Group Partitioning**: Eliminated frame-level data leakage present in naive random image splits by grouping exact byte duplicates (SHA-256) and near-duplicate physical fruit bursts (dHash perceptual hashing, Hamming distance $\le 8$).
* **Exact Three-Model Benchmarking**: Standardized evaluation protocol across:
  1. **EfficientNetV2-S**: Efficient compound-scaled CNN backbone.
  2. **ConvNeXt-Tiny**: Modernized pure convolutional architecture.
  3. **MobileViT**: Light-weight hybrid Vision Transformer combining spatial Convolutions with global self-attention.
* **Standardized Classification Head**: Uniform head architecture (`LayerNorm` $\rightarrow$ `Linear(256)` $\rightarrow$ `GELU` $\rightarrow$ `Dropout(0.3)` $\rightarrow$ `Linear(6)`) across all models to ensure fair empirical comparison.
* **Safety-Aware Abstention Layer**: Integrated out-of-distribution confidence thresholding ($P < 0.60 \rightarrow \text{UNCERTAIN}$) for consumer safety fallback.

## 3. Key Artifacts & Document Mapping
* `review/architecture.md`: Visual pipeline and end-to-end processing topology.
* `review/module_details.md`: Module specs (Deduplication, Feature Extraction, Classification, Abstention, Inference API).
* `review/algorithm.md`: Mathematical formulations for forward pass, dHash grouping, and loss/gradient updates.
* `review/hyperparameters.md`: Complete hyperparameter matrix with literature and empirical justifications.
* `review/results_discussion.md`: Empirical benchmark analysis and confusion matrix insights.
* `review/dataset_novelty.md`: Justification of 6-class dataset scope and deduplication methodology.
* `review/research_gap.md`: Identified literature limitations addressed by FreshGuard AI.
* `review/literature_survey.md`: Comprehensive 15-paper survey matrix categorized by focus area.
* `review/base_paper.md`: Deep-dive analysis of anchor baseline paper and novel extensions.
* `review/future_work.md`: Scalability roadmap for INT8 mobile deployment and hyper-spectral imaging.
* `review/review_slides_content.md`: Complete 18-slide content deck for academic review presentation.
