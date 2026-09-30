# FreshGuard AI — Subsystem & Module Specification

## 1. Module Inventory

| Module Name | Core Responsibility | Input $\rightarrow$ Output | Technical Implementation |
| :--- | :--- | :--- | :--- |
| **Deduplication Module** | Exact & near-duplicate fruit frame clustering | Raw Image Directory $\rightarrow$ Group ID Graph | SHA-256 Hashing, dHash (64-bit), Union-Find |
| **Data Partitioning Module** | Group-isolated stratified splitting | Image Group IDs $\rightarrow$ `train.csv`, `val.csv`, `test.csv` | Group-Stratified 80/10/10 Split Generator |
| **Augmentation Engine** | Training-only realistic transformations | PIL Image $\rightarrow$ Normalized $224 \times 224$ Tensor | PyTorch `transforms` (Flip, Rotation, ColorJitter) |
| **Feature Extraction Engine** | Deep visual representation extraction | $224 \times 224 \times 3$ Tensor $\rightarrow$ Feature Vector | Pretrained EfficientNetV2-S, ConvNeXt-Tiny, MobileViT |
| **Classification Head** | Logit projection & class scoring | Feature Vector $\rightarrow$ 6 Logit Predictions | `LayerNorm` $\rightarrow$ `Linear(256)` $\rightarrow$ `GELU` $\rightarrow$ `Linear(6)` |
| **Abstention & Safety Engine** | Out-of-Distribution rejection | Softmax Probabilities $\rightarrow$ Verdict / Abstain | Confidence Thresholding ($P \ge 0.60$) |
| **Inference Server API** | Real-time web application serving | Base64 Image JSON $\rightarrow$ Prediction JSON | Flask HTTP Server REST API (`/predict`, `/health`) |

## 2. Detailed Subsystem Specifications

### Deduplication Subsystem (`scripts/prepare_dataset.py`)
1. Reads all images for 6 target classes: `Apple_Fresh`, `Apple_Rotten`, `Banana_Fresh`, `Banana_Rotten`, `Orange_Fresh`, `Orange_Rotten`.
2. Computes 256-bit SHA-256 hash for byte-level duplicate removal.
3. Computes 64-bit difference hash (dHash) to capture visual similarity across burst frames.
4. Performs Union-Find grouping for pairs with Hamming distance $\le 8$.
5. Outputs group-isolated CSV manifests guaranteeing zero leakage.

### Training Engine (`scripts/train_*.py`)
1. Stage 1 (3 epochs): Backbone frozen, AdamW ($\text{lr}=10^{-3}$), warm-up of classification head.
2. Stage 2 (5 epochs): Unfrozen upper backbone, AdamW ($\text{lr}=10^{-4}$), CosineAnnealingLR decay.
3. Checkpoint Saver: Saves best state dict per model based on Validation Macro F1.

### Inference API Engine (`app.py`)
1. Serves UI front-end from `slides/ui_mockups/app.html`.
2. Encapsulates base64 decoding, RGB conversion, deterministic preprocessing, tensor normalization, and model inference.
3. Returns per-class probability vector, primary verdict, visual color badge, and model metadata.
