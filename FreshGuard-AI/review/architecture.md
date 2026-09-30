# FreshGuard AI — Architectural Specification

## 1. End-to-End System Pipeline
```
+-----------------------------------------------------------------------------------+
|                                 INPUT DATASET                                    |
|              Top Indian Fruits Dataset (Apple, Banana, Orange)                    |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                        PREPROCESSING & LEAKAGE PREVENTION                         |
|   1. Exact SHA-256 byte deduplication                                             |
|   2. Difference Hash (dHash) near-duplicate cluster detection                     |
|   3. Union-Find connected component grouping of physical fruit items               |
|   4. Group-Stratified 80/10/10 Split (train.csv, val.csv, test.csv)                 |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                             FEATURE EXTRACTION BACKBONES                          |
|   +-----------------------+ +-----------------------+ +-----------------------+   |
|   |   EfficientNetV2-S    | |     ConvNeXt-Tiny     | |       MobileViT       |   |
|   |  Compound-scaled CNN  | | Modernized Pure Conv  | |  Conv-ViT Hybrid    |   |
|   |   (Feature Dim: 1280) | |   (Feature Dim: 768)  | |   (Feature Dim: 640)  |   |
|   +-----------------------+ +-----------------------+ +-----------------------+   |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                        STANDARDIZED CLASSIFICATION HEAD                           |
|       LayerNorm(in_features) -> Linear(256) -> GELU -> Dropout(0.3) -> Linear(6)  |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                         INFERENCE & ABSTENTION ENGINE                             |
|          Calculates Softmax Probabilities P = max(p_1, ..., p_6)                  |
|          IF P >= 0.60 -> Output Predicted Class (Fresh/Rotten)                    |
|          IF P <  0.60 -> Output Abstain / UNCERTAIN Fallback                      |
+-----------------------------------------------------------------------------------+
```

## 2. Model Backbone Characteristics

| Architecture | Paradigm | Parameter Count | Key Architectural Advantage |
| :--- | :--- | :---: | :--- |
| **EfficientNetV2-S** | Progressive Compound-Scaled CNN | ~21.5 M | Fused-MBConv layers & progressive learning scale FLOP efficiency. |
| **ConvNeXt-Tiny** | Modernized Pure Convolutional Net | ~28.6 M | 7x7 depthwise convolutions & inverted bottleneck design mimicking ViT. |
| **MobileViT** | Hybrid CNN-Vision Transformer | ~5.6 M | Mobile-friendly local representation via Conv + global attention via Transformer. |

## 3. Standardization Protocol
To isolate backbone quality, all three models share:
1. **Input Dimensions**: $224 \times 224 \times 3$ RGB.
2. **Head Layer Norm**: `nn.LayerNorm(in_features)`.
3. **Hidden Representation**: `nn.Linear(in_features, 256)` followed by `nn.GELU()`.
4. **Regularization**: `nn.Dropout(0.3)`.
5. **Output Projection**: `nn.Linear(256, 6)`.
