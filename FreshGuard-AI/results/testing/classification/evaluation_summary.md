# FreshGuard AI -- Classification Evaluation Summary

**Status:** NEWLY MEASURED  
**Date:** 2026-10-12T12:27:14.546455  
**Device:** cpu (CPU only)  
**PyTorch:** 2.14.0+cpu  

---

## Methodology

- `results/splits/test.csv` is **fully held-out** (never used during training or selection).
- Confidence threshold: **0.6** from `config.py` (not tuned on test set).
- Uncertain predictions reported separately; NOT removed from primary metrics.
- Latency: single-image wall-clock after 10 warm-up passes, 50-run average.
- No results are fabricated.

---

## Overall Metrics (Primary)

| Model | Acc (%) | Prec Macro | Rec Macro | **Macro F1** | ROC-AUC | Ckpt MB | Params | Latency ms |
|:------|:-------:|:----------:|:---------:|:------------:|:-------:|:-------:|:------:|:----------:|
| EfficientNetV2-S | 98.08 | 0.9928 | 0.9833 | **0.9875** | 0.9996 | 82.94 | 20,509,526 | 119.925+/-12.825 |
| ConvNeXt-Tiny | 94.23 | 0.9516 | 0.9682 | **0.9589** | 0.997 | 112.15 | 28,020,070 | 99.49+/-11.751 |
| MobileViT-S | 94.23 | 0.9504 | 0.9682 | **0.9558** | 0.999 | 20.6 | 5,104,550 | 71.774+/-16.438 |

---

## Uncertainty Analysis

Confidence threshold: **0.6**

| Model | N Certain | N Uncertain | % Uncertain | Acc (certain) | Macro F1 (certain) |
|:------|:---------:|:-----------:|:-----------:|:--------------:|:------------------:|
| EfficientNetV2-S | 52 | 0 | 0.0% | 0.9808 | 0.9875 |
| ConvNeXt-Tiny | 51 | 1 | 1.92% | 0.9608 | 0.9705 |
| MobileViT-S | 47 | 5 | 9.62% | 0.9787 | 0.9824 |

---

## Per-Class Metrics

### EfficientNetV2-S

| Class | Precision | Recall | F1-Score | Support | ROC-AUC |
|:------|:---------:|:------:|:--------:|:-------:|:-------:|
| Apple_Fresh | 1.0 | 1.0 | 1.0 | 3 | 1.0 |
| Apple_Rotten | 1.0 | 1.0 | 1.0 | 6 | 1.0 |
| Banana_Fresh | 1.0 | 1.0 | 1.0 | 3 | 1.0 |
| Banana_Rotten | 1.0 | 0.9 | 0.9474 | 10 | 0.9976 |
| Orange_Fresh | 1.0 | 1.0 | 1.0 | 8 | 1.0 |
| Orange_Rotten | 0.9565 | 1.0 | 0.9778 | 22 | 1.0 |

### ConvNeXt-Tiny

| Class | Precision | Recall | F1-Score | Support | ROC-AUC |
|:------|:---------:|:------:|:--------:|:-------:|:-------:|
| Apple_Fresh | 1.0 | 1.0 | 1.0 | 3 | 1.0 |
| Apple_Rotten | 0.8571 | 1.0 | 0.9231 | 6 | 1.0 |
| Banana_Fresh | 1.0 | 1.0 | 1.0 | 3 | 1.0 |
| Banana_Rotten | 0.9 | 0.9 | 0.9 | 10 | 0.9881 |
| Orange_Fresh | 1.0 | 1.0 | 1.0 | 8 | 1.0 |
| Orange_Rotten | 0.9524 | 0.9091 | 0.9302 | 22 | 0.9939 |

### MobileViT-S

| Class | Precision | Recall | F1-Score | Support | ROC-AUC |
|:------|:---------:|:------:|:--------:|:-------:|:-------:|
| Apple_Fresh | 1.0 | 1.0 | 1.0 | 3 | 1.0 |
| Apple_Rotten | 0.75 | 1.0 | 0.8571 | 6 | 1.0 |
| Banana_Fresh | 1.0 | 1.0 | 1.0 | 3 | 1.0 |
| Banana_Rotten | 1.0 | 0.9 | 0.9474 | 10 | 1.0 |
| Orange_Fresh | 1.0 | 1.0 | 1.0 | 8 | 1.0 |
| Orange_Rotten | 0.9524 | 0.9091 | 0.9302 | 22 | 0.9939 |

---

*Results newly measured on held-out test split. No fabrication.*
