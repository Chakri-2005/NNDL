# EfficientNetV2-S Dedicated GAN — Evaluation Summary

## 1. Specification & Pairing
- **Target Classifier**: `efficientnetv2_s`
- **GAN Architecture**: `EfficientNetFruitGenerator` / `EfficientNetFruitDiscriminator`
- **Latent Space**: $\mathbb{R}^{100}$ | **Image Resolution**: $64 \times 64$
- **Epochs Trained**: 5 | **Device**: cpu

## 2. Quantitative Evaluation
| Metric | Measured Value | Standard Interpretation |
|---|---|---|
| **FID (Fréchet Inception Distance)** | **`424.43`** | Lower indicates closer convergence to real fruit distribution |
| **KID (Kernel Inception Distance)** | **`5.90329`** | Unbiased polynomial kernel MMD divergence |
| **Classifier Consistency** | **`21.9%`** | Endorsement rate by paired `efficientnetv2_s` |
| **Perceptual Diversity** | **`24.29 bits`** | Mean pairwise Hamming dispersion across samples |
| **Mode Collapse** | **`NO`** | Verified via non-degenerate hash distribution |
| **Generation Latency** | **`1.91 ms/img`** | High throughput ($> 524$ imgs/sec) |

## 3. Artifacts
- Grid: `synthetic_samples_grid.png`
- Loss Curves: `training_loss_curves.png`
- Metrics: `gan_metrics.json`
