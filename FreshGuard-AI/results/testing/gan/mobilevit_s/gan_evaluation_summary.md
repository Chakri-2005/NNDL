# MobileViT-S Dedicated GAN — Evaluation Summary

## 1. Specification & Pairing
- **Target Classifier**: `mobilevit_s`
- **GAN Architecture**: `MobileViTFruitGenerator` / `MobileViTFruitDiscriminator`
- **Latent Space**: $\mathbb{R}^{80}$ | **Image Resolution**: $64 \times 64$
- **Epochs Trained**: 5 | **Device**: cpu

## 2. Quantitative Evaluation
| Metric | Measured Value | Standard Interpretation |
|---|---|---|
| **FID (Fréchet Inception Distance)** | **`413.7`** | Lower indicates closer convergence to real fruit distribution |
| **KID (Kernel Inception Distance)** | **`6.87962`** | Unbiased polynomial kernel MMD divergence |
| **Classifier Consistency** | **`38.5%`** | Endorsement rate by paired `mobilevit_s` |
| **Perceptual Diversity** | **`13.65 bits`** | Mean pairwise Hamming dispersion across samples |
| **Mode Collapse** | **`NO`** | Verified via non-degenerate hash distribution |
| **Generation Latency** | **`2.25 ms/img`** | High throughput ($> 444$ imgs/sec) |

## 3. Artifacts
- Grid: `synthetic_samples_grid.png`
- Loss Curves: `training_loss_curves.png`
- Metrics: `gan_metrics.json`
