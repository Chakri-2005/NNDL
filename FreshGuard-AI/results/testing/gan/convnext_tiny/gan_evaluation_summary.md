# ConvNeXt-Tiny Dedicated GAN — Evaluation Summary

## 1. Specification & Pairing
- **Target Classifier**: `convnext_tiny`
- **GAN Architecture**: `ConvNeXtFruitGenerator` / `ConvNeXtFruitDiscriminator`
- **Latent Space**: $\mathbb{R}^{128}$ | **Image Resolution**: $64 \times 64$
- **Epochs Trained**: 5 | **Device**: cpu

## 2. Quantitative Evaluation
| Metric | Measured Value | Standard Interpretation |
|---|---|---|
| **FID (Fréchet Inception Distance)** | **`368.92`** | Lower indicates closer convergence to real fruit distribution |
| **KID (Kernel Inception Distance)** | **`4.96845`** | Unbiased polynomial kernel MMD divergence |
| **Classifier Consistency** | **`19.8%`** | Endorsement rate by paired `convnext_tiny` |
| **Perceptual Diversity** | **`20.11 bits`** | Mean pairwise Hamming dispersion across samples |
| **Mode Collapse** | **`NO`** | Verified via non-degenerate hash distribution |
| **Generation Latency** | **`3.98 ms/img`** | High throughput ($> 251$ imgs/sec) |

## 3. Artifacts
- Grid: `synthetic_samples_grid.png`
- Loss Curves: `training_loss_curves.png`
- Metrics: `gan_metrics.json`
