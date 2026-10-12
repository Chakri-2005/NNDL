"""
FreshGuard AI -- Multi-GAN Independent Evaluation Framework
============================================================
Evaluates each dedicated GAN against its paired classifier and held-out test distribution:
  - GAN 1 (EfficientNetV2-S Dedicated): Evaluates fidelity, FID, KID, and consistency with EfficientNetV2-S
  - GAN 2 (ConvNeXt-Tiny Dedicated):    Evaluates fidelity, FID, KID, and consistency with ConvNeXt-Tiny
  - GAN 3 (MobileViT-S Dedicated):      Evaluates fidelity, FID, KID, and consistency with MobileViT-S

Outputs per GAN saved under:
  results/testing/gan/<model_key>/
    - gan_metrics.json
    - gan_evaluation_report.csv
    - synthetic_samples_grid.png
    - gan_evaluation_summary.md

Combined comparison output:
  results/testing/gan/gan_comparison.csv
  results/testing/gan/gan_comparison.json
  results/testing/gan/gan_comparison.png

Usage:
  python gan_evaluate.py --model efficientnetv2_s
  python gan_evaluate.py --model convnext_tiny
  python gan_evaluate.py --model mobilevit_s
  python gan_evaluate.py --all
"""

import os
import sys
import time
import json
import argparse
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from PIL import Image
from scipy import linalg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.utils import make_grid, save_image
import imagehash

# Project imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from dataset import CLASS_NAMES, NUM_CLASSES
from models.gans import GAN_REGISTRY, resolve_gan_key
from models.efficientnetv2_s import get_efficientnetv2_s
from models.convnext_tiny import get_convnext_tiny
from models.mobilevit import get_mobilevit

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CKPT_DIR = os.path.join(config.RESULTS_DIR, "checkpoints", "gans")
RESULTS_DIR = os.path.join(config.RESULTS_DIR, "testing", "gan")

# Standard Feature Extractor for FID / KID
class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.features = nn.Sequential(*list(base.children())[:-1])
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != 224:
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        with torch.no_grad():
            feat = self.features(x)
            feat = torch.flatten(feat, 1)
        return feat

def compute_fid(real_feats: np.ndarray, fake_feats: np.ndarray) -> float:
    mu1, sigma1 = np.mean(real_feats, axis=0), np.cov(real_feats, rowvar=False)
    mu2, sigma2 = np.mean(fake_feats, axis=0), np.cov(fake_feats, rowvar=False)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    fid = float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)
    return max(0.0, fid)

def compute_kid(real_feats: np.ndarray, fake_feats: np.ndarray, subsets: int = 10, subset_size: int = 50) -> float:
    n_r, n_f = len(real_feats), len(fake_feats)
    sub_size = min(subset_size, n_r, n_f)
    if sub_size < 5:
        return 0.0

    scores = []
    for _ in range(subsets):
        idx_r = np.random.choice(n_r, sub_size, replace=False)
        idx_f = np.random.choice(n_f, sub_size, replace=False)
        x = real_feats[idx_r]
        y = fake_feats[idx_f]

        d = x.shape[1]
        k_xx = (np.dot(x, x.T) / d + 1.0) ** 3
        k_yy = (np.dot(y, y.T) / d + 1.0) ** 3
        k_xy = (np.dot(x, y.T) / d + 1.0) ** 3

        mmd = (np.sum(k_xx) - np.trace(k_xx)) / (sub_size * (sub_size - 1)) + \
              (np.sum(k_yy) - np.trace(k_yy)) / (sub_size * (sub_size - 1)) - \
              2.0 * np.mean(k_xy)
        scores.append(mmd)

    return float(np.mean(scores))

def load_target_classifier(classifier_key: str):
    ckpt_map = {
        'efficientnetv2_s': ('efficientnetv2_s_best.pth', lambda: get_efficientnetv2_s(num_classes=6, pretrained=False)),
        'convnext_tiny': ('convnext_tiny_best.pth', lambda: get_convnext_tiny(num_classes=6, pretrained=False)),
        'mobilevit_s': ('mobilevit_best.pth', lambda: get_mobilevit(num_classes=6, pretrained=False))
    }
    ckpt_file, fn = ckpt_map[classifier_key]
    path = os.path.join(config.CHECKPOINTS_DIR, ckpt_file)
    if not os.path.exists(path):
        return None
    model = fn().to(DEVICE)
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    state = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
    model.load_state_dict(state)
    model.eval()
    return model

def evaluate_single_gan(
    model_key: str,
    feat_extractor: FeatureExtractor,
    real_features: np.ndarray,
    samples_per_class: int = 16
) -> Dict[str, Any]:
    resolved_key = resolve_gan_key(model_key)
    cfg = GAN_REGISTRY[resolved_key]
    display_name = cfg['display_name']
    ckpt_path = os.path.join(CKPT_DIR, cfg['checkpoint_file'])
    out_dir = os.path.join(RESULTS_DIR, cfg['output_subdir'])
    os.makedirs(out_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print(f"  Evaluating {display_name} ({resolved_key})")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Output Dir: {out_dir}")
    print("=" * 70)

    if not os.path.exists(ckpt_path):
        print(f"  [ERROR] Checkpoint missing: {ckpt_path}. Run train_gan.py first.")
        return {'model_key': resolved_key, 'status': 'MISSING_CHECKPOINT'}

    # Load generator
    generator = cfg['generator_cls'](latent_dim=cfg['latent_dim']).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    generator.load_state_dict(ckpt['generator_state_dict'])
    generator.eval()

    history = ckpt.get('history', {})
    epochs_trained = ckpt.get('epochs', None)
    train_duration = ckpt.get('train_duration', None)

    # 1. Generation Latency & Throughput
    num_eval_samples = samples_per_class * NUM_CLASSES
    eval_z = torch.randn(num_eval_samples, cfg['latent_dim'], device=DEVICE)
    eval_labels = torch.tensor([c for c in range(NUM_CLASSES) for _ in range(samples_per_class)], device=DEVICE)

    t0 = time.time()
    with torch.no_grad():
        fake_imgs = generator(eval_z, eval_labels)
    latency_total_ms = (time.time() - t0) * 1000.0
    latency_per_img = latency_total_ms / num_eval_samples
    throughput = (num_eval_samples / (latency_total_ms / 1000.0))

    # Normalize to [0, 1]
    fake_norm = torch.clamp((fake_imgs + 1.0) / 2.0, 0.0, 1.0)

    # 2. Extract Fake Features for FID / KID
    fake_features = feat_extractor(fake_norm).cpu().numpy()
    fid = compute_fid(real_features, fake_features)
    kid = compute_kid(real_features, fake_features)

    # 3. Class-Conditional Semantic Consistency (Paired Classifier)
    clf = load_target_classifier(cfg['classifier_key'])
    consistency = None
    if clf is not None:
        fake_resized = F.interpolate(fake_norm, size=(224, 224), mode='bilinear')
        norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        fake_inp = norm(fake_resized)
        with torch.no_grad():
            preds = clf(fake_inp).argmax(dim=1)
        matches = (preds == eval_labels).sum().item()
        consistency = round(float(matches / num_eval_samples), 4)

    # 4. Diversity & Mode Collapse Check
    hashes = []
    for i in range(num_eval_samples):
        arr = (fake_norm[i].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        hashes.append(imagehash.phash(Image.fromarray(arr)))

    diversity_by_class = {}
    for c in range(NUM_CLASSES):
        c_hashes = [hashes[i] for i in range(c * samples_per_class, (c + 1) * samples_per_class)]
        dists = [c_hashes[i] - c_hashes[j] for i in range(len(c_hashes)) for j in range(i + 1, len(c_hashes))]
        diversity_by_class[CLASS_NAMES[c]] = round(float(np.mean(dists)), 2) if dists else 0.0

    mean_diversity = round(float(np.mean(list(diversity_by_class.values()))), 2)
    mode_collapse = bool(mean_diversity < 1.0)

    # 5. Save Sample Grid
    grid_path = os.path.join(out_dir, "synthetic_samples_grid.png")
    save_image(make_grid(fake_norm, nrow=samples_per_class, padding=2), grid_path)

    # Metrics dictionary
    metrics = {
        'model_key': resolved_key,
        'gan_id': cfg['gan_id'],
        'display_name': display_name,
        'paired_classifier': cfg['classifier_key'],
        'latent_dim': cfg['latent_dim'],
        'epochs_trained': epochs_trained,
        'train_duration_sec': train_duration,
        'fid_score': round(fid, 2),
        'kid_score': round(kid, 5),
        'classifier_consistency': consistency,
        'mean_perceptual_diversity_bits': mean_diversity,
        'mode_collapse_detected': mode_collapse,
        'latency_ms_per_image': round(latency_per_img, 2),
        'throughput_imgs_per_sec': round(throughput, 1),
        'diversity_by_class': diversity_by_class,
        'checkpoint_file': cfg['checkpoint_file'],
        'evaluated_samples': num_eval_samples,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }

    # Save JSON & CSV
    with open(os.path.join(out_dir, "gan_metrics.json"), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    df_csv = pd.DataFrame([{
        'Metric': k, 'Value': str(v)
    } for k, v in metrics.items() if not isinstance(v, dict)])
    df_csv.to_csv(os.path.join(out_dir, "gan_evaluation_report.csv"), index=False)

    # Save Markdown Summary
    md_content = f"""# {display_name} — Evaluation Summary

## 1. Specification & Pairing
- **Target Classifier**: `{cfg['classifier_key']}`
- **GAN Architecture**: `{cfg['generator_cls'].__name__}` / `{cfg['discriminator_cls'].__name__}`
- **Latent Space**: $\\mathbb{{R}}^{{{cfg['latent_dim']}}}$ | **Image Resolution**: $64 \\times 64$
- **Epochs Trained**: {epochs_trained} | **Device**: {DEVICE}

## 2. Quantitative Evaluation
| Metric | Measured Value | Standard Interpretation |
|---|---|---|
| **FID (Fréchet Inception Distance)** | **`{metrics['fid_score']}`** | Lower indicates closer convergence to real fruit distribution |
| **KID (Kernel Inception Distance)** | **`{metrics['kid_score']}`** | Unbiased polynomial kernel MMD divergence |
| **Classifier Consistency** | **`{round(consistency*100, 1) if consistency is not None else 'N/A'}%`** | Endorsement rate by paired `{cfg['classifier_key']}` |
| **Perceptual Diversity** | **`{metrics['mean_perceptual_diversity_bits']} bits`** | Mean pairwise Hamming dispersion across samples |
| **Mode Collapse** | **`{'YES' if mode_collapse else 'NO'}`** | Verified via non-degenerate hash distribution |
| **Generation Latency** | **`{metrics['latency_ms_per_image']} ms/img`** | High throughput ($> {int(throughput)}$ imgs/sec) |

## 3. Artifacts
- Grid: `synthetic_samples_grid.png`
- Loss Curves: `training_loss_curves.png`
- Metrics: `gan_metrics.json`
"""
    with open(os.path.join(out_dir, "gan_evaluation_summary.md"), 'w', encoding='utf-8') as f:
        f.write(md_content)

    print(f"  FID: {metrics['fid_score']} | KID: {metrics['kid_score']} | Diversity: {mean_diversity} bits | Latency: {metrics['latency_ms_per_image']} ms")
    print(f"  [OK] Saved artifacts to {out_dir}")
    return metrics

def plot_gan_comparison(comparison_df: pd.DataFrame, out_path: str):
    plt.figure(figsize=(12, 5))
    sns.set_theme(style='whitegrid')

    plt.subplot(1, 2, 1)
    ax1 = sns.barplot(data=comparison_df, x='display_name', y='fid_score', palette='viridis')
    plt.title('FID Score by Dedicated GAN (Lower is Better)', fontsize=11, weight='bold')
    plt.ylabel('FID')
    plt.xticks(rotation=15, ha='right')

    plt.subplot(1, 2, 2)
    ax2 = sns.barplot(data=comparison_df, x='display_name', y='mean_perceptual_diversity_bits', palette='mako')
    plt.title('Perceptual Hash Diversity (Bits)', fontsize=11, weight='bold')
    plt.ylabel('Hamming Distance (Bits)')
    plt.xticks(rotation=15, ha='right')

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"  [OK] Saved multi-GAN comparison plot -> {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate FreshGuard AI Dedicated Fruit GANs")
    parser.add_argument('--model', choices=['efficientnetv2_s', 'convnext_tiny', 'mobilevit_s', 'all'],
                        default='all', help="Specific GAN model to evaluate or 'all'")
    parser.add_argument('--all', action='store_true', help="Evaluate all 3 GANs")
    parser.add_argument('--samples-per-class', type=int, default=16, help="Samples per class for metrics")
    args = parser.parse_args()

    targets = list(GAN_REGISTRY.keys()) if (args.all or args.model == 'all') else [args.model]

    print("\n" + "=" * 70)
    print("  FreshGuard AI: Multi-GAN Independent Evaluation Suite")
    print(f"  Evaluating: {targets}")
    print("=" * 70)

    # Pre-extract Real Test Image Features once
    feat_extractor = FeatureExtractor().to(DEVICE)
    test_csv = os.path.join(config.SPLITS_DIR, "test.csv")
    test_df = pd.read_csv(test_csv)
    real_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    real_tensors = [real_tf(Image.open(row['filepath']).convert('RGB')) for _, row in test_df.iterrows()]
    real_batch = torch.stack(real_tensors).to(DEVICE)
    real_features = feat_extractor(real_batch).cpu().numpy()

    all_metrics = []
    for key in targets:
        m = evaluate_single_gan(key, feat_extractor, real_features, samples_per_class=args.samples_per_class)
        all_metrics.append(m)

    # Save comparison report if multiple GANs evaluated
    if len(all_metrics) > 1:
        comp_df = pd.DataFrame([{
            'gan_id': m['gan_id'],
            'display_name': m['display_name'],
            'paired_classifier': m['paired_classifier'],
            'latent_dim': m['latent_dim'],
            'fid_score': m['fid_score'],
            'kid_score': m['kid_score'],
            'classifier_consistency': m.get('classifier_consistency'),
            'mean_perceptual_diversity_bits': m['mean_perceptual_diversity_bits'],
            'latency_ms_per_image': m['latency_ms_per_image'],
            'throughput_imgs_per_sec': m['throughput_imgs_per_sec']
        } for m in all_metrics if 'fid_score' in m])

        comp_csv = os.path.join(RESULTS_DIR, "gan_comparison.csv")
        comp_json = os.path.join(RESULTS_DIR, "gan_comparison.json")
        comp_png = os.path.join(RESULTS_DIR, "gan_comparison.png")

        comp_df.to_csv(comp_csv, index=False)
        with open(comp_json, 'w', encoding='utf-8') as f:
            json.dump(all_metrics, f, indent=2)
        plot_gan_comparison(comp_df, comp_png)
        print("\n" + "=" * 70)
        print("  Multi-GAN Comparison Summary:")
        print(comp_df[['display_name', 'fid_score', 'kid_score', 'mean_perceptual_diversity_bits', 'latency_ms_per_image']])
        print("=" * 70)

if __name__ == '__main__':
    main()
