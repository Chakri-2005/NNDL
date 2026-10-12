"""
FreshGuard AI -- Image Integrity, Forensics & Robustness Testing
=================================================================
Performs:
  1. Image Integrity & Diagnostic Checks:
     - Missing, unreadable, or corrupted files
     - Image dimensions, aspect ratios, color channels
     - Sharpness (Laplacian variance)
     - Extreme brightness / darkness and histogram clipping
     - Exact duplicates (SHA-256)
     - Near-duplicates (Perceptual Hashing - dHash/pHash)
     - Cross-split contamination & leakage check (Train vs Val vs Test)
     * Diagnostic note: Blur and brightness are diagnostic indicators only
       and do not independently establish fruit freshness/rot.

  2. Robustness Evaluation on Held-out Test Set:
     - Controlled transformations:
       * Brightness (+30%, -30%)
       * Contrast (1.5x, 0.6x)
       * Gaussian Blur (kernel 5x5, sigma=1.0)
       * Gaussian Noise (mean 0, std 0.05)
       * Small Rotations (+15 deg, -15 deg)
       * Resampling / Downscale (downsample 128x128 -> 256x256)
     - Evaluates EfficientNetV2-S, ConvNeXt-Tiny, MobileViT-S
     - Measures prediction consistency, flip rate, confidence drift,
       and accuracy/macro-F1 degradation.

Outputs saved to results/testing/robustness/:
  - image_integrity_report.csv
  - duplicate_report.csv
  - robustness_metrics.csv
  - robustness_summary.json
  - robustness_comparison.png
"""

import os
import sys
import glob
import time
import json
import hashlib
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import imagehash
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn.functional as F
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score

# Repository imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from dataset import (
    CLASS_NAMES, NUM_CLASSES, INVERSE_CLASS_MAPPING,
    get_eval_transforms
)
from models.efficientnetv2_s import get_efficientnetv2_s
from models.convnext_tiny import get_convnext_tiny
from models.mobilevit import get_mobilevit_s

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =====================================================================
# PART 1: Image Integrity & Dataset Forensics
# =====================================================================

def compute_image_diagnostics(image_path: str) -> Dict[str, Any]:
    """Computes diagnostic indicators for a single image file."""
    record = {
        'filepath': image_path,
        'filename': os.path.basename(image_path),
        'exists': os.path.exists(image_path),
        'readable': False,
        'corrupted': False,
        'format': None,
        'width': None,
        'height': None,
        'aspect_ratio': None,
        'channels': None,
        'file_size_kb': None,
        'sha256': None,
        'phash': None,
        'dhash': None,
        'mean_brightness': None,
        'std_brightness': None,
        'is_dark': False,
        'is_overexposed': False,
        'sharpness_laplacian_var': None,
        'is_blurry': False,
        'is_unusually_small': False,
        'diagnostic_flags': []
    }

    if not record['exists']:
        record['diagnostic_flags'].append('MISSING_FILE')
        return record

    try:
        record['file_size_kb'] = round(os.path.getsize(image_path) / 1024.0, 2)
        with open(image_path, 'rb') as f:
            content = f.read()
            record['sha256'] = hashlib.sha256(content).hexdigest()

        # PIL validation
        with Image.open(image_path) as pil_img:
            record['format'] = pil_img.format
            record['width'], record['height'] = pil_img.size
            record['aspect_ratio'] = round(record['width'] / max(1, record['height']), 3)
            pil_img_rgb = pil_img.convert('RGB')
            record['phash'] = str(imagehash.phash(pil_img_rgb))
            record['dhash'] = str(imagehash.dhash(pil_img_rgb))

            if record['width'] < 128 or record['height'] < 128:
                record['is_unusually_small'] = True
                record['diagnostic_flags'].append('UNUSUALLY_SMALL')

            if abs(record['aspect_ratio'] - 1.0) > 0.3:
                record['diagnostic_flags'].append('UNUSUAL_ASPECT_RATIO')

        # OpenCV diagnostics
        cv_img = cv2.imread(image_path)
        if cv_img is None:
            record['corrupted'] = True
            record['diagnostic_flags'].append('CORRUPTED_OR_UNREADABLE')
            return record

        record['readable'] = True
        record['channels'] = cv_img.shape[2] if len(cv_img.shape) > 2 else 1

        # Brightness & contrast
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        mean_b = float(np.mean(gray))
        std_b = float(np.std(gray))
        record['mean_brightness'] = round(mean_b, 2)
        record['std_brightness'] = round(std_b, 2)

        if mean_b < 35.0:
            record['is_dark'] = True
            record['diagnostic_flags'].append('VERY_DARK')
        elif mean_b > 220.0:
            record['is_overexposed'] = True
            record['diagnostic_flags'].append('OVEREXPOSED')

        # Sharpness via Laplacian variance
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        record['sharpness_laplacian_var'] = round(lap_var, 2)
        if lap_var < 50.0:
            record['is_blurry'] = True
            record['diagnostic_flags'].append('LOW_SHARPNESS_BLUR')

    except Exception as exc:
        record['corrupted'] = True
        record['diagnostic_flags'].append(f'READ_ERROR: {str(exc)}')

    return record


def run_dataset_integrity_audit(output_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Audits train, val, and test splits for corruption, duplicates, and leakage."""
    print("=" * 70)
    print("  FreshGuard AI -- Dataset Integrity & Split Forensics Audit")
    print("=" * 70)

    splits_dir = config.SPLITS_DIR
    split_files = {
        'train': os.path.join(splits_dir, 'train.csv'),
        'val': os.path.join(splits_dir, 'val.csv'),
        'test': os.path.join(splits_dir, 'test.csv'),
    }

    records = []
    split_hashes = {'train': {}, 'val': {}, 'test': {}}
    split_dhashes = {'train': {}, 'val': {}, 'test': {}}

    for split_name, split_path in split_files.items():
        if not os.path.exists(split_path):
            print(f"  [WARN] Split file not found: {split_path}")
            continue

        df_split = pd.read_csv(split_path)
        print(f"  Auditing split '{split_name}' ({len(df_split)} samples)...")

        for _, row in df_split.iterrows():
            img_p = row['filepath']
            diag = compute_image_diagnostics(img_p)
            diag['split'] = split_name
            diag['label'] = row.get('label', None)
            diag['class_name'] = CLASS_NAMES[row['label']] if 'label' in row and row['label'] in CLASS_NAMES else None
            records.append(diag)

            if diag['sha256']:
                split_hashes[split_name][img_p] = diag['sha256']
            if diag['dhash']:
                split_dhashes[split_name][img_p] = imagehash.hex_to_hash(diag['dhash'])

    df_report = pd.DataFrame(records)

    # Cross-split duplicate and near-duplicate detection
    print("\n  Analyzing cross-split exact and perceptual duplicates (leakage check)...")
    duplicate_records = []

    # 1. Exact duplicates (SHA-256)
    sha_map = {}
    for r in records:
        sha = r['sha256']
        if sha:
            sha_map.setdefault(sha, []).append(r)

    for sha, items in sha_map.items():
        if len(items) > 1:
            splits_involved = set(it['split'] for it in items)
            is_cross_split = len(splits_involved) > 1
            duplicate_records.append({
                'duplicate_type': 'EXACT_SHA256',
                'hash_value': sha,
                'count': len(items),
                'cross_split_leakage': is_cross_split,
                'splits': list(splits_involved),
                'sample_files': [it['filepath'] for it in items]
            })

    # 2. Perceptual near-duplicates across splits (dHash Hamming distance <= 3)
    test_dhashes = {r['filepath']: imagehash.hex_to_hash(r['dhash']) for r in records if r['split'] == 'test' and r['dhash']}
    train_val_dhashes = {r['filepath']: (r['split'], imagehash.hex_to_hash(r['dhash'])) for r in records if r['split'] in ('train', 'val') and r['dhash']}

    for test_fp, test_h in test_dhashes.items():
        for tv_fp, (tv_split, tv_h) in train_val_dhashes.items():
            dist = test_h - tv_h
            if dist <= 3:
                duplicate_records.append({
                    'duplicate_type': f'NEAR_DUPLICATE_DHASH_DIST_{dist}',
                    'hash_value': f'{test_h} vs {tv_h}',
                    'count': 2,
                    'cross_split_leakage': True,
                    'splits': ['test', tv_split],
                    'sample_files': [test_fp, tv_fp]
                })

    df_duplicates = pd.DataFrame(duplicate_records)

    # Save reports
    rep_csv = os.path.join(output_dir, 'image_integrity_report.csv')
    dup_csv = os.path.join(output_dir, 'duplicate_report.csv')
    df_report.to_csv(rep_csv, index=False)
    df_duplicates.to_csv(dup_csv, index=False)

    total_audited = len(df_report)
    corrupted_count = (df_report['corrupted'] == True).sum()
    blurry_count = (df_report['is_blurry'] == True).sum()
    dark_count = (df_report['is_dark'] == True).sum()
    overexposed_count = (df_report['is_overexposed'] == True).sum()
    leakage_count = sum(1 for d in duplicate_records if d.get('cross_split_leakage', False))

    print(f"\n  [Audit Summary]")
    print(f"    Total Images Audited    : {total_audited}")
    print(f"    Corrupted/Unreadable    : {corrupted_count}")
    print(f"    Low Sharpness (Blurry)  : {blurry_count}")
    print(f"    Very Dark Images        : {dark_count}")
    print(f"    Overexposed Images      : {overexposed_count}")
    print(f"    Cross-Split Leakage Hits: {leakage_count}")
    print(f"    Integrity report saved  : {rep_csv}")
    print(f"    Duplicate report saved  : {dup_csv}")

    return df_report, df_duplicates


# =====================================================================
# PART 2: Robustness Evaluation on Held-out Test Data
# =====================================================================

def apply_transformation(img: Image.Image, trans_name: str, rng: np.random.RandomState) -> Image.Image:
    """Applies a specific controlled transformation to a PIL image."""
    img_rgb = img.convert('RGB')

    if trans_name == 'baseline':
        return img_rgb

    elif trans_name == 'brightness_plus_30':
        enhancer = ImageEnhance.Brightness(img_rgb)
        return enhancer.enhance(1.3)

    elif trans_name == 'brightness_minus_30':
        enhancer = ImageEnhance.Brightness(img_rgb)
        return enhancer.enhance(0.7)

    elif trans_name == 'contrast_high_1.5x':
        enhancer = ImageEnhance.Contrast(img_rgb)
        return enhancer.enhance(1.5)

    elif trans_name == 'contrast_low_0.6x':
        enhancer = ImageEnhance.Contrast(img_rgb)
        return enhancer.enhance(0.6)

    elif trans_name == 'gaussian_blur':
        return img_rgb.filter(ImageFilter.GaussianBlur(radius=1.5))

    elif trans_name == 'gaussian_noise':
        arr = np.array(img_rgb, dtype=np.float32) / 255.0
        noise = rng.normal(loc=0.0, scale=0.05, size=arr.shape).astype(np.float32)
        noisy = np.clip(arr + noise, 0.0, 1.0)
        return Image.fromarray((noisy * 255).astype(np.uint8))

    elif trans_name == 'rotation_plus_15':
        return img_rgb.rotate(15, resample=Image.BILINEAR, expand=False)

    elif trans_name == 'rotation_minus_15':
        return img_rgb.rotate(-15, resample=Image.BILINEAR, expand=False)

    elif trans_name == 'resample_downscale_128':
        down = img_rgb.resize((128, 128), resample=Image.BILINEAR)
        return down.resize((256, 256), resample=Image.BILINEAR)

    else:
        raise ValueError(f"Unknown transformation: {trans_name}")


TRANSFORMATIONS = [
    'baseline',
    'brightness_plus_30',
    'brightness_minus_30',
    'contrast_high_1.5x',
    'contrast_low_0.6x',
    'gaussian_blur',
    'gaussian_noise',
    'rotation_plus_15',
    'rotation_minus_15',
    'resample_downscale_128'
]


def load_model(model_name: str):
    """Loads a trained classifier checkpoint in eval mode."""
    ckpt_name = f"{model_name}_best.pth"
    ckpt_path = os.path.join(config.CHECKPOINTS_DIR, ckpt_name)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    if model_name == 'efficientnetv2_s':
        model = get_efficientnetv2_s(num_classes=NUM_CLASSES, pretrained=False)
    elif model_name == 'convnext_tiny':
        model = get_convnext_tiny(num_classes=NUM_CLASSES, pretrained=False)
    elif model_name == 'mobilevit':
        model = get_mobilevit_s(num_classes=NUM_CLASSES, pretrained=False)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


def evaluate_model_robustness(
    model_name: str,
    test_df: pd.DataFrame,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """Evaluates a model across all perturbation transformations on held-out test data."""
    print(f"\n  Evaluating robustness for '{model_name}' on {len(test_df)} test samples...")
    model = load_model(model_name)
    eval_tf = get_eval_transforms()

    # Baseline inferences first
    rng_baseline = np.random.RandomState(seed)
    baseline_preds = []
    baseline_confs = []

    for _, row in test_df.iterrows():
        img = Image.open(row['filepath']).convert('RGB')
        t_img = eval_tf(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = model(t_img)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        pred_cls = int(np.argmax(probs))
        conf = float(probs[pred_cls])
        baseline_preds.append(pred_cls)
        baseline_confs.append(conf)

    y_true = test_df['label'].to_numpy()
    baseline_acc = float(accuracy_score(y_true, baseline_preds))
    baseline_f1 = float(f1_score(y_true, baseline_preds, average='macro', zero_division=0))

    results = []

    for t_idx, trans in enumerate(TRANSFORMATIONS):
        rng = np.random.RandomState(seed + t_idx)
        preds = []
        confs = []

        for _, row in test_df.iterrows():
            img = Image.open(row['filepath']).convert('RGB')
            p_img = apply_transformation(img, trans, rng)
            t_img = eval_tf(p_img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                logits = model(t_img)
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            pred_cls = int(np.argmax(probs))
            conf = float(probs[pred_cls])
            preds.append(pred_cls)
            confs.append(conf)

        acc = float(accuracy_score(y_true, preds))
        f1 = float(f1_score(y_true, preds, average='macro', zero_division=0))

        # Prediction consistency compared to baseline
        agreed = sum(1 for p, b in zip(preds, baseline_preds) if p == b)
        agreement_rate = float(agreed / len(preds))
        flip_rate = float(1.0 - agreement_rate)
        mean_conf = float(np.mean(confs))
        conf_shift = float(mean_conf - np.mean(baseline_confs))

        res = {
            'model': model_name,
            'transformation': trans,
            'accuracy': round(acc, 4),
            'macro_f1': round(f1, 4),
            'acc_delta_vs_baseline': round(acc - baseline_acc, 4),
            'f1_delta_vs_baseline': round(f1 - baseline_f1, 4),
            'prediction_agreement_rate': round(agreement_rate, 4),
            'flip_rate': round(flip_rate, 4),
            'mean_confidence': round(mean_conf, 4),
            'confidence_shift': round(conf_shift, 4),
            'test_sample_count': len(preds)
        }
        results.append(res)
        print(f"    [{trans:24s}] Acc: {acc:.3f} (Delta {acc - baseline_acc:+.3f}) | F1: {f1:.3f} | Flip Rate: {flip_rate*100:.1f}%")

    return results


def plot_robustness_comparison(df_rob: pd.DataFrame, output_path: str):
    """Generates comparison bar plots showing accuracy degradation across perturbations."""
    plt.figure(figsize=(14, 7))
    sns.set_theme(style='whitegrid')

    palette = {'efficientnetv2_s': '#10b981', 'convnext_tiny': '#3b82f6', 'mobilevit': '#f59e0b'}
    labels_map = {
        'efficientnetv2_s': 'EfficientNetV2-S',
        'convnext_tiny': 'ConvNeXt-Tiny',
        'mobilevit': 'MobileViT-S'
    }

    df_rob_plot = df_rob.copy()
    df_rob_plot['model_label'] = df_rob_plot['model'].map(labels_map)

    ax = sns.barplot(
        data=df_rob_plot,
        x='transformation',
        y='macro_f1',
        hue='model_label',
        palette=['#10b981', '#3b82f6', '#f59e0b']
    )

    plt.title('FreshGuard AI -- Classifier Robustness Under Environmental Perturbations (Macro-F1)', fontsize=14, weight='bold', pad=15)
    plt.xlabel('Perturbation Transformation', fontsize=12, labelpad=10)
    plt.ylabel('Macro-F1 Score', fontsize=12, labelpad=10)
    plt.xticks(rotation=30, ha='right', fontsize=10)
    plt.ylim(0.0, 1.05)
    plt.legend(title='Model Architecture', title_fontsize='11', fontsize='10', loc='lower left')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"  [OK] Robustness comparison plot saved -> {output_path}")


# =====================================================================
# Main Execution
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="FreshGuard AI -- Forensics & Robustness Testing")
    parser.add_argument('--models', nargs='+', default=['efficientnetv2_s', 'convnext_tiny', 'mobilevit'],
                        choices=['efficientnetv2_s', 'convnext_tiny', 'mobilevit'])
    parser.add_argument('--output-dir', default='results/testing/robustness',
                        help="Directory to save robustness artifacts")
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Dataset Integrity & Forensics Audit
    df_report, df_duplicates = run_dataset_integrity_audit(args.output_dir)

    # 2. Robustness Testing on Test Set
    test_csv = os.path.join(config.SPLITS_DIR, 'test.csv')
    if not os.path.exists(test_csv):
        print(f"  [ERROR] Held-out test set not found at {test_csv}!")
        sys.exit(1)

    test_df = pd.read_csv(test_csv)
    print(f"\n  Loaded held-out test set with {len(test_df)} samples.")

    all_robustness_metrics = []
    for model_name in args.models:
        metrics = evaluate_model_robustness(model_name, test_df, seed=args.seed)
        all_robustness_metrics.extend(metrics)

    df_rob = pd.DataFrame(all_robustness_metrics)
    csv_out = os.path.join(args.output_dir, 'robustness_metrics.csv')
    df_rob.to_csv(csv_out, index=False)

    # Save JSON summary
    json_out = os.path.join(args.output_dir, 'robustness_summary.json')
    summary_dict = {
        'audit_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'models_tested': args.models,
        'test_sample_count': len(test_df),
        'perturbations': TRANSFORMATIONS,
        'results': df_rob.to_dict(orient='records')
    }
    with open(json_out, 'w', encoding='utf-8') as f:
        json.dump(summary_dict, f, indent=2)

    # Plot
    plot_out = os.path.join(args.output_dir, 'robustness_comparison.png')
    plot_robustness_comparison(df_rob, plot_out)

    print("\n" + "=" * 70)
    print("  FreshGuard AI Forensics & Robustness Testing Complete!")
    print(f"  Artifacts stored in: {args.output_dir}")
    print("=" * 70)

if __name__ == '__main__':
    main()
