"""
FreshGuard AI — Standardized Evaluation Framework
Evaluates all 3 benchmark models on the leakage-safe test set (test.csv).
Computes complete metrics, per-class breakdown, model size, parameter count, FLOPs, and latency.
"""
import os
import sys
import time
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report
)
try:
    from torchinfo import summary
except ImportError:
    summary = None

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from dataset import get_dataloaders
from models import get_efficientnetv2_s, get_convnext_tiny, get_mobilevit

MODELS_MAP = {
    'efficientnetv2_s': ('EfficientNetV2-S', get_efficientnetv2_s, 'efficientnetv2_s_best.pth'),
    'convnext_tiny': ('ConvNeXt-Tiny', get_convnext_tiny, 'convnext_tiny_best.pth'),
    'mobilevit': ('MobileViT', get_mobilevit, 'mobilevit_best.pth')
}

def measure_latency(model, device, num_runs=50):
    model.eval()
    dummy = torch.randn(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE).to(device)
    # Warmup
    for _ in range(5):
        _ = model(dummy)
    
    start_t = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy)
    end_t = time.time()
    
    avg_latency_ms = ((end_t - start_t) / num_runs) * 1000.0
    return round(avg_latency_ms, 2)

def evaluate_single_model(model_key):
    name, get_model_fn, ckpt_name = MODELS_MAP[model_key]
    device = config.DEVICE
    print(f"\nEvaluating {name} on leakage-safe test set...")

    _, _, test_loader, class_names = get_dataloaders()
    model = get_model_fn(num_classes=config.NUM_CLASSES, pretrained=False).to(device)

    ckpt_path = os.path.join(config.CHECKPOINTS_DIR, ckpt_name)
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f" Loaded checkpoint: {ckpt_path}")
    else:
        print(f" ⚠️ Checkpoint not found at {ckpt_path}. Evaluating with default weights.")

    model.eval()

    # Metrics computation
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            preds = torch.argmax(out, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    acc = float(accuracy_score(all_labels, all_preds))
    prec_macro = float(precision_score(all_labels, all_preds, average='macro', zero_division=0))
    rec_macro = float(recall_score(all_labels, all_preds, average='macro', zero_division=0))
    f1_macro = float(f1_score(all_labels, all_preds, average='macro', zero_division=0))

    prec_weighted = float(precision_score(all_labels, all_preds, average='weighted', zero_division=0))
    rec_weighted = float(recall_score(all_labels, all_preds, average='weighted', zero_division=0))
    f1_weighted = float(f1_score(all_labels, all_preds, average='weighted', zero_division=0))

    # Parameter count and size
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = round(total_params * 4 / (1024 * 1024), 2)  # FP32 MB

    # FLOPs estimation
    flops = "N/A"
    if summary is not None:
        try:
            s_info = summary(model, input_size=(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE), verbose=0, device=device)
            flops = f"{(s_info.total_mult_adds * 2 / 1e9):.2f} GFLOPs"
        except Exception:
            pass

    # Latency
    latency_ms = measure_latency(model, device)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix — {name}')
    plt.tight_layout()
    cm_path = os.path.join(config.PLOTS_DIR, f"{model_key}_confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()

    # Per-class report dataframe
    clf_dict = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True, zero_division=0)
    per_class_rows = []
    for cls_name in class_names:
        per_class_rows.append({
            'class_name': cls_name,
            'precision': round(clf_dict[cls_name]['precision'], 4),
            'recall': round(clf_dict[cls_name]['recall'], 4),
            'f1_score': round(clf_dict[cls_name]['f1-score'], 4),
            'support': int(clf_dict[cls_name]['support'])
        })
    df_per_class = pd.DataFrame(per_class_rows)
    per_class_csv = os.path.join(config.PER_CLASS_DIR, f"{model_key}.csv")
    df_per_class.to_csv(per_class_csv, index=False)

    metrics_res = {
        'model_key': model_key,
        'model_name': name,
        'accuracy': round(acc, 4),
        'precision_macro': round(prec_macro, 4),
        'recall_macro': round(rec_macro, 4),
        'f1_macro': round(f1_macro, 4),
        'precision_weighted': round(prec_weighted, 4),
        'recall_weighted': round(rec_weighted, 4),
        'f1_weighted': round(f1_weighted, 4),
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': model_size_mb,
        'flops': flops,
        'inference_time_ms': latency_ms,
        'per_class_csv': per_class_csv,
        'confusion_matrix_png': cm_path
    }

    json_path = os.path.join(config.METRICS_DIR, f"{model_key}_metrics.json")
    with open(json_path, 'w') as f:
        json.dump(metrics_res, f, indent=4)

    print(f" Metrics saved: Accuracy={acc*100:.2f}%, Macro F1={f1_macro*100:.2f}%, Latency={latency_ms} ms")
    return metrics_res

def evaluate_all_models():
    print("="*60)
    print("  FreshGuard AI — Evaluating All 3 Benchmark Architectures")
    print("="*60)
    all_results = []
    for model_key in MODELS_MAP:
        res = evaluate_single_model(model_key)
        all_results.append(res)
    print("\nAll model evaluations complete!")
    return all_results

if __name__ == "__main__":
    evaluate_all_models()
