"""
FreshGuard AI -- Comprehensive Quantitative Evaluation Framework
================================================================
Evaluates EfficientNetV2-S, ConvNeXt-Tiny, and MobileViT-S on the
held-out test set (results/splits/test.csv).

Design rules
------------
* Test set is NEVER used for tuning. It is the immutable benchmark.
* All three models use identical preprocessing (val/test transforms from dataset.py).
* Confidence threshold (0.60) from config.py is applied as-is; uncertain predictions
  are reported separately but NOT excluded from primary accuracy/F1 because the
  ground-truth labels are always known for test-set evaluation.
* Latency: 10 warm-up passes, then 50-run wall-clock average (time.perf_counter).
* Results labelled [NEWLY MEASURED]. Previously recorded results in
  results/metrics/*.json are NOT overwritten by this script.

Usage
-----
    python evaluate.py
    python evaluate.py --models efficientnetv2_s mobilevit
    python evaluate.py --latency-runs 30 --batch-size 16
    python evaluate.py --output-dir my_results/
"""

import os
import sys
import time
import json
import argparse
import warnings
import datetime
import platform

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
)
from sklearn.preprocessing import label_binarize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from dataset import FreshnessDataset, get_transforms
from models import get_efficientnetv2_s, get_convnext_tiny, get_mobilevit

# ── Output directory (overridable via CLI) ────────────────────────────────────
OUT_DIR = os.path.join(config.RESULTS_DIR, "testing", "classification")
os.makedirs(OUT_DIR, exist_ok=True)

CLASS_NAMES = config.CLASS_NAMES  # locked canonical ordering

MODELS_REGISTRY = {
    "efficientnetv2_s": {
        "display_name": "EfficientNetV2-S",
        "factory_fn":   get_efficientnetv2_s,
        "checkpoint":   os.path.join(config.CHECKPOINTS_DIR, "efficientnetv2_s_best.pth"),
    },
    "convnext_tiny": {
        "display_name": "ConvNeXt-Tiny",
        "factory_fn":   get_convnext_tiny,
        "checkpoint":   os.path.join(config.CHECKPOINTS_DIR, "convnext_tiny_best.pth"),
    },
    "mobilevit": {
        "display_name": "MobileViT-S",
        "factory_fn":   get_mobilevit,
        "checkpoint":   os.path.join(config.CHECKPOINTS_DIR, "mobilevit_best.pth"),
    },
}

# =============================================================================
# Helpers
# =============================================================================

def get_hardware_info():
    info = {
        "timestamp":      datetime.datetime.now().isoformat(),
        "platform":       platform.platform(),
        "python_version": platform.python_version(),
        "torch_version":  torch.__version__,
        "device":         str(config.DEVICE),
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
    return info


def load_model(model_key, device):
    """Load checkpoint into eval mode. Returns (model, checkpoint_was_found)."""
    reg = MODELS_REGISTRY[model_key]
    model = reg["factory_fn"](num_classes=config.NUM_CLASSES, pretrained=False).to(device)
    ckpt_path = reg["checkpoint"]
    loaded = False
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        loaded = True
        mb = os.path.getsize(ckpt_path) / 1e6
        print(f"  [OK] Checkpoint: {ckpt_path}  ({mb:.1f} MB)")
    else:
        warnings.warn(f"  [WARN] Checkpoint not found: {ckpt_path}. Using default weights.")
    model.eval()
    return model, loaded


def build_test_loader(batch_size):
    test_csv = os.path.join(config.SPLITS_DIR, "test.csv")
    if not os.path.exists(test_csv):
        raise FileNotFoundError(
            f"test.csv not found at {test_csv}. "
            "Run scripts/prepare_dataset.py first."
        )
    _, val_test_tf = get_transforms()
    ds = FreshnessDataset(test_csv, transform=val_test_tf)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=torch.cuda.is_available(),
    )


def run_inference(model, loader, device):
    """Forward pass over the full loader. Returns (labels, preds, probs, max_confs)."""
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            probs  = F.softmax(logits, dim=1)
            preds  = torch.argmax(probs, dim=1)
            all_labels.extend(labels.numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    labels_arr = np.array(all_labels)
    preds_arr  = np.array(all_preds)
    probs_arr  = np.vstack(all_probs)
    max_confs  = probs_arr.max(axis=1)
    return labels_arr, preds_arr, probs_arr, max_confs


def measure_latency(model, device, num_warmup=10, num_runs=50):
    """Single-image wall-clock latency in milliseconds."""
    model.eval()
    dummy = torch.randn(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE).to(device)
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy)
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = model(dummy)
            times.append((time.perf_counter() - t0) * 1000.0)
    return {
        "mean_ms":        round(float(np.mean(times)),   3),
        "std_ms":         round(float(np.std(times)),    3),
        "min_ms":         round(float(np.min(times)),    3),
        "max_ms":         round(float(np.max(times)),    3),
        "median_ms":      round(float(np.median(times)), 3),
        "num_runs":       num_runs,
        "batch_size":     1,
        "device":         str(device),
        "timing_method":  "time.perf_counter (wall-clock)",
    }


def measure_throughput(model, loader, device):
    """Batch throughput in images / second."""
    model.eval()
    total = 0
    t0 = time.perf_counter()
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            _ = model(images)
            total += images.size(0)
    elapsed = time.perf_counter() - t0
    return {
        "total_images":            total,
        "elapsed_sec":             round(elapsed, 3),
        "throughput_imgs_per_sec": round(total / elapsed, 2),
        "batch_size":              loader.batch_size,
    }


def compute_roc_auc(labels, probs, class_names):
    n = len(class_names)
    lb = label_binarize(labels, classes=list(range(n)))
    per_class, valid_idxs = {}, []
    for i, cls in enumerate(class_names):
        if lb[:, i].sum() > 0:
            try:
                auc = roc_auc_score(lb[:, i], probs[:, i])
                per_class[cls] = round(float(auc), 4)
                valid_idxs.append(i)
            except Exception:
                per_class[cls] = None
        else:
            per_class[cls] = None
    macro = None
    if valid_idxs:
        try:
            macro = round(float(
                roc_auc_score(lb[:, valid_idxs], probs[:, valid_idxs], average="macro")
            ), 4)
        except Exception:
            pass
    return {"per_class_auc": per_class, "macro_auc": macro}


# ── Plot helpers ──────────────────────────────────────────────────────────────

def plot_confusion_matrix(labels, preds, class_names, model_key, display_name):
    cm = confusion_matrix(labels, preds, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label",      fontsize=11)
    ax.set_title(
        f"Confusion Matrix -- {display_name}\n[NEWLY MEASURED on held-out test set]",
        fontsize=11, fontweight="bold",
    )
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"confusion_matrix_{model_key}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {path}")
    return path


def plot_roc_curves(labels, probs, class_names, model_key, display_name):
    n  = len(class_names)
    lb = label_binarize(labels, classes=list(range(n)))
    fig, ax = plt.subplots(figsize=(9, 7))
    colors  = plt.cm.Set2(np.linspace(0, 1, n))
    drawn   = False
    for i, (cls, color) in enumerate(zip(class_names, colors)):
        if lb[:, i].sum() == 0:
            continue
        try:
            fpr, tpr, _ = roc_curve(lb[:, i], probs[:, i])
            auc_v = roc_auc_score(lb[:, i], probs[:, i])
            ax.plot(fpr, tpr, label=f"{cls} (AUC={auc_v:.3f})", color=color, lw=1.8)
            drawn = True
        except Exception:
            pass
    if not drawn:
        plt.close()
        return None
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate",  fontsize=11)
    ax.set_title(f"ROC Curves (One-vs-Rest) -- {display_name}", fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"roc_curve_{model_key}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {path}")
    return path


def plot_pr_curves(labels, probs, class_names, model_key, display_name):
    n  = len(class_names)
    lb = label_binarize(labels, classes=list(range(n)))
    fig, ax = plt.subplots(figsize=(9, 7))
    colors  = plt.cm.tab10(np.linspace(0, 1, n))
    drawn   = False
    for i, (cls, color) in enumerate(zip(class_names, colors)):
        if lb[:, i].sum() == 0:
            continue
        try:
            p_arr, r_arr, _ = precision_recall_curve(lb[:, i], probs[:, i])
            ap = average_precision_score(lb[:, i], probs[:, i])
            ax.plot(r_arr, p_arr, label=f"{cls} (AP={ap:.3f})", color=color, lw=1.8)
            drawn = True
        except Exception:
            pass
    if not drawn:
        plt.close()
        return None
    ax.set_xlabel("Recall",    fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title(f"Precision-Recall Curves -- {display_name}", fontweight="bold")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"precision_recall_{model_key}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {path}")
    return path


# =============================================================================
# Core per-model evaluation
# =============================================================================

def evaluate_model(model_key, batch_size, latency_runs, device):
    reg          = MODELS_REGISTRY[model_key]
    display_name = reg["display_name"]
    ckpt_path    = reg["checkpoint"]

    print(f"\n{'='*55}")
    print(f"  Evaluating: {display_name}")
    print(f"{'='*55}")

    model, ckpt_loaded = load_model(model_key, device)

    ckpt_size_mb = (
        round(os.path.getsize(ckpt_path) / 1e6, 2)
        if os.path.exists(ckpt_path) else None
    )
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_fp32_mb    = round(total_params * 4 / 1024**2, 2)

    loader = build_test_loader(batch_size)
    n_test = len(loader.dataset)
    print(f"  Inference on {n_test} test samples (batch_size={batch_size}) ...")

    labels, preds, probs, max_confs = run_inference(model, loader, device)

    # ── Primary metrics (all N samples) ───────────────────────────────────────
    threshold   = config.CONFIDENCE_THRESHOLD
    uncertain   = max_confs < threshold
    n_uncertain = int(uncertain.sum())
    n_certain   = n_test - n_uncertain

    acc      = float(accuracy_score(labels, preds))
    prec_mac = float(precision_score(labels, preds, average="macro",    zero_division=0))
    rec_mac  = float(recall_score  (labels, preds, average="macro",    zero_division=0))
    f1_mac   = float(f1_score      (labels, preds, average="macro",    zero_division=0))
    prec_w   = float(precision_score(labels, preds, average="weighted", zero_division=0))
    rec_w    = float(recall_score  (labels, preds, average="weighted", zero_division=0))
    f1_w     = float(f1_score      (labels, preds, average="weighted", zero_division=0))

    # ── Subset metrics for samples above threshold (informational) ─────────────
    acc_cert = f1_cert = None
    if n_certain > 0:
        lc, pc   = labels[~uncertain], preds[~uncertain]
        acc_cert = round(float(accuracy_score(lc, pc)), 4)
        f1_cert  = round(float(f1_score(lc, pc, average="macro", zero_division=0)), 4)

    auc_res = compute_roc_auc(labels, probs, CLASS_NAMES)

    conf_stats = {
        "mean":          round(float(max_confs.mean()), 4),
        "std":           round(float(max_confs.std()),  4),
        "min":           round(float(max_confs.min()),  4),
        "max":           round(float(max_confs.max()),  4),
        "p25":           round(float(np.percentile(max_confs, 25)), 4),
        "p50":           round(float(np.percentile(max_confs, 50)), 4),
        "p75":           round(float(np.percentile(max_confs, 75)), 4),
        "threshold":     threshold,
        "n_uncertain":   n_uncertain,
        "n_certain":     n_certain,
        "pct_uncertain": round(100.0 * n_uncertain / n_test, 2),
    }

    print(f"  Measuring latency ({latency_runs} runs) ...")
    latency    = measure_latency(model, device, num_warmup=10, num_runs=latency_runs)
    throughput = measure_throughput(model, loader, device)

    # ── Per-class breakdown ───────────────────────────────────────────────────
    clf_dict = classification_report(
        labels, preds, target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )
    per_class_rows = [
        {
            "model_key":  model_key,
            "model_name": display_name,
            "class_name": cls,
            "precision":  round(clf_dict[cls]["precision"], 4),
            "recall":     round(clf_dict[cls]["recall"],    4),
            "f1_score":   round(clf_dict[cls]["f1-score"],  4),
            "support":    int(clf_dict[cls]["support"]),
            "roc_auc":    auc_res["per_class_auc"].get(cls),
        }
        for cls in CLASS_NAMES
    ]

    # ── Plots ─────────────────────────────────────────────────────────────────
    cm_path  = plot_confusion_matrix(labels, preds, CLASS_NAMES, model_key, display_name)
    roc_path = plot_roc_curves      (labels, probs, CLASS_NAMES, model_key, display_name)
    pr_path  = plot_pr_curves       (labels, probs, CLASS_NAMES, model_key, display_name)

    # ── Raw predictions CSV ───────────────────────────────────────────────────
    pred_rows = []
    for i in range(n_test):
        row = {
            "index":          i,
            "true_label":     int(labels[i]),
            "true_class":     CLASS_NAMES[int(labels[i])],
            "pred_label":     int(preds[i]),
            "pred_class":     CLASS_NAMES[int(preds[i])],
            "max_confidence": round(float(max_confs[i]), 4),
            "is_correct":     bool(labels[i] == preds[i]),
            "is_uncertain":   bool(max_confs[i] < threshold),
        }
        for j, cls in enumerate(CLASS_NAMES):
            row[f"prob_{cls}"] = round(float(probs[i, j]), 4)
        pred_rows.append(row)
    preds_csv = os.path.join(OUT_DIR, f"predictions_{model_key}.csv")
    pd.DataFrame(pred_rows).to_csv(preds_csv, index=False)
    print(f"  [OK] Raw predictions: {preds_csv}")

    print(
        f"\n  [NEWLY MEASURED] Acc={acc*100:.2f}%  Macro-F1={f1_mac*100:.2f}%"
        f"  ROC-AUC={auc_res['macro_auc']}"
        f"  Uncertain={n_uncertain}/{n_test} ({conf_stats['pct_uncertain']}%)"
        f"  Latency={latency['mean_ms']}+/-{latency['std_ms']} ms"
    )

    return {
        "model_key":             model_key,
        "model_name":            display_name,
        "checkpoint_loaded":     ckpt_loaded,
        "checkpoint_size_mb":    ckpt_size_mb,
        "total_parameters":      total_params,
        "trainable_parameters":  trainable_params,
        "param_size_fp32_mb":    param_fp32_mb,
        "n_test_samples":        n_test,
        "accuracy":              round(acc,      4),
        "precision_macro":       round(prec_mac, 4),
        "recall_macro":          round(rec_mac,  4),
        "f1_macro":              round(f1_mac,   4),
        "precision_weighted":    round(prec_w,   4),
        "recall_weighted":       round(rec_w,    4),
        "f1_weighted":           round(f1_w,     4),
        "accuracy_certain":      acc_cert,
        "f1_macro_certain":      f1_cert,
        "roc_auc_macro":         auc_res["macro_auc"],
        "roc_auc_per_class":     auc_res["per_class_auc"],
        "confidence_stats":      conf_stats,
        "latency":               latency,
        "throughput":            throughput,
        "per_class_rows":        per_class_rows,
        "confusion_matrix_path": cm_path,
        "roc_curve_path":        roc_path,
        "pr_curve_path":         pr_path,
        "predictions_csv":       preds_csv,
    }


# =============================================================================
# Output serialisation
# =============================================================================

def save_outputs(all_results, hw_info):
    # 1. classification_metrics.csv
    flat = []
    skip = {"per_class_rows", "latency", "throughput", "confidence_stats", "roc_auc_per_class"}
    for r in all_results:
        row = {k: v for k, v in r.items() if k not in skip}
        row.update({f"lat_{k}": v for k, v in r["latency"].items()})
        row.update({f"thr_{k}": v for k, v in r["throughput"].items()})
        row.update({f"conf_{k}": v for k, v in r["confidence_stats"].items()})
        flat.append(row)
    pd.DataFrame(flat).to_csv(os.path.join(OUT_DIR, "classification_metrics.csv"), index=False)

    # 2. classification_metrics.json
    json_doc = {
        "experiment":           "FreshGuard AI Classification Evaluation",
        "status":               "NEWLY MEASURED",
        "hardware":             hw_info,
        "confidence_threshold": config.CONFIDENCE_THRESHOLD,
        "class_names":          CLASS_NAMES,
        "note": (
            "Uncertain predictions (confidence < threshold) are reported "
            "separately but NOT excluded from primary accuracy/F1. "
            "Ground-truth labels are always known on the test set."
        ),
        "models": {
            r["model_key"]: {k: v for k, v in r.items() if k != "per_class_rows"}
            for r in all_results
        },
    }
    with open(os.path.join(OUT_DIR, "classification_metrics.json"), "w") as f:
        json.dump(json_doc, f, indent=2)

    # 3. per_class_metrics.csv
    pc_rows = []
    for r in all_results:
        pc_rows.extend(r["per_class_rows"])
    pd.DataFrame(pc_rows).to_csv(os.path.join(OUT_DIR, "per_class_metrics.csv"), index=False)

    # 4. latency_benchmark.csv
    lat_rows = []
    for r in all_results:
        row = {"model_key": r["model_key"], "model_name": r["model_name"]}
        row.update(r["latency"])
        row.update(r["throughput"])
        lat_rows.append(row)
    pd.DataFrame(lat_rows).to_csv(os.path.join(OUT_DIR, "latency_benchmark.csv"), index=False)

    # 5. evaluation_summary.md
    _write_summary_md(all_results, hw_info)
    print(f"\n[OK] All outputs -> {OUT_DIR}")


def _write_summary_md(all_results, hw_info):
    path = os.path.join(OUT_DIR, "evaluation_summary.md")
    cuda_str = (
        f"CUDA -- {hw_info.get('cuda_device_name', '')}"
        if hw_info["cuda_available"] else "CPU only"
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write("# FreshGuard AI -- Classification Evaluation Summary\n\n")
        f.write(f"**Status:** NEWLY MEASURED  \n")
        f.write(f"**Date:** {hw_info['timestamp']}  \n")
        f.write(f"**Device:** {hw_info['device']} ({cuda_str})  \n")
        f.write(f"**PyTorch:** {hw_info['torch_version']}  \n\n")
        f.write("---\n\n## Methodology\n\n")
        f.write("- `results/splits/test.csv` is **fully held-out** (never used during training or selection).\n")
        f.write(f"- Confidence threshold: **{config.CONFIDENCE_THRESHOLD}** from `config.py` (not tuned on test set).\n")
        f.write("- Uncertain predictions reported separately; NOT removed from primary metrics.\n")
        f.write("- Latency: single-image wall-clock after 10 warm-up passes, 50-run average.\n")
        f.write("- No results are fabricated.\n\n")
        f.write("---\n\n## Overall Metrics (Primary)\n\n")
        f.write("| Model | Acc (%) | Prec Macro | Rec Macro | **Macro F1** |")
        f.write(" ROC-AUC | Ckpt MB | Params | Latency ms |\n")
        f.write("|:------|:-------:|:----------:|:---------:|:------------:|")
        f.write(":-------:|:-------:|:------:|:----------:|\n")
        for r in all_results:
            f.write(
                f"| {r['model_name']} | {r['accuracy']*100:.2f} |"
                f" {r['precision_macro']:.4f} | {r['recall_macro']:.4f} |"
                f" **{r['f1_macro']:.4f}** | {r['roc_auc_macro']} |"
                f" {r['checkpoint_size_mb']} | {r['total_parameters']:,} |"
                f" {r['latency']['mean_ms']}+/-{r['latency']['std_ms']} |\n"
            )
        f.write("\n---\n\n## Uncertainty Analysis\n\n")
        f.write(f"Confidence threshold: **{config.CONFIDENCE_THRESHOLD}**\n\n")
        f.write("| Model | N Certain | N Uncertain | % Uncertain |"
                " Acc (certain) | Macro F1 (certain) |\n")
        f.write("|:------|:---------:|:-----------:|:-----------:|"
                ":--------------:|:------------------:|\n")
        for r in all_results:
            cs = r["confidence_stats"]
            f.write(
                f"| {r['model_name']} | {cs['n_certain']} | {cs['n_uncertain']} |"
                f" {cs['pct_uncertain']}% | {r['accuracy_certain']} |"
                f" {r['f1_macro_certain']} |\n"
            )
        f.write("\n---\n\n## Per-Class Metrics\n")
        for r in all_results:
            f.write(f"\n### {r['model_name']}\n\n")
            f.write("| Class | Precision | Recall | F1-Score | Support | ROC-AUC |\n")
            f.write("|:------|:---------:|:------:|:--------:|:-------:|:-------:|\n")
            for pc in r["per_class_rows"]:
                f.write(
                    f"| {pc['class_name']} | {pc['precision']} |"
                    f" {pc['recall']} | {pc['f1_score']} |"
                    f" {pc['support']} | {pc['roc_auc']} |\n"
                )
        f.write("\n---\n\n*Results newly measured on held-out test split. No fabrication.*\n")
    print(f"  [OK] {path}")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="FreshGuard AI -- Comprehensive Classification Evaluation"
    )
    p.add_argument(
        "--models", nargs="+",
        choices=list(MODELS_REGISTRY.keys()),
        default=list(MODELS_REGISTRY.keys()),
    )
    p.add_argument("--batch-size",    type=int, default=config.BATCH_SIZE)
    p.add_argument("--latency-runs",  type=int, default=50)
    p.add_argument("--output-dir",    type=str, default=OUT_DIR)
    return p.parse_args()


def main():
    args = parse_args()
    global OUT_DIR
    OUT_DIR = args.output_dir
    os.makedirs(OUT_DIR, exist_ok=True)

    device  = config.DEVICE
    hw_info = get_hardware_info()

    print("\n" + "="*55)
    print("  FreshGuard AI -- Quantitative Evaluation")
    print("="*55)
    print(f"  Device    : {device}")
    print(f"  Classes   : {CLASS_NAMES}")
    print(f"  Threshold : {config.CONFIDENCE_THRESHOLD}")
    print(f"  Seed      : {config.SEED}")
    print(f"  Output    : {OUT_DIR}")
    print("="*55)

    all_results = []
    for model_key in args.models:
        try:
            result = evaluate_model(model_key, args.batch_size, args.latency_runs, device)
            all_results.append(result)
        except Exception as exc:
            print(f"\n  [ERROR] {model_key}: {exc}")
            import traceback
            traceback.print_exc()

    if all_results:
        save_outputs(all_results, hw_info)
    else:
        print("\n  No models were successfully evaluated.")
    return all_results


if __name__ == "__main__":
    main()
