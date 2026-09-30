"""
FreshGuard AI — Model Comparison & Reporting Generator
Reads evaluation JSON metrics, creates comparison table CSV, comparison plots,
and writes the authoritative empirical report to results/FINAL_RESULTS.md.
"""
import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def generate_comparison():
    print("\n" + "="*60)
    print("  FreshGuard AI — Model Comparison & Final Results Generator")
    print("="*60)

    model_keys = ['efficientnetv2_s', 'convnext_tiny', 'mobilevit']
    metrics_list = []

    for m_key in model_keys:
        json_path = os.path.join(config.METRICS_DIR, f"{m_key}_metrics.json")
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                metrics_list.append(json.load(f))
        else:
            print(f"Warning: Metrics JSON for {m_key} not found at {json_path}")

    if not metrics_list:
        print("No evaluation metrics found to compare. Run scripts/evaluate_all.py first.")
        return

    df_comp = pd.DataFrame(metrics_list)
    df_comp_sorted = df_comp.sort_values('f1_macro', ascending=False).reset_index(drop=True)

    # Save comparison CSV
    csv_path = os.path.join(config.METRICS_DIR, "model_comparison.csv")
    df_comp_sorted.to_csv(csv_path, index=False)
    print(f"Saved model comparison table to {csv_path}")

    # Generate Comparison Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    models = df_comp_sorted['model_name'].tolist()
    x = np.arange(len(models))
    width = 0.2

    accs = [r * 100 for r in df_comp_sorted['accuracy']]
    precs = [r * 100 for r in df_comp_sorted['precision_macro']]
    recs = [r * 100 for r in df_comp_sorted['recall_macro']]
    f1s = [r * 100 for r in df_comp_sorted['f1_macro']]

    rects1 = ax.bar(x - 1.5*width, accs, width, label='Accuracy (%)', color='#0284c7')
    rects2 = ax.bar(x - 0.5*width, precs, width, label='Precision (%)', color='#0d9488')
    rects3 = ax.bar(x + 0.5*width, recs, width, label='Recall (%)', color='#d97706')
    rects4 = ax.bar(x + 1.5*width, f1s, width, label='Macro F1 (%)', color='#16a34a')

    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title('FreshGuard AI — 3 Benchmark Architectures Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 110)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, axis='y')

    # Add text labels on bars
    for rects in [rects1, rects2, rects3, rects4]:
        for bar in rects:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    chart_path = os.path.join(config.PLOTS_DIR, "model_comparison.png")
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"Saved comparison bar chart to {chart_path}")

    # Generate FINAL_RESULTS.md
    final_md_path = os.path.join(config.RESULTS_DIR, "FINAL_RESULTS.md")
    with open(final_md_path, 'w') as f:
        f.write("# FreshGuard AI — Authoritative Benchmark Results\n\n")
        f.write("This document presents the **empirical evaluation results** of the three target deep-learning architectures trained on the leakage-safe FreshGuard AI dataset split (80% Train, 10% Validation, 10% Test).\n\n")

        f.write("## 1. Executive Summary & Model Comparison Table\n\n")
        f.write("| Model | Accuracy | Precision (Macro) | Recall (Macro) | Macro F1 | Total Parameters | Model Size (FP32) | FLOPs | Latency (ms/img) |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")

        for _, row in df_comp_sorted.iterrows():
            f.write(f"| **{row['model_name']}** | {row['accuracy']*100:.2f}% | {row['precision_macro']*100:.2f}% | {row['recall_macro']*100:.2f}% | **{row['f1_macro']*100:.2f}%** | {row['total_parameters']:,} | {row['model_size_mb']} MB | {row['flops']} | {row['inference_time_ms']} ms |\n")

        f.write("\n\n## 2. Per-Class Performance Breakdown\n\n")
        for m_key in model_keys:
            per_cls_path = os.path.join(config.PER_CLASS_DIR, f"{m_key}.csv")
            if os.path.exists(per_cls_path):
                m_name = [r['model_name'] for r in metrics_list if r['model_key'] == m_key][0]
                f.write(f"### {m_name}\n\n")
                df_pc = pd.read_csv(per_cls_path)
                f.write("| Class Name | Precision | Recall | F1-Score | Support |\n")
                f.write("| :--- | :---: | :---: | :---: | :---: |\n")
                for _, pc_row in df_pc.iterrows():
                    f.write(f"| `{pc_row['class_name']}` | {pc_row['precision']:.4f} | {pc_row['recall']:.4f} | {pc_row['f1_score']:.4f} | {pc_row['support']} |\n")
                f.write("\n")

        f.write("## 3. Methodological Integrity & Leakage Prevention\n\n")
        f.write("- **Zero Data Leakage**: Split partitioning performed at the physical fruit instance level via SHA-256 exact matching and dHash perceptual hash connected components (Hamming distance $\\le 4$).\n")
        f.write("- **Deterministic Protocol**: Fixed seed (`SEED=42`), 6 standardized classes, identical classifier head structure across models (`LayerNorm` $\\rightarrow$ `Linear(256)` $\\rightarrow$ `GELU` $\\rightarrow$ `Dropout(0.3)` $\\rightarrow$ `Linear(6)`).\n")

    print(f"Generated authoritative summary report at {final_md_path}\n")

if __name__ == "__main__":
    generate_comparison()
