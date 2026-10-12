"""
GAN Laboratory Routes — FreshGuard AI
Serves per-model GAN metrics, synthetic sample grids,
and loss-curve images for EfficientNetV2-S, ConvNeXt-Tiny,
and MobileViT-S dedicated GANs.
"""
from flask import Blueprint, jsonify, send_file, abort
import os
import json
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

gan_bp = Blueprint("gan", __name__)

# Resolve project root so we can build absolute artifact paths
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_GAN_RESULTS_DIR = os.path.join(_PROJECT_ROOT, "results", "testing", "gan")

_VALID_MODELS = ["efficientnetv2_s", "convnext_tiny", "mobilevit_s"]

_DISPLAY_NAMES = {
    "efficientnetv2_s": "EfficientNetV2-S",
    "convnext_tiny": "ConvNeXt-Tiny",
    "mobilevit_s": "MobileViT-S",
}

_GAN_IDS = {
    "efficientnetv2_s": "efficientnet_gan",
    "convnext_tiny": "convnext_gan",
    "mobilevit_s": "mobilevit_gan",
}


def _model_dir(model_key: str) -> str:
    return os.path.join(_GAN_RESULTS_DIR, model_key)


# ── GET /api/gan/all ────────────────────────────────────────────────────────
@gan_bp.route("/gan/all", methods=["GET"])
def get_all_gan_metrics():
    """Return metrics for all three GANs in a single response."""
    result = {}
    for key in _VALID_MODELS:
        metrics_path = os.path.join(_model_dir(key), "gan_metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, "r", encoding="utf-8") as f:
                result[key] = json.load(f)
        else:
            result[key] = {"error": "metrics not found", "model_key": key}
    return jsonify({"gan_metrics": result})


# ── GET /api/gan/<model_key>/metrics ────────────────────────────────────────
@gan_bp.route("/gan/<model_key>/metrics", methods=["GET"])
def get_gan_metrics(model_key: str):
    """Return GAN evaluation metrics for a specific classifier's GAN."""
    if model_key not in _VALID_MODELS:
        abort(404, description=f"Unknown model: {model_key}. Valid: {_VALID_MODELS}")
    metrics_path = os.path.join(_model_dir(model_key), "gan_metrics.json")
    if not os.path.exists(metrics_path):
        abort(404, description=f"Metrics file not found for {model_key}")
    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return jsonify(data)


# ── GET /api/gan/<model_key>/samples ────────────────────────────────────────
@gan_bp.route("/gan/<model_key>/samples", methods=["GET"])
def get_gan_samples_grid(model_key: str):
    """Return the synthetic-samples grid PNG for a specific GAN."""
    if model_key not in _VALID_MODELS:
        abort(404, description=f"Unknown model: {model_key}")
    grid_path = os.path.join(_model_dir(model_key), "synthetic_samples_grid.png")
    if not os.path.exists(grid_path):
        abort(404, description=f"Sample grid not found for {model_key}")
    return send_file(grid_path, mimetype="image/png")


# ── GET /api/gan/<model_key>/loss_curve ─────────────────────────────────────
@gan_bp.route("/gan/<model_key>/loss_curve", methods=["GET"])
def get_gan_loss_curve(model_key: str):
    """Return the training loss-curve PNG for a specific GAN."""
    if model_key not in _VALID_MODELS:
        abort(404, description=f"Unknown model: {model_key}")
    curve_path = os.path.join(_model_dir(model_key), "training_loss_curves.png")
    if not os.path.exists(curve_path):
        abort(404, description=f"Loss curve not found for {model_key}")
    return send_file(curve_path, mimetype="image/png")


# ── GET /api/gan/<model_key>/report ─────────────────────────────────────────
@gan_bp.route("/gan/<model_key>/report", methods=["GET"])
def get_gan_report(model_key: str):
    """Return the markdown evaluation summary for a specific GAN."""
    if model_key not in _VALID_MODELS:
        abort(404, description=f"Unknown model: {model_key}")
    report_path = os.path.join(_model_dir(model_key), "gan_evaluation_summary.md")
    if not os.path.exists(report_path):
        abort(404, description=f"Report not found for {model_key}")
    with open(report_path, "r", encoding="utf-8") as f:
        content = f.read()
    return jsonify({"model_key": model_key, "report_md": content})


# ── GET /api/gan/summary ─────────────────────────────────────────────────────
@gan_bp.route("/gan/summary", methods=["GET"])
def get_gan_summary():
    """Return a lean comparison table for all three GANs."""
    rows = []
    for key in _VALID_MODELS:
        metrics_path = os.path.join(_model_dir(key), "gan_metrics.json")
        if not os.path.exists(metrics_path):
            continue
        with open(metrics_path, "r", encoding="utf-8") as f:
            m = json.load(f)
        rows.append({
            "model_key": key,
            "display_name": _DISPLAY_NAMES[key],
            "gan_id": _GAN_IDS[key],
            "fid_score": m.get("fid_score"),
            "kid_score": m.get("kid_score"),
            "classifier_consistency": m.get("classifier_consistency"),
            "mean_perceptual_diversity_bits": m.get("mean_perceptual_diversity_bits"),
            "mode_collapse_detected": m.get("mode_collapse_detected"),
            "latency_ms_per_image": m.get("latency_ms_per_image"),
            "epochs_trained": m.get("epochs_trained"),
            "train_duration_sec": m.get("train_duration_sec"),
        })
    return jsonify({"gan_summary": rows})
