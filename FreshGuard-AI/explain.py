"""
FreshGuard AI -- Explainability via Grad-CAM
============================================
Generates gradient-weighted class activation maps (Grad-CAM) for a given
fruit image and overlays them on the original image.

Architecture-specific target layers
-------------------------------------
EfficientNetV2-S  : model.features[-1]  (last conv block)
ConvNeXt-Tiny     : model.features[-1][-1].block[-1]  (last LayerNorm after conv)
MobileViT-S (timm): model.backbone.stages[-1]          (last stage)

Note: Grad-CAM is based on convolutional feature maps. For transformer-heavy
layers that lack spatial conv activations (pure attention), the heatmap may
be less spatially precise. This is documented in the output and is not treated
as an error -- the heatmap is labelled as a model explanation, NOT proof of
correctness.

Captum (torch.autograd.grad) is used where available; a lightweight manual
Grad-CAM fallback is included for robustness.

Usage
-----
    python explain.py --image path/to/fruit.jpg
    python explain.py --image path/to/fruit.jpg --model convnext_tiny
    python explain.py --image path/to/fruit.jpg --model all
    python explain.py --image path/to/fruit.jpg --target-class Apple_Fresh
"""

import os
import sys
import json
import argparse
import warnings
import numpy as np

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from models import get_efficientnetv2_s, get_convnext_tiny, get_mobilevit

OUT_DIR = os.path.join(config.RESULTS_DIR, "testing", "explainability")
os.makedirs(OUT_DIR, exist_ok=True)

INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

DISPLAY_TRANSFORM = transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.ToTensor(),
])

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


def load_model(model_key, device):
    reg = MODELS_REGISTRY[model_key]
    model = reg["factory_fn"](num_classes=config.NUM_CLASSES, pretrained=False).to(device)
    ckpt_path = reg["checkpoint"]
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def get_target_layer(model, model_key):
    """
    Returns the best convolutional target layer for Grad-CAM.
    Documented per architecture.
    """
    if model_key == "efficientnetv2_s":
        # EfficientNetV2-S: last block of the features sequence
        return model.features[-1]

    elif model_key == "convnext_tiny":
        # ConvNeXt-Tiny: last ConvNeXt stage block
        return model.features[-1][-1]

    elif model_key == "mobilevit":
        # MobileViT (timm wrapper): last stage of backbone
        # The backbone is model.backbone for the MobileViTWrapper
        backbone = getattr(model, "backbone", model)
        # timm MobileViT has 'stages' attribute
        stages = getattr(backbone, "stages", None)
        if stages is not None:
            return stages[-1]
        # Fallback: last named module with 'conv' in name
        conv_modules = [(n, m) for n, m in backbone.named_modules()
                        if isinstance(m, torch.nn.Conv2d)]
        if conv_modules:
            return conv_modules[-1][1]
        raise RuntimeError("Could not identify target layer for MobileViT")

    else:
        raise ValueError(f"Unknown model key: {model_key}")


class GradCAM:
    """
    Minimal hook-based Grad-CAM implementation.
    Compatible with any conv layer in the model graph.
    Does NOT update model parameters.
    """

    def __init__(self, model, target_layer):
        self.model        = model
        self.target_layer = target_layer
        self.activations  = None
        self.gradients    = None
        self._hooks       = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        h1 = self.target_layer.register_forward_hook(forward_hook)
        h2 = self.target_layer.register_full_backward_hook(backward_hook)
        self._hooks.extend([h1, h2])

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()

    def generate(self, input_tensor, target_class_idx):
        """
        Run a forward+backward pass and return a (H, W) heatmap in [0,1].
        Model parameters are NOT updated (no optimiser step).
        """
        # Forward pass (gradients needed here, not using no_grad)
        self.model.zero_grad()
        output = self.model(input_tensor)
        score  = output[0, target_class_idx]
        score.backward()

        # Pool gradients over spatial dims
        grads = self.gradients   # (1, C, H, W)
        acts  = self.activations # (1, C, H, W)

        if grads is None or acts is None:
            raise RuntimeError(
                "Grad-CAM: no gradients or activations captured. "
                "The target layer may not produce spatial feature maps "
                "(e.g. pure attention or non-conv layer)."
            )

        # Handle 4D conv maps
        if grads.dim() == 4:
            weights = grads.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
            cam     = (weights * acts).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        elif grads.dim() == 3:
            # (1, T, C) -- transformer-style; less spatially precise
            warnings.warn(
                "Grad-CAM target layer produced 3D activations (transformer-style). "
                "Heatmap will be less spatially precise than for CNN layers."
            )
            weights = grads.mean(dim=1, keepdim=True)
            cam     = (weights * acts).sum(dim=-1, keepdim=True)
            # Reshape to square
            sq = int(cam.shape[1] ** 0.5)
            cam = cam[:, :sq*sq].view(1, 1, sq, sq)
        else:
            raise RuntimeError(f"Unexpected activation shape: {grads.shape}")

        cam = F.relu(cam)
        cam = cam.squeeze()

        # Normalise to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if (cam_max - cam_min) > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)

        return cam.cpu().numpy()


def make_overlay(original_pil, heatmap_np, alpha=0.5):
    """
    Overlay a (H, W) heatmap on the original PIL image.
    Returns a PIL.Image (RGB).
    """
    import cv2

    h, w = original_pil.size[1], original_pil.size[0]
    heatmap_resized = cv2.resize(heatmap_np, (w, h))
    heatmap_uint8   = np.uint8(255 * heatmap_resized)
    heatmap_color   = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_rgb     = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    orig_np = np.array(original_pil)
    overlay = np.uint8(alpha * heatmap_rgb + (1 - alpha) * orig_np)
    return Image.fromarray(overlay)


def explain_image(model_key, image_path, device, target_class_name=None):
    """
    Generate Grad-CAM explanation for one image + one model.
    Returns dict with paths and metadata.
    """
    reg          = MODELS_REGISTRY[model_key]
    display_name = reg["display_name"]

    # Load image
    original_pil = Image.open(image_path).convert("RGB")
    input_tensor = INFERENCE_TRANSFORM(original_pil).unsqueeze(0).to(device)

    # Load model
    model = load_model(model_key, device)

    # Forward for prediction (no grad)
    with torch.no_grad():
        logits = model(input_tensor)
        probs  = F.softmax(logits, dim=1)[0]

    confidence, pred_idx = torch.max(probs, dim=0)
    pred_class   = config.CLASS_NAMES[pred_idx.item()]
    confidence_f = confidence.item()

    # Target class for Grad-CAM
    if target_class_name is not None:
        if target_class_name not in config.CLASS_TO_IDX:
            raise ValueError(
                f"Unknown target class '{target_class_name}'. "
                f"Valid classes: {config.CLASS_NAMES}"
            )
        target_idx = config.CLASS_TO_IDX[target_class_name]
        explain_class = target_class_name
    else:
        target_idx    = pred_idx.item()
        explain_class = pred_class

    # Grad-CAM
    try:
        target_layer = get_target_layer(model, model_key)
        gcam         = GradCAM(model, target_layer)
        heatmap      = gcam.generate(input_tensor, target_idx)
        gcam.remove_hooks()
        method_note  = "Grad-CAM (hook-based)"
    except Exception as exc:
        warnings.warn(
            f"Grad-CAM failed for {model_key}: {exc}. "
            "No heatmap will be saved for this model."
        )
        return {
            "model_key":     model_key,
            "model_name":    display_name,
            "image_path":    image_path,
            "predicted_class": pred_class,
            "confidence":    round(confidence_f, 4),
            "explain_class": explain_class,
            "status":        f"FAILED: {exc}",
            "heatmap_path":  None,
            "overlay_path":  None,
        }

    # Build output paths
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    orig_path    = os.path.join(OUT_DIR, f"{base_name}_{model_key}_original.png")
    heatmap_path = os.path.join(OUT_DIR, f"{base_name}_{model_key}_heatmap.png")
    overlay_path = os.path.join(OUT_DIR, f"{base_name}_{model_key}_overlay.png")

    # Save original (resized)
    orig_resized = original_pil.resize((config.IMAGE_SIZE, config.IMAGE_SIZE))
    orig_resized.save(orig_path)

    # Save heatmap
    plt.figure(figsize=(5, 5))
    plt.imshow(heatmap, cmap="jet", vmin=0, vmax=1)
    plt.colorbar(label="Activation intensity")
    plt.title(
        f"Grad-CAM -- {display_name}\n"
        f"Explaining class: {explain_class}\n"
        "[Model explanation -- NOT proof of correctness]",
        fontsize=9,
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=150, bbox_inches="tight")
    plt.close()

    # Save overlay
    overlay_pil = make_overlay(orig_resized, heatmap, alpha=0.5)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    axes[0].imshow(orig_resized)
    axes[0].set_title("Original Image", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title(f"Grad-CAM Heatmap\n(target: {explain_class})", fontsize=10)
    axes[1].axis("off")

    axes[2].imshow(overlay_pil)
    axes[2].set_title(
        f"Overlay\nPredicted: {pred_class} ({confidence_f*100:.1f}%)\n"
        "[Explanation -- not certification]",
        fontsize=9,
    )
    axes[2].axis("off")

    fig.suptitle(
        f"FreshGuard AI -- Grad-CAM Explainability | {display_name}",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(overlay_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  [OK] {display_name}: heatmap + overlay saved -> {OUT_DIR}")

    return {
        "model_key":       model_key,
        "model_name":      display_name,
        "image_path":      image_path,
        "predicted_class": pred_class,
        "confidence":      round(confidence_f, 4),
        "explain_class":   explain_class,
        "method":          method_note,
        "status":          "OK",
        "original_path":   orig_path,
        "heatmap_path":    heatmap_path,
        "overlay_path":    overlay_path,
    }


def parse_args():
    p = argparse.ArgumentParser(
        description="FreshGuard AI -- Grad-CAM Explainability"
    )
    p.add_argument("--image", required=True, help="Path to the fruit image")
    p.add_argument(
        "--model", default="efficientnetv2_s",
        choices=list(MODELS_REGISTRY.keys()) + ["all"],
    )
    p.add_argument(
        "--target-class", default=None,
        choices=config.CLASS_NAMES,
        help="Explain for a specific class instead of the predicted class",
    )
    p.add_argument("--output-dir", default=OUT_DIR)
    return p.parse_args()


def main():
    args   = parse_args()
    global OUT_DIR
    OUT_DIR = args.output_dir
    os.makedirs(OUT_DIR, exist_ok=True)

    device = config.DEVICE

    if not os.path.exists(args.image):
        print(f"[ERROR] Image not found: {args.image}")
        sys.exit(1)

    model_keys = list(MODELS_REGISTRY.keys()) if args.model == "all" else [args.model]

    records = []
    print(f"\nGenerating Grad-CAM explanations -> {OUT_DIR}")
    for mk in model_keys:
        try:
            rec = explain_image(mk, args.image, device, args.target_class)
            records.append(rec)
        except Exception as exc:
            print(f"  [ERROR] {mk}: {exc}")
            import traceback; traceback.print_exc()

    # Save CSV log
    if records:
        import pandas as pd
        csv_path = os.path.join(OUT_DIR, "explainability_log.csv")
        pd.DataFrame(records).to_csv(csv_path, index=False, mode="a",
                                      header=not os.path.exists(csv_path))
        print(f"\n[OK] Log saved: {csv_path}")

    return records


if __name__ == "__main__":
    main()
