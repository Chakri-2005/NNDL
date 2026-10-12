"""
FreshGuard AI -- Individual Image Prediction
============================================
Loads a trained checkpoint and classifies a single fruit image.

Usage
-----
    python predict.py --image path/to/fruit.jpg
    python predict.py --image path/to/fruit.jpg --model efficientnetv2_s
    python predict.py --image path/to/fruit.jpg --model all
    python predict.py --image path/to/fruit.jpg --model convnext_tiny --no-threshold

Notes
-----
* Checkpoint weights are NEVER modified (eval mode + no_grad).
* The same inference preprocessing used in training is applied (val/test transforms).
* Returns UNCERTAIN when max confidence < CONFIDENCE_THRESHOLD (0.60 from config.py).
* When --model all is specified, all three models are run and compared.
"""

import os
import sys
import json
import time
import argparse
import warnings

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, UnidentifiedImageError

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from models import get_efficientnetv2_s, get_convnext_tiny, get_mobilevit

# ── Supported formats ──────────────────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".gif"}

# ── Model registry (same as model_manager.py) ─────────────────────────────────
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

# ── Inference preprocessing (identical to val/test in dataset.py) ──────────────
INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def validate_image_path(image_path: str) -> str:
    """Validate that image_path is a readable, supported image file."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    if not os.path.isfile(image_path):
        raise ValueError(f"Path is not a file: {image_path}")
    ext = os.path.splitext(image_path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported image format '{ext}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    return image_path


def load_image(image_path: str) -> Image.Image:
    """Load and validate a PIL image; raise informative errors on failure."""
    try:
        img = Image.open(image_path)
        img.verify()           # detect truncated/corrupt images
    except UnidentifiedImageError:
        raise ValueError(f"File is not a valid image: {image_path}")
    except Exception as exc:
        raise ValueError(f"Failed to read image '{image_path}': {exc}")

    # Re-open after verify (verify() consumes the stream)
    img = Image.open(image_path).convert("RGB")
    return img


def load_model_for_inference(model_key: str, device: torch.device):
    """Load checkpoint in eval mode. Weights are NOT modified."""
    reg = MODELS_REGISTRY[model_key]
    model = reg["factory_fn"](num_classes=config.NUM_CLASSES, pretrained=False).to(device)
    ckpt_path = reg["checkpoint"]
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            "Train the model first or verify the checkpoint path in config.py."
        )
    try:
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
    except Exception as exc:
        raise RuntimeError(f"Failed to load checkpoint '{ckpt_path}': {exc}")
    model.eval()
    return model


def predict_single(
    model_key: str,
    image: Image.Image,
    device: torch.device,
    apply_threshold: bool = True,
) -> dict:
    """
    Run inference on a single PIL image.

    Returns a dict with:
        model_key, model_name, predicted_class, fruit, freshness,
        confidence, is_uncertain, probabilities (all 6 classes), inference_ms
    """
    reg    = MODELS_REGISTRY[model_key]
    model  = load_model_for_inference(model_key, device)

    tensor = INFERENCE_TRANSFORM(image).unsqueeze(0).to(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1)[0]
    inference_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    confidence, pred_idx = torch.max(probs, dim=0)
    confidence_f = confidence.item()
    pred_label   = pred_idx.item()
    pred_class   = config.CLASS_NAMES[pred_label]

    threshold    = config.CONFIDENCE_THRESHOLD
    is_uncertain = confidence_f < threshold if apply_threshold else False
    display_class = "UNCERTAIN" if is_uncertain else pred_class

    fruit, freshness = pred_class.split("_")

    class_probs = {
        config.CLASS_NAMES[i]: round(probs[i].item(), 4)
        for i in range(config.NUM_CLASSES)
    }

    return {
        "model_key":       model_key,
        "model_name":      reg["display_name"],
        "predicted_class": display_class,
        "raw_class":       pred_class,
        "fruit":           fruit,
        "freshness":       freshness,
        "confidence":      round(confidence_f, 4),
        "confidence_pct":  round(confidence_f * 100, 2),
        "is_uncertain":    is_uncertain,
        "threshold":       threshold,
        "probabilities":   class_probs,
        "inference_ms":    inference_ms,
    }


def format_result(result: dict, image_path: str) -> str:
    """Pretty-print a prediction result."""
    lines = [
        f"\n{'='*55}",
        f"  Image      : {image_path}",
        f"  Model      : {result['model_name']}",
        f"  Prediction : {result['predicted_class']}",
    ]
    if result["is_uncertain"]:
        lines.append(
            f"  [WARNING] Confidence {result['confidence_pct']:.1f}% is below "
            f"threshold ({result['threshold']*100:.0f}%). "
            "A clearer image may give a more reliable prediction."
        )
    else:
        lines.append(f"  Fruit      : {result['fruit']}")
        lines.append(f"  Freshness  : {result['freshness']}")
        lines.append(f"  Confidence : {result['confidence_pct']:.2f}%")
    lines.append(f"  Latency    : {result['inference_ms']} ms")
    lines.append(f"\n  Probabilities (all 6 classes):")
    sorted_probs = sorted(result["probabilities"].items(), key=lambda x: -x[1])
    for cls, prob in sorted_probs:
        bar = "#" * int(prob * 30)
        lines.append(f"    {cls:<20} {prob*100:5.1f}%  |{bar}")
    lines.append("="*55)
    return "\n".join(lines)


def parse_args():
    p = argparse.ArgumentParser(
        description="FreshGuard AI -- Individual Image Prediction"
    )
    p.add_argument(
        "--image", required=True,
        help="Path to the fruit image to classify",
    )
    p.add_argument(
        "--model", default="efficientnetv2_s",
        choices=list(MODELS_REGISTRY.keys()) + ["all"],
        help=(
            "Model to use for prediction. "
            "Use 'all' to compare all three models. "
            "(default: efficientnetv2_s)"
        ),
    )
    p.add_argument(
        "--no-threshold", action="store_true",
        help="Disable the confidence threshold (always return a class label)",
    )
    p.add_argument(
        "--json", action="store_true",
        help="Output results as JSON",
    )
    return p.parse_args()


def main():
    args    = parse_args()
    device  = config.DEVICE

    # Validate image
    try:
        image_path = validate_image_path(args.image)
        image      = load_image(image_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)

    apply_threshold = not args.no_threshold
    model_keys = list(MODELS_REGISTRY.keys()) if args.model == "all" else [args.model]

    results = []
    for model_key in model_keys:
        try:
            result = predict_single(model_key, image, device, apply_threshold)
            results.append(result)
        except (FileNotFoundError, RuntimeError) as exc:
            print(f"[ERROR] {model_key}: {exc}")
            sys.exit(1)
        except Exception as exc:
            print(f"[ERROR] Inference failed for {model_key}: {exc}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    if args.json:
        output = results[0] if len(results) == 1 else {"results": results}
        print(json.dumps(output, indent=2))
    else:
        for result in results:
            print(format_result(result, image_path))

    return results


if __name__ == "__main__":
    main()
