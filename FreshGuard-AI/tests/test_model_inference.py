"""
Unit & Integration Tests for Model Inference and Checkpoint Integrity.
"""
import os
import pytest
import torch
import torch.nn.functional as F

import config
from dataset import NUM_CLASSES, CLASS_NAMES, get_eval_transforms
from models.efficientnetv2_s import get_efficientnetv2_s
from models.convnext_tiny import get_convnext_tiny
from models.mobilevit import get_mobilevit_s

MODELS_CONFIG = [
    ('efficientnetv2_s', get_efficientnetv2_s, 'efficientnetv2_s_best.pth'),
    ('convnext_tiny', get_convnext_tiny, 'convnext_tiny_best.pth'),
    ('mobilevit', get_mobilevit_s, 'mobilevit_best.pth'),
]

@pytest.mark.parametrize("model_name,model_fn,ckpt_file", MODELS_CONFIG)
def test_model_architecture_output_shape(model_name, model_fn, ckpt_file):
    """Checks that model instances produce [batch_size, 6] outputs for dummy input."""
    model = model_fn(num_classes=NUM_CLASSES, pretrained=False)
    model.eval()

    dummy_input = torch.randn(2, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
    with torch.no_grad():
        out = model(dummy_input)

    assert out.shape == (2, NUM_CLASSES), f"{model_name} expected (2, {NUM_CLASSES}) but got {out.shape}"
    assert not torch.isnan(out).any(), f"{model_name} produced NaN outputs"
    assert not torch.isinf(out).any(), f"{model_name} produced Inf outputs"

@pytest.mark.parametrize("model_name,model_fn,ckpt_file", MODELS_CONFIG)
def test_checkpoint_loading_and_inference(model_name, model_fn, ckpt_file, sample_image):
    """Verifies that trained checkpoints exist, load cleanly, and produce valid probabilities."""
    ckpt_path = os.path.join(config.CHECKPOINTS_DIR, ckpt_file)
    assert os.path.exists(ckpt_path), f"Checkpoint missing: {ckpt_path}"

    model = model_fn(num_classes=NUM_CLASSES, pretrained=False)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
    model.load_state_dict(state_dict)
    model.eval()

    tf = get_eval_transforms()
    inp = tf(sample_image).unsqueeze(0)

    with torch.no_grad():
        logits = model(inp)
        probs = F.softmax(logits, dim=1).numpy()[0]

    assert len(probs) == 6
    assert abs(sum(probs) - 1.0) < 1e-4, f"Probabilities do not sum to 1.0: {sum(probs)}"
    assert (probs >= 0.0).all() and (probs <= 1.0).all(), "Invalid probability range"

def test_confidence_threshold_abstention():
    """Tests the uncertainty threshold logic (confidence < 0.60 results in UNCERTAIN)."""
    # Case 1: High confidence (should not abstain)
    high_conf_probs = [0.05, 0.05, 0.80, 0.05, 0.02, 0.03]
    max_conf = max(high_conf_probs)
    is_uncertain = max_conf < config.CONFIDENCE_THRESHOLD
    assert not is_uncertain
    assert max_conf == 0.80

    # Case 2: Low confidence (should abstain as UNCERTAIN)
    low_conf_probs = [0.20, 0.25, 0.15, 0.15, 0.10, 0.15]
    max_conf = max(low_conf_probs)
    is_uncertain = max_conf < config.CONFIDENCE_THRESHOLD
    assert is_uncertain
    assert max_conf < 0.60
