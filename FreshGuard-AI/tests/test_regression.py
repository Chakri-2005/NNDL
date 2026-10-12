"""
Regression Tests: Confirms consistent predictions on stable fixtures.
"""
import os
import pytest
import torch
import numpy as np
from PIL import Image

import config
from dataset import CLASS_NAMES, get_eval_transforms
from models.efficientnetv2_s import get_efficientnetv2_s

def test_prediction_reproducibility_on_fixed_seed(test_image_path):
    """Verifies that running inference multiple times on the same input yields identical outputs."""
    ckpt_path = os.path.join(config.CHECKPOINTS_DIR, 'efficientnetv2_s_best.pth')
    if not os.path.exists(ckpt_path):
        pytest.skip("Checkpoint not found for regression test")

    model = get_efficientnetv2_s(num_classes=6, pretrained=False)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
    model.load_state_dict(state_dict)
    model.eval()

    tf = get_eval_transforms()
    img = Image.open(test_image_path).convert('RGB')
    tensor = tf(img).unsqueeze(0)

    with torch.no_grad():
        out1 = model(tensor).numpy()
        out2 = model(tensor).numpy()

    # Exact deterministic equivalence
    np.testing.assert_allclose(out1, out2, atol=1e-6)

def test_no_grad_inference_does_not_modify_weights():
    """Verifies that inference runs without modifying any model weights."""
    model = get_efficientnetv2_s(num_classes=6, pretrained=False)
    model.eval()

    initial_weights = [p.clone() for p in model.parameters()]
    dummy_input = torch.randn(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)

    with torch.no_grad():
        _ = model(dummy_input)

    for p_init, p_after in zip(initial_weights, model.parameters()):
        assert torch.equal(p_init, p_after), "Model weights modified during inference!"
