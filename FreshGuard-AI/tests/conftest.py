"""
Pytest configuration and fixtures for FreshGuard AI tests.
"""
import os
import sys
import io
import pytest
import torch
import numpy as np
from PIL import Image

# Ensure project root is in path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import config
from dataset import CLASS_NAMES, NUM_CLASSES, INVERSE_CLASS_MAPPING

@pytest.fixture(scope="session")
def project_root():
    return ROOT_DIR

@pytest.fixture(scope="session")
def class_names():
    return CLASS_NAMES

@pytest.fixture(scope="session")
def num_classes():
    return NUM_CLASSES

@pytest.fixture(scope="session")
def sample_image():
    """Generates an in-memory sample RGB image for testing."""
    img = Image.new('RGB', (256, 256), color=(200, 100, 50))
    return img

@pytest.fixture(scope="session")
def sample_image_bytes(sample_image):
    """Provides sample image as byte stream."""
    buf = io.BytesIO()
    sample_image.save(buf, format='JPEG')
    buf.seek(0)
    return buf.getvalue()

@pytest.fixture(scope="session")
def test_image_path():
    """Finds an existing sample image from the test or val set."""
    for split_dir in ['data/test', 'data/val', 'data/train']:
        p = os.path.join(ROOT_DIR, split_dir)
        if os.path.exists(p):
            for root, _, files in os.walk(p):
                for f in files:
                    if f.endswith(('.jpg', '.jpeg', '.png')):
                        return os.path.join(root, f)
    # Fallback: create temporary image
    tmp_path = os.path.join(ROOT_DIR, 'results', 'sample_test_img.jpg')
    os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
    img = Image.new('RGB', (256, 256), color=(180, 120, 60))
    img.save(tmp_path)
    return tmp_path
