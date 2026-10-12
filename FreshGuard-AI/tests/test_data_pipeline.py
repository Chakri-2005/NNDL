"""
Unit & Integration Tests for Data Pipeline.
"""
import os
import pytest
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms

import config
from dataset import (
    CLASS_NAMES, NUM_CLASSES, CLASS_TO_IDX, INVERSE_CLASS_MAPPING,
    get_train_transforms, get_eval_transforms, FreshGuardDataset
)

def test_class_names_and_indices():
    """Validates that exactly 6 classes are defined with valid names and 1-to-1 mapping."""
    expected_classes = [
        'Apple_Fresh', 'Apple_Rotten',
        'Banana_Fresh', 'Banana_Rotten',
        'Orange_Fresh', 'Orange_Rotten'
    ]
    assert NUM_CLASSES == 6
    assert set(CLASS_NAMES) == set(expected_classes)
    assert len(CLASS_TO_IDX) == 6
    assert len(INVERSE_CLASS_MAPPING) == 6

    for name, idx in CLASS_TO_IDX.items():
        assert 0 <= idx < 6
        assert INVERSE_CLASS_MAPPING[idx] == name

def test_eval_transforms_output_properties(sample_image):
    """Checks eval transforms tensor output dimensions, type, and normalization."""
    tf = get_eval_transforms()
    tensor = tf(sample_image)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, config.IMAGE_SIZE, config.IMAGE_SIZE)
    assert tensor.dtype == torch.float32
    assert not torch.isnan(tensor).any()
    assert not torch.isinf(tensor).any()

def test_train_transforms_data_augmentation(sample_image):
    """Verifies train augmentation produces valid non-empty tensors."""
    tf = get_train_transforms()
    tensors = [tf(sample_image) for _ in range(3)]

    for t in tensors:
        assert isinstance(t, torch.Tensor)
        assert t.shape == (3, config.IMAGE_SIZE, config.IMAGE_SIZE)
        assert not torch.isnan(t).any()

def test_missing_image_handling():
    """Checks that accessing a non-existent image fails gracefully."""
    non_existent_path = "non_existent_image_12345.jpg"
    with pytest.raises(Exception):
        Image.open(non_existent_path)

def test_dataset_splits_exist_and_valid():
    """Checks that train, val, and test split CSVs exist and contain valid headers and labels."""
    splits = {
        'train': os.path.join(config.SPLITS_DIR, 'train.csv'),
        'val': os.path.join(config.SPLITS_DIR, 'val.csv'),
        'test': os.path.join(config.SPLITS_DIR, 'test.csv'),
    }

    for name, path in splits.items():
        assert os.path.exists(path), f"Split file missing: {path}"
        df = pd.read_csv(path)
        assert len(df) > 0, f"Split {name} is empty!"
        assert 'filepath' in df.columns, f"filepath column missing in {name}"
        assert 'label' in df.columns, f"label column missing in {name}"
        assert df['label'].isin(list(range(6))).all(), f"Invalid label index found in {name}"

def test_test_split_isolation():
    """Verifies that the test split does not share filepaths with train or val."""
    df_train = pd.read_csv(os.path.join(config.SPLITS_DIR, 'train.csv'))
    df_val = pd.read_csv(os.path.join(config.SPLITS_DIR, 'val.csv'))
    df_test = pd.read_csv(os.path.join(config.SPLITS_DIR, 'test.csv'))

    train_files = set(df_train['filepath'])
    val_files = set(df_val['filepath'])
    test_files = set(df_test['filepath'])

    overlap_train_test = train_files.intersection(test_files)
    overlap_val_test = val_files.intersection(test_files)

    assert len(overlap_train_test) == 0, f"Data leakage detected between train and test: {overlap_train_test}"
    assert len(overlap_val_test) == 0, f"Data leakage detected between val and test: {overlap_val_test}"
