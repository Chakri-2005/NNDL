"""
FreshGuard AI — Centralized Configuration Management
Single source of truth for paths, hyperparameters, class labels, and random seeds.
"""
import os
import torch

# ── Project Roots ────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RAW_DATASET_DIR = os.path.join(
    BASE_DIR, 
    "Top Indian Fruits Dataset_Clean Images/FruitDataSet of 256x256 Size_12000 Images"
)
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Subdirectories for outputs
SPLITS_DIR = os.path.join(RESULTS_DIR, "splits")
CHECKPOINTS_DIR = os.path.join(RESULTS_DIR, "checkpoints")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
PER_CLASS_DIR = os.path.join(RESULTS_DIR, "per_class")
REVIEW_DIR = os.path.join(BASE_DIR, "review")

# Ensure required output directories exist
for path in [SPLITS_DIR, CHECKPOINTS_DIR, METRICS_DIR, PLOTS_DIR, PER_CLASS_DIR, REVIEW_DIR]:
    os.makedirs(path, exist_ok=True)

# ── Class Definitions & Standardization ──────────────────────────────────────
CLASS_NAMES = [
    "Apple_Fresh",
    "Apple_Rotten",
    "Banana_Fresh",
    "Banana_Rotten",
    "Orange_Fresh",
    "Orange_Rotten"
]

NUM_CLASSES = len(CLASS_NAMES)
CLASS_TO_IDX = {cls_name: i for i, cls_name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {i: cls_name for i, cls_name in enumerate(CLASS_NAMES)}

# Mapping from raw dataset directory names to standardized target class names
RAW_CLASS_MAPPING = {
    "Apple_Good": "Apple_Fresh",
    "Apple_Bad": "Apple_Rotten",
    "Banana_Good": "Banana_Fresh",
    "Banana_Bad": "Banana_Rotten",
    "Orange_Good": "Orange_Fresh",
    "Orange_Bad": "Orange_Rotten"
}

# ── Model & Training Hyperparameters ─────────────────────────────────────────
IMAGE_SIZE = 224  # Standard resolution across EfficientNetV2-S, ConvNeXt-Tiny, MobileViT
SEED = 42

BATCH_SIZE = 32
NUM_WORKERS = 2

# Two-Stage Fine-Tuning Strategy (Optimal 2+2 epochs for transfer learning speed and quality)
STAGE1_EPOCHS = 2      # Header warm-up with frozen backbone
STAGE2_EPOCHS = 2      # End-to-end fine-tuning with un-frozen upper layers

STAGE1_LR = 1e-3       # Classifier head learning rate
STAGE2_LR = 1e-4       # Unfrozen backbone fine-tuning learning rate
WEIGHT_DECAY = 1e-4

CONFIDENCE_THRESHOLD = 0.60  # Abstention threshold for out-of-distribution rejection

# ── Device Setup ─────────────────────────────────────────────────────────────
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = get_device()
