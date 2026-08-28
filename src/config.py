"""
config.py
---------
Single source of truth for paths and hyperparameters.
Every other script imports from here instead of hard-coding values,
so you only ever change settings in ONE place.
"""

import os

# ----------------------------------------------------------------------
# PATHS
# ----------------------------------------------------------------------
# Root of the project (one level above src/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Put your downloaded real dataset here, ONE SUB-FOLDER PER CLASS, e.g.:
#   data/raw/Stop/img001.jpg
#   data/raw/Stop/img002.jpg
#   data/raw/No_Entry/img001.jpg
#   data/raw/Speed_Limit/img001.jpg
#   ...
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")

# After running src/data_preprocessing.py, split data lands here as
# data/processed/train, data/processed/val, data/processed/test
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
TRAIN_DIR = os.path.join(PROCESSED_DATA_DIR, "train")
VAL_DIR = os.path.join(PROCESSED_DATA_DIR, "val")
TEST_DIR = os.path.join(PROCESSED_DATA_DIR, "test")

MODELS_DIR = os.path.join(BASE_DIR, "models")
CUSTOM_CNN_PATH = os.path.join(MODELS_DIR, "custom_cnn_best.keras")
MOBILENET_PATH = os.path.join(MODELS_DIR, "mobilenetv2_best.keras")
EFFICIENTNET_PATH = os.path.join(MODELS_DIR, "efficientnetb0_best.keras")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.keras")
CLASS_INDICES_PATH = os.path.join(MODELS_DIR, "class_indices.json")
HISTORY_DIR = os.path.join(MODELS_DIR, "history")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

# ----------------------------------------------------------------------
# DATA / IMAGE SETTINGS
# ----------------------------------------------------------------------
IMG_SIZE = (64, 64)      # (height, width) — change to (128, 128) if your GPU/CPU can handle it
IMG_CHANNELS = 3
INPUT_SHAPE = (IMG_SIZE[0], IMG_SIZE[1], IMG_CHANNELS)

# Real Bangladeshi traffic sign class names.
#
# AUTO-DETECTED from the sub-folder names inside data/raw/ — you do NOT
# need to type these by hand. Whatever folders exist under data/raw/
# (e.g. after running src/convert_yolo_dataset.py) automatically become
# the class list, sorted alphabetically so the order is always consistent.
#
# If data/raw/ is empty (e.g. fresh clone, before you've added a dataset),
# this falls back to a placeholder 6-class list so imports don't crash —
# that fallback list is NOT used for real training.
def _detect_class_names():
    if os.path.isdir(RAW_DATA_DIR):
        folders = sorted(
            d for d in os.listdir(RAW_DATA_DIR)
            if os.path.isdir(os.path.join(RAW_DATA_DIR, d)) and not d.startswith(".")
        )
        if folders:
            return folders
    # Fallback placeholder — only used when data/raw/ has no class folders yet
    return [
        "Stop",
        "No_Entry",
        "Speed_Limit",
        "Turn_Left",
        "Turn_Right",
        "Pedestrian_Crossing",
    ]


CLASS_NAMES = _detect_class_names()
NUM_CLASSES = len(CLASS_NAMES)

# Train / Val / Test split ratios (must sum to 1.0)
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42

# ----------------------------------------------------------------------
# TRAINING HYPERPARAMETERS
# ----------------------------------------------------------------------
BATCH_SIZE = 32
EPOCHS = 50                 # EarlyStopping will usually stop well before this
LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 8
REDUCE_LR_PATIENCE = 4

# Confidence below this -> prediction is flagged "Uncertain" in the app
CONFIDENCE_THRESHOLD = 0.60

# Grad-CAM: name of the last conv layer differs per architecture.
# custom CNN uses "conv2d_last"; MobileNetV2 uses "Conv_1"; EfficientNetB0 uses "top_conv"
GRADCAM_LAYER_NAMES = {
    "custom_cnn": "conv2d_last",
    "mobilenetv2": "Conv_1",
    "efficientnetb0": "top_conv",
}
