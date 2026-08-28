"""
data_preprocessing.py
----------------------
Takes the RAW real dataset (one folder per class, e.g. data/raw/Stop/*.jpg)
and produces a clean train/val/test split on disk at data/processed/.

WHY split on disk instead of in-memory?
Keras' image_dataset_from_directory / ImageDataGenerator can stream
directly from folders, which is far more memory-efficient than loading
every image into a NumPy array at once — important on a normal laptop.

This script does NOT invent or synthesize any images. It only copies
real files that already exist under data/raw/<class_name>/ into a
train/val/test folder layout, using a stratified random split so every
class is represented proportionally in each split.

Usage:
    python -m src.data_preprocessing
"""

import os
import shutil
import random

from src import config
from src.utils import ensure_dir


def _list_real_images(class_dir):
    """Return only real image files inside a class folder (ignore junk files)."""
    valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    return [
        f for f in os.listdir(class_dir)
        if f.lower().endswith(valid_ext) and os.path.isfile(os.path.join(class_dir, f))
    ]


def verify_raw_dataset():
    """
    Sanity-check that the real dataset actually exists before we do anything.
    Raises a clear, helpful error instead of failing deep inside Keras later.
    """
    if not os.path.isdir(config.RAW_DATA_DIR):
        raise FileNotFoundError(
            f"Raw data folder not found: {config.RAW_DATA_DIR}\n"
            "Download a real Bangladeshi traffic sign dataset (e.g. a BTSRB / "
            "BDTSR style dataset from Kaggle/Mendeley — see README.md) and place "
            "it as: data/raw/<ClassName>/<image files>"
        )

    found_classes = sorted(
        d for d in os.listdir(config.RAW_DATA_DIR)
        if os.path.isdir(os.path.join(config.RAW_DATA_DIR, d))
    )
    if not found_classes:
        raise FileNotFoundError(
            f"No class folders found inside {config.RAW_DATA_DIR}. "
            "Expected one sub-folder per traffic sign class."
        )

    print(f"Found {len(found_classes)} class folders in data/raw/: {found_classes}")

    empty_classes = []
    for c in found_classes:
        n_imgs = len(_list_real_images(os.path.join(config.RAW_DATA_DIR, c)))
        print(f"  - {c}: {n_imgs} real images")
        if n_imgs == 0:
            empty_classes.append(c)

    if empty_classes:
        raise ValueError(
            f"These class folders contain zero real images: {empty_classes}. "
            "Remove empty folders or add real images before continuing."
        )

    return found_classes


def split_dataset(seed=config.RANDOM_SEED):
    """
    Stratified split of the REAL images in data/raw/ into
    data/processed/{train,val,test}/<class_name>/ folders.
    Copies files (does not modify data/raw/).
    """
    random.seed(seed)
    class_names = verify_raw_dataset()

    # Fresh processed folder each run so we never mix old + new splits
    if os.path.isdir(config.PROCESSED_DATA_DIR):
        shutil.rmtree(config.PROCESSED_DATA_DIR)

    for split_dir in (config.TRAIN_DIR, config.VAL_DIR, config.TEST_DIR):
        for class_name in class_names:
            ensure_dir(os.path.join(split_dir, class_name))

    summary = {}
    for class_name in class_names:
        class_dir = os.path.join(config.RAW_DATA_DIR, class_name)
        images = _list_real_images(class_dir)
        random.shuffle(images)

        n_total = len(images)
        n_train = int(n_total * config.TRAIN_SPLIT)
        n_val = int(n_total * config.VAL_SPLIT)
        # test gets the remainder, so rounding never drops an image
        train_files = images[:n_train]
        val_files = images[n_train:n_train + n_val]
        test_files = images[n_train + n_val:]

        for files, split_dir in (
            (train_files, config.TRAIN_DIR),
            (val_files, config.VAL_DIR),
            (test_files, config.TEST_DIR),
        ):
            dest_dir = os.path.join(split_dir, class_name)
            for fname in files:
                shutil.copy2(os.path.join(class_dir, fname), os.path.join(dest_dir, fname))

        summary[class_name] = {
            "train": len(train_files), "val": len(val_files), "test": len(test_files)
        }
        print(f"{class_name:25s} -> train={len(train_files):4d}  "
              f"val={len(val_files):4d}  test={len(test_files):4d}")

    return summary


if __name__ == "__main__":
    print("=" * 60)
    print("Splitting REAL dataset into train / val / test ...")
    print("=" * 60)
    split_dataset()
    print("\nDone. Processed data is at:", config.PROCESSED_DATA_DIR)
