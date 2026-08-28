"""
convert_yolo_dataset.py
------------------------
Converts a Roboflow-style YOLO object-detection export (train/valid/test,
each with images/ + labels/, plus a data.yaml listing class names) into the
simple "one folder per class" layout our classification pipeline expects
at data/raw/<ClassName>/<image>.jpg.

WHAT IT DOES (no synthetic data — only crops real, already-photographed
signs out of real, already-photographed images):
1. Reads class names from data.yaml
2. Walks through train/ + valid/ + test/ (we merge them — our own
   src/data_preprocessing.py will re-split into train/val/test later)
3. For every image, reads its matching YOLO label .txt file
4. For every bounding box in that file, crops just that sign out of the
   real photo and saves it into data/raw/<ClassName>/<unique_name>.jpg

Usage:
    python -m src.convert_yolo_dataset --source "/path/to/Annotated Images Withoug Augmentation"

After this finishes, data/raw/ will be ready for:
    python -m src.data_preprocessing
    python -m src.train
"""

import os
import argparse
import uuid

import yaml
from PIL import Image

from src import config
from src.utils import ensure_dir


def load_class_names(source_dir):
    """Read class id -> class name mapping from data.yaml (Roboflow format)."""
    yaml_path = os.path.join(source_dir, "data.yaml")
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(
            f"data.yaml not found in {source_dir}. This file should list the "
            "class names Roboflow used when you exported the dataset."
        )
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    names = data.get("names")
    if names is None:
        raise ValueError(f"'names' key not found in {yaml_path}")

    # Roboflow's data.yaml sometimes stores names as a list, sometimes as a dict
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names, key=lambda k: int(k))]

    print(f"Found {len(names)} classes in data.yaml: {names}")
    return names


def _sanitize_folder_name(name):
    """Make a class name filesystem-safe (spaces -> underscores, etc.)."""
    return name.strip().replace(" ", "_").replace("/", "-")


def convert_split(source_dir, split_name, class_names, padding=6):
    """
    Crop every labeled sign out of one split (train/valid/test) and save
    it into data/raw/<ClassName>/.
    `padding` adds a few pixels of margin around each crop so the sign's
    edge/border isn't cut off — helps the classifier see the full shape.
    """
    images_dir = os.path.join(source_dir, split_name, "images")
    labels_dir = os.path.join(source_dir, split_name, "labels")

    if not os.path.isdir(images_dir):
        print(f"Skipping '{split_name}' — folder not found: {images_dir}")
        return 0

    image_files = [
        f for f in os.listdir(images_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
    ]

    n_crops = 0
    n_skipped_no_label = 0

    for img_file in image_files:
        img_path = os.path.join(images_dir, img_file)
        label_file = os.path.splitext(img_file)[0] + ".txt"
        label_path = os.path.join(labels_dir, label_file)

        if not os.path.exists(label_path):
            n_skipped_no_label += 1
            continue

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"  Could not open {img_path}: {e}")
            continue

        img_w, img_h = image.size

        with open(label_path, "r") as f:
            lines = [ln.strip() for ln in f if ln.strip()]

        for line in lines:
            parts = line.split()
            if len(parts) < 5:
                continue
            class_id = int(float(parts[0]))
            x_center, y_center, box_w, box_h = map(float, parts[1:5])

            if class_id >= len(class_names):
                print(f"  Warning: class_id {class_id} outside known class list, skipping.")
                continue
            class_name = _sanitize_folder_name(class_names[class_id])

            # Convert normalized YOLO coords -> real pixel box
            cx, cy = x_center * img_w, y_center * img_h
            bw, bh = box_w * img_w, box_h * img_h
            left = max(0, int(cx - bw / 2) - padding)
            top = max(0, int(cy - bh / 2) - padding)
            right = min(img_w, int(cx + bw / 2) + padding)
            bottom = min(img_h, int(cy + bh / 2) + padding)

            if right <= left or bottom <= top:
                continue  # degenerate box, skip

            crop = image.crop((left, top, right, bottom))

            dest_dir = os.path.join(config.RAW_DATA_DIR, class_name)
            ensure_dir(dest_dir)
            out_name = f"{split_name}_{uuid.uuid4().hex[:10]}.jpg"
            crop.save(os.path.join(dest_dir, out_name), quality=95)
            n_crops += 1

    print(f"[{split_name}] real images seen: {len(image_files)}, "
          f"crops saved: {n_crops}, skipped (no label file): {n_skipped_no_label}")
    return n_crops


def main(source_dir):
    class_names = load_class_names(source_dir)
    ensure_dir(config.RAW_DATA_DIR)

    total = 0
    for split_name in ("train", "valid", "test"):
        total += convert_split(source_dir, split_name, class_names)

    print(f"\nDone. Total real sign crops saved: {total}")
    print(f"Saved under: {config.RAW_DATA_DIR}")
    print(
        "\nNext: open src/config.py and set CLASS_NAMES to match the folder "
        "names now inside data/raw/ (some classes may end up with very few "
        "images — you can drop those or collect more before training)."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        required=True,
        help="Path to the extracted 'Annotated Images Withoug Augmentation' folder "
             "(the one containing train/, valid/, test/, data.yaml)",
    )
    args = parser.parse_args()
    main(args.source)
