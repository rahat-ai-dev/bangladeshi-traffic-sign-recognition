"""
train.py
--------
Trains the custom CNN and both transfer-learning models (MobileNetV2,
EfficientNetB0) on the REAL, pre-processed dataset, then picks whichever
model scores highest validation accuracy as the "best model" for the app.

Usage:
    python -m src.train                  # train all 3 models and compare
    python -m src.train --model custom_cnn   # train just one
"""

import os
import argparse
import json

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from src import config
from src.utils import ensure_dir, save_class_indices, save_history
from src.data_augmentation import get_datasets, compute_class_weights
from src.model_builder import MODEL_REGISTRY


MODEL_PATHS = {
    "custom_cnn": config.CUSTOM_CNN_PATH,
    "mobilenetv2": config.MOBILENET_PATH,
    "efficientnetb0": config.EFFICIENTNET_PATH,
}


def get_callbacks(checkpoint_path):
    """
    - EarlyStopping: stops training once validation loss stops improving,
      restoring the best epoch's weights. Prevents wasting time/overfitting.
    - ModelCheckpoint: saves ONLY the best-performing epoch to disk, so the
      final saved file is always the best version seen during training.
    - ReduceLROnPlateau: lowers the learning rate when progress stalls,
      helping the model fine-tune into a better minimum.
    """
    ensure_dir(os.path.dirname(checkpoint_path))
    return [
        EarlyStopping(
            monitor="val_loss",
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=config.REDUCE_LR_PATIENCE,
            min_lr=1e-6,
            verbose=1,
        ),
    ]


def train_one_model(model_name, train_ds, val_ds, class_weights):
    print(f"\n{'#' * 70}\n# Training: {model_name}\n{'#' * 70}")
    model = MODEL_REGISTRY[model_name]()
    model.summary()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.EPOCHS,
        callbacks=get_callbacks(MODEL_PATHS[model_name]),
        class_weight=class_weights,  # counter class imbalance
    )
    save_history(history.history, model_name)

    best_val_acc = max(history.history["val_accuracy"])
    print(f"{model_name}: best validation accuracy = {best_val_acc:.4f}")
    return best_val_acc


def main(models_to_train):
    ensure_dir(config.MODELS_DIR)

    print("Loading real train/val/test datasets from data/processed/ ...")
    train_ds, val_ds, test_ds = get_datasets()

    # Save the class name <-> index mapping now, so predict.py/app.py
    # can decode model outputs back into readable class names later.
    class_indices = {name: i for i, name in enumerate(config.CLASS_NAMES)}
    save_class_indices(class_indices)

    class_weights = compute_class_weights()

    results = {}
    for model_name in models_to_train:
        results[model_name] = train_one_model(model_name, train_ds, val_ds, class_weights)

    # Pick the best model by validation accuracy and copy it to best_model.keras
    best_name = max(results, key=results.get)
    print(f"\nBest model: {best_name} (val_accuracy={results[best_name]:.4f})")

    best_model = tf.keras.models.load_model(MODEL_PATHS[best_name])
    best_model.save(config.BEST_MODEL_PATH)

    with open(os.path.join(config.MODELS_DIR, "comparison_results.json"), "w") as f:
        json.dump({"results": results, "best_model": best_name}, f, indent=2)

    print(f"\nSaved best model to {config.BEST_MODEL_PATH}")
    print("Comparison results:", results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=list(MODEL_REGISTRY.keys()) + ["all"],
        default="all",
        help="Which model(s) to train.",
    )
    args = parser.parse_args()

    models_to_train = list(MODEL_REGISTRY.keys()) if args.model == "all" else [args.model]
    main(models_to_train)
