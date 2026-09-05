"""
evaluate.py
-----------
Evaluates a trained model on the REAL, held-out test set and produces:
  - Accuracy, Precision, Recall, F1-score (macro + weighted)
  - Full classification report (per class)
  - Confusion matrix (as a plot)
  - Training vs validation accuracy/loss curves (from saved history)

Usage:
    python -m src.evaluate --model best_model
    python -m src.evaluate --model custom_cnn
"""

import os
import argparse
import json

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix,
)

from src import config
from src.utils import ensure_dir, load_history, plot_training_curves, load_class_indices
from src.data_augmentation import get_datasets


MODEL_PATH_LOOKUP = {
    "custom_cnn": config.CUSTOM_CNN_PATH,
    "mobilenetv2": config.MOBILENET_PATH,
    "efficientnetb0": config.EFFICIENTNET_PATH,
    "best_model": config.BEST_MODEL_PATH,
}


def evaluate_model(model_name):
    ensure_dir(config.REPORTS_DIR)
    model_path = MODEL_PATH_LOOKUP[model_name]
    print(f"Loading model from {model_path}")
    model = tf.keras.models.load_model(model_path)

    _, _, test_ds = get_datasets()
    _, index_to_class = load_class_indices()
    class_names = [index_to_class[i] for i in range(len(index_to_class))]

    # --- Collect true labels and predictions over the whole real test set ---
    y_true, y_pred, y_prob = [], [], []
    for images, labels in test_ds:
        probs = model.predict(images, verbose=0)
        y_prob.extend(probs)
        y_pred.extend(np.argmax(probs, axis=1))
        y_true.extend(np.argmax(labels.numpy(), axis=1))

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    # --- Metrics ---
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    report = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0
    )

    print(f"\n{'=' * 60}\nEvaluation results for: {model_name}\n{'=' * 60}")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f} (weighted)")
    print(f"Recall   : {recall:.4f} (weighted)")
    print(f"F1-score : {f1:.4f} (weighted)")
    print("\nClassification report:\n", report)

    metrics = {
        "model": model_name,
        "accuracy": float(acc),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1),
    }
    with open(os.path.join(config.REPORTS_DIR, f"{model_name}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(config.REPORTS_DIR, f"{model_name}_classification_report.txt"), "w") as f:
        f.write(report)

    # --- Confusion matrix plot ---
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(f"Confusion Matrix — {model_name}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    cm_path = os.path.join(config.REPORTS_DIR, f"{model_name}_confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix to {cm_path}")

    # --- Training curves (if history was saved for this model) ---
    try:
        history = load_history(model_name)
        curve_path = os.path.join(config.REPORTS_DIR, f"{model_name}_training_curves.png")
        plot_training_curves(history, model_name, save_path=curve_path)
    except FileNotFoundError:
        print(f"No saved training history for '{model_name}' (skipping curves plot). "
              "This is expected if you evaluated 'best_model' — check the underlying "
              "architecture's history file instead.")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", choices=list(MODEL_PATH_LOOKUP.keys()), default="best_model"
    )
    args = parser.parse_args()
    evaluate_model(args.model)
