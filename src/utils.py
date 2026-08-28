"""
utils.py
--------
Small shared helper functions used by more than one script
(folder creation, saving history, loading class names, etc.).
Keeping these in one place avoids copy-pasting the same code everywhere.
"""

import os
import json
import pickle
import matplotlib.pyplot as plt

from src import config


def ensure_dir(path):
    """Create a folder if it doesn't already exist (no error if it does)."""
    os.makedirs(path, exist_ok=True)


def save_class_indices(class_indices, path=config.CLASS_INDICES_PATH):
    """
    Save the {class_name: index} mapping that Keras' image loaders create.
    We need this exact mapping again later at prediction time, so the
    model's output index 0,1,2... can be turned back into a human label.
    """
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(class_indices, f, indent=2)


def load_class_indices(path=config.CLASS_INDICES_PATH):
    """Load the {class_name: index} mapping saved during training."""
    with open(path, "r") as f:
        class_indices = json.load(f)
    # invert to {index: class_name} — this is what we need for predictions
    index_to_class = {v: k for k, v in class_indices.items()}
    return class_indices, index_to_class


def save_history(history, name):
    """Pickle a Keras History.history dict so we can re-plot it later
    without retraining."""
    ensure_dir(config.HISTORY_DIR)
    path = os.path.join(config.HISTORY_DIR, f"{name}_history.pkl")
    with open(path, "wb") as f:
        pickle.dump(history, f)


def load_history(name):
    path = os.path.join(config.HISTORY_DIR, f"{name}_history.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def plot_training_curves(history, model_name, save_path=None):
    """
    Plot training vs validation accuracy and loss side by side.
    `history` can be a Keras History.history dict.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # --- Accuracy ---
    axes[0].plot(history["accuracy"], label="Train Accuracy")
    axes[0].plot(history["val_accuracy"], label="Validation Accuracy")
    axes[0].set_title(f"{model_name} — Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # --- Loss ---
    axes[1].plot(history["loss"], label="Train Loss")
    axes[1].plot(history["val_loss"], label="Validation Loss")
    axes[1].set_title(f"{model_name} — Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved training curves to {save_path}")
    plt.close(fig)
    return fig
