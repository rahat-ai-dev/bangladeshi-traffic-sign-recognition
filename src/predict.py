"""
predict.py
----------
The single-image prediction system used by both the CLI and the
Streamlit app. Given a path (or a PIL image) it:
  1. Preprocesses the image the same way training images were processed
  2. Predicts probabilities for every class
  3. Returns the top prediction + confidence + top-3 predictions
  4. Flags the prediction as "uncertain" if confidence is below threshold

Usage (CLI):
    python -m src.predict path/to/image.jpg
"""

import sys
import numpy as np
from PIL import Image
import tensorflow as tf

from src import config
from src.utils import load_class_indices


_model_cache = {}


def load_prediction_model(model_path=config.BEST_MODEL_PATH):
    """Cache the loaded model so Streamlit doesn't reload it on every click."""
    if model_path not in _model_cache:
        _model_cache[model_path] = tf.keras.models.load_model(model_path)
    return _model_cache[model_path]


def preprocess_image(image: Image.Image):
    """
    Same preprocessing used at training time: resize to IMG_SIZE and
    keep raw pixel values in [0, 255]. We do NOT divide by 255 here —
    each model (see model_builder.py) now applies its own correct
    normalization internally as its first layer(s), so every architecture
    can be fed the exact same raw-pixel input here.
    Returns a batched array (1, H, W, 3).
    """
    image = image.convert("RGB")
    image = image.resize((config.IMG_SIZE[1], config.IMG_SIZE[0]))  # PIL wants (W, H)
    arr = np.array(image).astype("float32")
    return np.expand_dims(arr, axis=0)


def predict_image(image_input, model_path=config.BEST_MODEL_PATH, top_k=3):
    """
    image_input: file path (str) OR a PIL.Image
    Returns a dict:
        {
          "predicted_class": str,
          "confidence": float,          # 0..1
          "is_uncertain": bool,
          "top_k": [(class_name, prob), ...],   # sorted, length top_k
          "raw_probs": np.ndarray
        }
    """
    if isinstance(image_input, str):
        image = Image.open(image_input)
    else:
        image = image_input

    model = load_prediction_model(model_path)
    _, index_to_class = load_class_indices()

    batch = preprocess_image(image)
    probs = model.predict(batch, verbose=0)[0]  # shape (num_classes,)

    top_indices = np.argsort(probs)[::-1][:top_k]
    top_k_results = [(index_to_class[int(i)], float(probs[i])) for i in top_indices]

    best_idx = int(top_indices[0])
    best_class = index_to_class[best_idx]
    best_conf = float(probs[best_idx])

    return {
        "predicted_class": best_class,
        "confidence": best_conf,
        "is_uncertain": best_conf < config.CONFIDENCE_THRESHOLD,
        "top_k": top_k_results,
        "raw_probs": probs,
    }


def _print_result(result):
    print(f"\nPredicted class : {result['predicted_class']}")
    print(f"Confidence      : {result['confidence'] * 100:.2f}%")
    if result["is_uncertain"]:
        print(f"WARNING: confidence is below the {config.CONFIDENCE_THRESHOLD * 100:.0f}% "
              "threshold — this prediction may be unreliable.")
    print("\nTop 3 predictions:")
    for rank, (cls, prob) in enumerate(result["top_k"], start=1):
        print(f"  {rank}. {cls:25s} {prob * 100:6.2f}%")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.predict path/to/image.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    result = predict_image(image_path)
    _print_result(result)
