"""
model_builder.py
------------------
Defines three model architectures we compare:
  1. build_custom_cnn()     -> a CNN built from scratch
  2. build_mobilenetv2()    -> transfer learning with MobileNetV2
  3. build_efficientnetb0() -> transfer learning with EfficientNetB0

Each function returns a compiled tf.keras.Model ready to .fit().
"""

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

from src import config


def build_custom_cnn(input_shape=config.INPUT_SHAPE, num_classes=config.NUM_CLASSES):
    """
    A CNN built from scratch. Layer-by-layer reasoning:

    - Conv2D: learns local visual patterns (edges, corners, colors of the
      sign's border/shape). We stack multiple Conv2D blocks so early layers
      learn simple edges and later layers combine them into complex shapes
      like arrows, octagons, letters.
    - BatchNormalization: normalizes activations between layers so training
      is faster and more stable, and it also acts as a mild regularizer.
    - MaxPooling2D: shrinks the spatial size (keeps the strongest signal),
      which reduces computation and makes the model tolerant to the sign
      appearing slightly shifted in the photo.
    - Dropout: randomly "turns off" neurons during training so the network
      can't over-rely on any single feature — this fights overfitting,
      which matters a lot with a modestly-sized real dataset.
    - GlobalAveragePooling2D: instead of a huge Flatten() (which creates a
      massive, overfitting-prone Dense layer), this averages each feature
      map down to one number. Fewer parameters, less overfitting.
    - Dense (ReLU): combines the extracted features to reason about which
      class they best match.
    - Dense (Softmax): the output layer — turns raw scores into a
      probability distribution over the traffic sign classes so we can
      read off a class + confidence score.
    """
    inputs = layers.Input(shape=input_shape, name="input_image")

    # Normalize raw [0, 255] pixels to [0, 1] — the custom CNN's own
    # preprocessing step (kept inside the model so predict.py/app.py can
    # always feed raw pixel values, regardless of which architecture is used).
    x = layers.Rescaling(1.0 / 255, name="rescaling")(inputs)

    # --- Conv block 1 ---
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # --- Conv block 2 ---
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # --- Conv block 3 (named so Grad-CAM can find it later) ---
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu", name="conv2d_last")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)

    # --- Classifier head ---
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs, outputs, name="custom_cnn")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _build_transfer_model(base_model_fn, name, input_shape, num_classes,
                           preprocess_fn=None, fine_tune_at=None):
    """
    Shared logic for both transfer-learning models.

    WHY transfer learning at all? MobileNetV2 / EfficientNetB0 were
    pre-trained on 1.3M+ ImageNet images, so they already know general
    visual features (edges, shapes, colors, textures). We reuse that
    knowledge instead of learning it from zero — this typically gives
    higher accuracy with a smaller real dataset than a from-scratch CNN.

    preprocess_fn: the architecture-specific preprocessing function
    (e.g. tf.keras.applications.mobilenet_v2.preprocess_input). Each
    pretrained backbone was trained with a specific input range/normalization,
    and using the wrong one badly hurts accuracy (this bit us with
    EfficientNetB0 previously — its layer expects raw [0,255] and doesn't
    need a manual preprocess_fn, so we pass None for it).
    """
    base_model = base_model_fn(
        input_shape=input_shape, include_top=False, weights="imagenet"
    )
    base_model.trainable = False  # freeze pre-trained weights initially

    if fine_tune_at is not None:
        # Optionally unfreeze the last N layers for fine-tuning on our data
        base_model.trainable = True
        for layer in base_model.layers[:-fine_tune_at]:
            layer.trainable = False

    inputs = layers.Input(shape=input_shape)
    if preprocess_fn is not None:
        # e.g. MobileNetV2 expects pixels scaled to [-1, 1]
        x = layers.Lambda(preprocess_fn, name="preprocess")(inputs)
    else:
        # e.g. EfficientNetB0 has its own built-in Rescaling layer and
        # expects raw [0, 255] pixels directly — no manual step needed here.
        x = inputs
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name=name)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_mobilenetv2(input_shape=config.INPUT_SHAPE, num_classes=config.NUM_CLASSES):
    """MobileNetV2 backbone — small and fast, good for a laptop/edge device."""
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    return _build_transfer_model(
        MobileNetV2, "mobilenetv2", input_shape, num_classes, preprocess_fn=preprocess_input
    )


def build_efficientnetb0(input_shape=config.INPUT_SHAPE, num_classes=config.NUM_CLASSES):
    """EfficientNetB0 backbone — typically stronger accuracy, a bit heavier.
    Note: EfficientNetB0 already includes its own internal Rescaling layer
    and expects raw [0, 255] pixels, so we pass preprocess_fn=None."""
    from tensorflow.keras.applications import EfficientNetB0
    return _build_transfer_model(
        EfficientNetB0, "efficientnetb0", input_shape, num_classes, preprocess_fn=None
    )


MODEL_REGISTRY = {
    "custom_cnn": build_custom_cnn,
    "mobilenetv2": build_mobilenetv2,
    "efficientnetb0": build_efficientnetb0,
}


if __name__ == "__main__":
    # Quick sanity check: build each model and print its summary/param count
    for name, builder in MODEL_REGISTRY.items():
        m = builder()
        n_params = m.count_params()
        print(f"\n{'=' * 60}\n{name} — {n_params:,} parameters\n{'=' * 60}")
