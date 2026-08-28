"""
data_augmentation.py
---------------------
Builds tf.data pipelines that:
  1. Stream real images from data/processed/{train,val,test}
  2. Resize to a fixed size and normalize pixel values to [0, 1]
  3. Apply data augmentation to the TRAINING set only

Why augmentation only on train, not val/test?
Validation and test sets must reflect real, unmodified images so that
accuracy numbers are trustworthy. Augmentation is a training-time trick
to make the model more robust — it should never touch evaluation data.

We use Keras' modern *preprocessing layers* (tf.keras.layers) instead of
the older ImageDataGenerator, because layers run on the GPU as part of
the model graph and are faster, but the same effect (rotation, shift,
zoom, shear, brightness) is what ImageDataGenerator used to provide.
"""

import tensorflow as tf
from tensorflow.keras import layers

from src import config


def get_augmentation_layer():
    """
    A small stack of augmentation layers, applied only during training
    (Keras automatically disables them at inference/prediction time).

    - RandomRotation      -> handles signs photographed at an angle
    - RandomTranslation   -> width/height shift (sign not centered in frame)
    - RandomZoom          -> sign closer/farther from camera
    - RandomBrightness    -> different lighting/weather conditions in Bangladesh
    - RandomContrast      -> haze, glare, dusty camera lens etc.
    (Shear isn't a built-in Keras layer; RandomRotation + RandomZoom
    together approximate the same "signs are never perfectly upright" effect.)
    """
    return tf.keras.Sequential(
        [
            layers.RandomRotation(0.08),               # ~ +/-15 degrees
            layers.RandomTranslation(0.10, 0.10),       # width/height shift 10%
            layers.RandomZoom(0.15),                    # zoom in/out 15%
            layers.RandomBrightness(0.15, value_range=(0, 255)),
            layers.RandomContrast(0.15),
        ],
        name="augmentation",
    )


def _make_dataset(directory, shuffle, augment):
    """Internal helper: load real images from a folder-per-class directory."""
    ds = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="categorical",          # one-hot, matches softmax + categorical_crossentropy
        class_names=config.CLASS_NAMES,
        color_mode="rgb",
        batch_size=config.BATCH_SIZE,
        image_size=config.IMG_SIZE,
        shuffle=shuffle,
        seed=config.RANDOM_SEED,
    )

    # NOTE: pixel normalization is NOT done here anymore. Different backbones
    # expect different input ranges (custom CNN wants [0,1], MobileNetV2 wants
    # [-1,1], EfficientNetB0 has its own built-in rescaling and wants raw
    # [0,255]). Each model in model_builder.py now applies its own correct
    # preprocessing as the first layer(s) of the model itself, so here we
    # just pass through raw pixel values in [0,255].

    if augment:
        aug_layer = get_augmentation_layer()
        ds = ds.map(lambda x, y: (aug_layer(x, training=True), y),
                    num_parallel_calls=tf.data.AUTOTUNE)

    return ds.prefetch(tf.data.AUTOTUNE)


def get_datasets():
    """
    Returns (train_ds, val_ds, test_ds) tf.data.Dataset objects built from
    the REAL images already split into data/processed/{train,val,test}.
    Run src/data_preprocessing.py first to create that folder structure.
    """
    train_ds = _make_dataset(config.TRAIN_DIR, shuffle=True, augment=True)
    val_ds = _make_dataset(config.VAL_DIR, shuffle=False, augment=False)
    test_ds = _make_dataset(config.TEST_DIR, shuffle=False, augment=False)
    return train_ds, val_ds, test_ds


def compute_class_weights():
    """
    Handle class imbalance: real-world sign photos are rarely perfectly
    balanced across classes (e.g. far fewer 'Pedestrian Crossing' photos
    than 'Stop'). We compute weights so the loss function penalizes
    mistakes on under-represented classes more heavily.
    """
    import os
    import numpy as np
    from sklearn.utils.class_weight import compute_class_weight

    labels = []
    for idx, class_name in enumerate(config.CLASS_NAMES):
        class_dir = os.path.join(config.TRAIN_DIR, class_name)
        if os.path.isdir(class_dir):
            n = len([f for f in os.listdir(class_dir)
                      if os.path.isfile(os.path.join(class_dir, f))])
            labels += [idx] * n

    labels = np.array(labels)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(config.NUM_CLASSES),
        y=labels,
    )
    class_weight_dict = {i: float(w) for i, w in enumerate(weights)}
    print("Computed class weights (to counter class imbalance):", class_weight_dict)
    return class_weight_dict
