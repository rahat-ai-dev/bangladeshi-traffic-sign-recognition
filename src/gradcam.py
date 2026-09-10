"""
gradcam.py
----------
Grad-CAM (Gradient-weighted Class Activation Mapping) shows WHICH pixels
of an image most influenced the model's prediction, as a heatmap overlay.
This gives a visual "explanation" for why the model predicted a class —
useful for debugging and for building trust in the model.

How it works (short version):
1. Run the image through the model up to the last convolutional layer.
2. Compute the gradient of the predicted class score with respect to that
   layer's feature maps — this tells us "how much would the prediction
   change if this region changed".
3. Weight each feature map by how important it was (average gradient),
   sum them up, and apply ReLU -> a coarse heatmap of "important regions".
4. Resize that heatmap to the original image size and overlay it.
"""

import numpy as np
import tensorflow as tf
import matplotlib  
import matplotlib.cm as cm


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    img_array: preprocessed image batch, shape (1, H, W, 3), pixel values in [0,1]
    model: trained Keras model
    last_conv_layer_name: name of the last Conv2D layer to explain from
    pred_index: which class to explain (defaults to the model's top prediction)
    """
    grad_model = tf.keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # Gradient of the chosen class score w.r.t. the conv layer's output
    grads = tape.gradient(class_channel, conv_outputs)

    # Average gradient over width/height -> importance weight per channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize to [0, 1] for visualization
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), int(pred_index)


def overlay_heatmap(original_img, heatmap, alpha=0.4):
    """
    original_img: HxWx3 uint8 or float [0,1] image (NOT batched)
    heatmap: 2D array from make_gradcam_heatmap()
    Returns an RGB uint8 image with the heatmap overlaid.
    """
    if original_img.max() <= 1.0:
        original_img = (original_img * 255).astype(np.uint8)

    heatmap_uint8 = np.uint8(255 * heatmap)
    jet = matplotlib.colormaps["jet"]
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap_uint8]

    jet_heatmap_img = tf.keras.utils.array_to_img(jet_heatmap)
    jet_heatmap_img = jet_heatmap_img.resize((original_img.shape[1], original_img.shape[0]))
    jet_heatmap_arr = tf.keras.utils.img_to_array(jet_heatmap_img)

    overlaid = jet_heatmap_arr * alpha + original_img
    overlaid_img = tf.keras.utils.array_to_img(overlaid)
    return np.array(overlaid_img)
