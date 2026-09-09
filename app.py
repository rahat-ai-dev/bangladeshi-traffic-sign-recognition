"""
app.py
------
Streamlit web app for the Bangladeshi Traffic Sign Classifier.

Run with:
    streamlit run app.py

Features:
- Upload a traffic sign image
- Preview the uploaded image
- Click "Predict" to classify it
- See predicted class, confidence score, and top-3 predictions
- See a Grad-CAM heatmap of what the model focused on
- Uncertain predictions (below the confidence threshold) are flagged
"""

import os
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf

from src import config
from src.predict import load_prediction_model, preprocess_image, predict_image
from src.gradcam import make_gradcam_heatmap, overlay_heatmap

st.set_page_config(
    page_title="Bangladeshi Traffic Sign Classifier",
    page_icon="🚦",
    layout="centered",
)

st.title("🚦 Bangladeshi Traffic Sign Classifier")
st.write(
    "Upload a photo of a Bangladeshi traffic sign and the model will "
    "predict which sign it is, using a CNN trained on real traffic sign images."
)

# ----------------------------------------------------------------------
# Model availability check — fail gracefully with a helpful message
# instead of a raw stack trace if training hasn't been run yet.
# ----------------------------------------------------------------------
if not os.path.exists(config.BEST_MODEL_PATH):
    st.error(
        "No trained model found at `models/best_model.keras`.\n\n"
        "Train a model first:\n"
        "```\n"
        "python -m src.data_preprocessing\n"
        "python -m src.train\n"
        "```"
    )
    st.stop()

model = load_prediction_model()

with st.sidebar:
    st.header("About")
    st.write(
        "This app classifies real Bangladeshi traffic signs into "
        f"**{config.NUM_CLASSES}** categories:"
    )
    for c in config.CLASS_NAMES:
        st.write(f"- {c.replace('_', ' ')}")
    st.divider()
    st.write(f"Confidence threshold: **{config.CONFIDENCE_THRESHOLD * 100:.0f}%**")
    show_gradcam = st.checkbox("Show Grad-CAM explanation", value=True)

uploaded_file = st.file_uploader(
    "Upload a traffic sign image", type=["jpg", "jpeg", "png", "bmp", "webp"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", use_container_width=True)

    if st.button("🔍 Predict", type="primary"):
        with st.spinner("Classifying..."):
            result = predict_image(image)

        # --- Main prediction result ---
        if result["is_uncertain"]:
            st.warning(
                f"⚠️ Uncertain prediction — confidence "
                f"({result['confidence'] * 100:.1f}%) is below the "
                f"{config.CONFIDENCE_THRESHOLD * 100:.0f}% threshold. "
                "Try a clearer, well-lit, front-facing photo."
            )
        else:
            st.success(f"✅ Predicted sign: **{result['predicted_class'].replace('_', ' ')}**")

        st.metric("Confidence", f"{result['confidence'] * 100:.2f}%")

        # --- Top-3 predictions ---
        st.subheader("Top 3 predictions")
        for rank, (cls, prob) in enumerate(result["top_k"], start=1):
            st.write(f"{rank}. **{cls.replace('_', ' ')}** — {prob * 100:.2f}%")
            st.progress(min(max(prob, 0.0), 1.0))

        # --- Grad-CAM visualization ---
        if show_gradcam:
            st.subheader("Grad-CAM: where the model looked")
            try:
                arch_name = model.name if model.name in config.GRADCAM_LAYER_NAMES else "custom_cnn"
                layer_name = config.GRADCAM_LAYER_NAMES.get(arch_name, "conv2d_last")

                batch = preprocess_image(image)
                heatmap, _ = make_gradcam_heatmap(batch, model, layer_name)
                overlay = overlay_heatmap(batch[0], heatmap)
                st.image(overlay, caption="Regions that most influenced the prediction",
                         use_container_width=True)
            except Exception as e:
                st.info(
                    "Grad-CAM isn't available for this model's architecture "
                    f"(layer lookup failed: {e})."
                )
else:
    st.info("👆 Upload an image to get started.")
