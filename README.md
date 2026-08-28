# 🚦 Bangladeshi Traffic Sign Classification

An end-to-end deep learning project that classifies **real photographs of Bangladeshi traffic signs**
into categories such as Stop, No Entry, Speed Limit, Turn Left, Turn Right, and Pedestrian Crossing,
using a custom CNN and transfer learning (MobileNetV2, EfficientNetB0), with a Streamlit web app for
live predictions and Grad-CAM explanations.

> **No synthetic data.** This project is built to train on real photographed traffic sign images
> that you download and place under `data/raw/`. Nothing in the pipeline generates fake images.

---

## 1. Project Overview

Traffic sign recognition is a core building block of driver-assistance and road-safety systems.
Bangladesh has its own traffic sign standards (defined by the BRTA Traffic Signs Manual) that differ
visually from Western datasets like GTSRB, so a model trained specifically on Bangladeshi signs is
more useful for local applications (e.g. dash-cam alert systems, driver training apps, smart traffic
management).

This project:
- Loads and preprocesses a real Bangladeshi traffic sign image dataset
- Trains and compares three CNN architectures
- Evaluates them with standard classification metrics
- Explains predictions visually with Grad-CAM
- Serves the best model through a Streamlit web app

---

## 2. Features

- ✅ Clean, modular Python codebase (`src/`) — one responsibility per file
- ✅ Real-data pipeline: load → resize → normalize → stratified train/val/test split
- ✅ Data augmentation (rotation, shift, zoom, shear-equivalent, brightness, contrast)
- ✅ Custom CNN built from Conv2D + BatchNorm + MaxPooling + Dropout + GlobalAveragePooling
- ✅ Transfer learning with **MobileNetV2** and **EfficientNetB0**, auto-compared, best one kept
- ✅ EarlyStopping + ModelCheckpoint + ReduceLROnPlateau during training
- ✅ Full evaluation suite: Accuracy, Precision, Recall, F1, Confusion Matrix, Classification Report
- ✅ Class-imbalance handling via computed class weights
- ✅ Grad-CAM visual explanations ("why did the model predict this?")
- ✅ Confidence threshold to flag uncertain predictions
- ✅ Prediction system with Top-3 results + confidence scores
- ✅ Streamlit web app: upload → preview → predict → results + Grad-CAM

---

## 3. Dataset Information

This project is dataset-agnostic **as long as you provide real images organized as one folder per
class**. It does not ship with any images.

### Recommended real datasets (Bangladeshi traffic signs)

| Dataset | Notes |
|---|---|
| **BTSRB** — Bangladeshi Traffic Sign Recognition Benchmark (Sayeed, Islam, Islam — ICDCECE 2023) | Purpose-built Bangladeshi sign benchmark used in published CNN + transfer-learning research |
| **BDTSR** — Bangladesh Traffic Sign Recognition dataset (48 sign types, collected from roads across the country) | Larger class coverage; good for extending beyond the 6 example classes |
| Kaggle search: `"Bangladeshi traffic sign"` | Several community-uploaded datasets exist — check licensing before use |
| Your own collected photos | Photograph real signs around Dhaka/your area — this is completely valid and often gives the most locally-relevant model |

### Expected folder layout

Download/collect your dataset and arrange it like this **before running any script**:

```
data/raw/
├── Stop/
│   ├── img_0001.jpg
│   ├── img_0002.jpg
│   └── ...
├── No_Entry/
│   └── ...
├── Speed_Limit/
│   └── ...
├── Turn_Left/
│   └── ...
├── Turn_Right/
│   └── ...
└── Pedestrian_Crossing/
    └── ...
```

Folder names become class names automatically. If your dataset uses different classes or more than
six, edit `CLASS_NAMES` in `src/config.py` to match your folder names exactly — every other script
reads from that one list.

---

## 4. Project Structure

```
bd-traffic-sign-classifier/
├── data/
│   ├── raw/                  # <- put your real downloaded dataset here (one folder per class)
│   └── processed/            # <- auto-generated train/val/test split (created by a script)
├── models/
│   ├── best_model.keras      # best model overall, used by the app
│   ├── custom_cnn_best.keras
│   ├── mobilenetv2_best.keras
│   ├── efficientnetb0_best.keras
│   ├── class_indices.json    # class name <-> index mapping
│   └── history/              # pickled training histories (for plotting)
├── notebooks/
│   └── 01_exploration_and_training.ipynb
├── src/
│   ├── config.py              # all paths & hyperparameters in one place
│   ├── utils.py                # shared helper functions
│   ├── data_preprocessing.py   # real dataset verification + train/val/test split
│   ├── data_augmentation.py    # tf.data pipelines + augmentation layers + class weights
│   ├── model_builder.py        # custom CNN + MobileNetV2 + EfficientNetB0 architectures
│   ├── train.py                 # training loop, callbacks, model comparison
│   ├── evaluate.py              # metrics, confusion matrix, classification report, plots
│   ├── gradcam.py               # Grad-CAM heatmap generation
│   └── predict.py               # single-image prediction system (top-3, confidence)
├── reports/                    # auto-generated metrics, plots, confusion matrices
├── app.py                      # Streamlit web application
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 5. Installation

```bash
# 1. Clone / unzip the project, then move into it
cd bd-traffic-sign-classifier

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

Requires Python 3.9–3.11 (TensorFlow compatibility).

---

## 6. How to Train the Model

**Step 1 — Add your real dataset**
Place your downloaded dataset under `data/raw/<ClassName>/...` as shown above.

**Step 2 — Split into train/val/test**
```bash
python -m src.data_preprocessing
```
This copies real images into `data/processed/{train,val,test}/<class>/` using a 70/15/15
stratified split (ratios configurable in `src/config.py`).

**Step 3 — Train and compare all models**
```bash
python -m src.train
```
This trains the custom CNN, MobileNetV2, and EfficientNetB0 (with EarlyStopping,
ModelCheckpoint, and class-weighted loss for imbalance), then saves the highest-validation-accuracy
model as `models/best_model.keras`.

To train just one architecture:
```bash
python -m src.train --model custom_cnn
python -m src.train --model mobilenetv2
python -m src.train --model efficientnetb0
```

**Step 4 — Evaluate on the real test set**
```bash
python -m src.evaluate --model best_model
```
Outputs accuracy/precision/recall/F1, a classification report, and a confusion matrix image,
all saved under `reports/`.

**Step 5 — Predict a single image from the command line**
```bash
python -m src.predict path/to/photo.jpg
```

---

## 7. How to Run the Streamlit App

```bash
streamlit run app.py
```
Then open the local URL Streamlit prints (usually `http://localhost:8501`). Upload a traffic sign
photo, click **Predict**, and view the predicted class, confidence score, top-3 predictions, and a
Grad-CAM heatmap of the regions that most influenced the prediction.

> The app requires `models/best_model.keras` to exist — run training first (Section 6).

---

## 8. Model Architecture — Why Each Layer Is Used

**Custom CNN** (`src/model_builder.py::build_custom_cnn`):

| Layer | Purpose |
|---|---|
| `Conv2D` | Learns visual patterns — edges/colors first, then shapes like arrows and octagons in deeper layers |
| `BatchNormalization` | Stabilizes and speeds up training; mild regularization |
| `MaxPooling2D` | Shrinks spatial size, keeps strongest signals, adds tolerance to the sign being off-center |
| `Dropout` | Randomly disables neurons during training to prevent overfitting on a modest real dataset |
| `GlobalAveragePooling2D` | Replaces a huge `Flatten()` with a compact summary per feature map — fewer parameters, less overfitting |
| `Dense (ReLU)` | Combines extracted features to reason about the class |
| `Dense (Softmax)` | Outputs a probability distribution over classes — gives us the confidence score |

**Transfer learning models** (`build_mobilenetv2`, `build_efficientnetb0`): reuse ImageNet-pretrained
weights (already knows general shapes/textures/edges) with the base frozen and a small trainable
classification head on top — typically higher accuracy with less real training data than a
from-scratch CNN.

---

## 9. Advanced Features Included

- **Model comparison**: `train.py` trains all three architectures and automatically selects the best
  one by validation accuracy for use in the app.
- **Grad-CAM**: `gradcam.py` + the Streamlit app visualize which pixels influenced each prediction.
- **Class imbalance handling**: `data_augmentation.py::compute_class_weights()` weights the loss
  function so under-represented sign classes aren't ignored.
- **Confidence threshold**: predictions below `CONFIDENCE_THRESHOLD` (default 60%, in `config.py`)
  are flagged as "uncertain" instead of silently shown as confident.

---

## 10. Technologies Used

- **TensorFlow / Keras** — model building and training
- **scikit-learn** — metrics, classification report, class weights
- **Matplotlib / Seaborn** — training curves, confusion matrix visualization
- **Streamlit** — web application
- **Pillow (PIL)** — image loading/preprocessing
- **NumPy / Pandas** — data handling

---

## 11. Example Results

After training on a real dataset, `python -m src.evaluate` populates `reports/` with results like:

```
Accuracy : 0.94xx
Precision: 0.94xx (weighted)
Recall   : 0.93xx (weighted)
F1-score : 0.93xx (weighted)
```
(exact numbers depend entirely on your real dataset's size and quality — these are illustrative of
what the evaluation script reports, not a guaranteed benchmark)

Plots saved automatically:
- `reports/<model>_training_curves.png` — train vs. validation accuracy & loss
- `reports/<model>_confusion_matrix.png` — per-class confusion matrix
- `reports/<model>_classification_report.txt` — per-class precision/recall/F1

---

## 12. Notes & Honest Limitations

- This repo ships **no images**. You must supply a real dataset (see Section 3) — this is
  intentional, both for licensing reasons and because using real, locally-relevant photos is what
  makes the resulting model actually useful.
- `IMG_SIZE` defaults to `(64, 64)` for fast CPU training; bump to `(128, 128)` in `src/config.py`
  if you have a GPU and want higher accuracy.
- Grad-CAM layer names in `config.GRADCAM_LAYER_NAMES` assume standard MobileNetV2/EfficientNetB0
  layer names — if you modify those architectures, update the layer name there too.
