import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ---------------------------------------------------------------------------
# Configuration (must match how the models were trained in the notebook)
# ---------------------------------------------------------------------------
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]   # label 0 = NORMAL, label 1 = PNEUMONIA
THRESHOLD = 0.5

MODELS = {
    "CNN": {
        "path": "best_chestscan_cnn.keras",
        "size": (180, 180),
        "preprocess": lambda x: x / 255.0,          # Rescaling(1/255)
    },
    "MobileNetV2": {
        "path": "best_mobilenetv2.keras",
        "size": (224, 224),
        "preprocess": preprocess_input,             # pixels -> [-1, 1]
    },
}

st.set_page_config(page_title="ChestScan AI", page_icon="🫁", layout="centered")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model(path):
    return tf.keras.models.load_model(path)


def predict(model_name, image):
    cfg = MODELS[model_name]
    model = load_model(cfg["path"])

    img = image.convert("RGB").resize(cfg["size"], Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32)
    arr = cfg["preprocess"](arr)
    arr = np.expand_dims(arr, axis=0)

    prob = float(model.predict(arr, verbose=0).ravel()[0])   # P(PNEUMONIA)
    label = CLASS_NAMES[int(prob >= THRESHOLD)]
    return label, prob


def show_result(model_name, label, prob):
    st.subheader(model_name)
    if label == "PNEUMONIA":
        st.error(f"Prediction: **{label}**")
    else:
        st.success(f"Prediction: **{label}**")
    st.progress(prob, text=f"P(PNEUMONIA) = {prob:.1%}")
    st.caption(f"P(NORMAL) = {1 - prob:.1%}")


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("🫁 ChestScan AI")
st.write("Upload a chest X-ray and the model predicts **NORMAL** or **PNEUMONIA**.")

choice = st.sidebar.radio("Model", ["MobileNetV2", "CNN", "Compare both"])
st.sidebar.info(
    "Educational project. Not a medical device and not a substitute "
    "for a doctor's diagnosis."
)

uploaded = st.file_uploader("Chest X-ray image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded image", use_container_width=True)

    if st.button("Analyze", type="primary"):
        names = list(MODELS) if choice == "Compare both" else [choice]
        cols = st.columns(len(names))
        for col, name in zip(cols, names):
            with col:
                with st.spinner(f"Running {name}..."):
                    try:
                        label, prob = predict(name, image)
                        show_result(name, label, prob)
                    except Exception as e:
                        st.error(f"{name} failed: {e}")