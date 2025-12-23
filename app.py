import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
import gdown

# PAGE CONFIG
st.set_page_config(
    page_title="AI vs Human Art Detector",
    page_icon="🎨",
    layout="centered"
)

# MODEL CONFIG
MODEL_PATH = "model.h5"
FILE_ID = "12dT7IBR6UjebfBv-v_TE_jhcZp8irH_7"

# DOWNLOAD MODEL
def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model (first launch only)..."):
            url = f"https://drive.google.com/uc?export=download&id={FILE_ID}"
            gdown.download(url, MODEL_PATH, quiet=False)

# LOAD MODEL
@st.cache_resource
def load_trained_model():
    download_model()
    return load_model(MODEL_PATH)

# PREPROCESS IMAGE
def preprocess_image(image):
    image = image.resize((224, 224))
    image = image.convert("RGB")
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# PREDICTION
def predict_image(model, image):
    img = preprocess_image(image)
    pred = model.predict(img, verbose=0)
    return float(pred[0][0])

# UI
st.title("🎨 AI vs Human Art Detector")
st.markdown(
    "Powered by **InceptionV3**, this model intelligently distinguishes "
    "between **AI-generated** and **human-created** artworks."
)

with st.sidebar:
    st.header("ℹ️ How It Works")
    st.markdown("""
    Upload an artwork and let the model analyze it.
    The output reflects how strongly the image resembles
    AI-generated or human-created art.

    Scores closer to **1** lean toward AI,
    while scores closer to **0** lean toward human.
    """)

# LOAD MODEL
try:
    model = load_trained_model()
    st.success("Model successfully loaded ✅")
except Exception as e:
    model = None
    st.error("Model failed to load ❌")
    st.exception(e)

# IMAGE UPLOAD
if model:
    uploaded_file = st.file_uploader(
        "Upload your artwork",
        type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Input Image", use_container_width=True)

        if st.button("🔍 Detect Artwork", use_container_width=True):
            with st.spinner("Analyzing artwork..."):
                confidence = predict_image(model, image)

                if confidence >= 0.5:
                    label = "🤖 AI Generated"
                    ai_score = confidence
                    human_score = 1 - confidence
                else:
                    label = "👨‍🎨 Human Made"
                    ai_score = confidence
                    human_score = 1 - confidence

                st.markdown("---")
                st.subheader(f"Result: {label}")

                st.progress(confidence)

                col1, col2 = st.columns(2)
                col1.metric("AI Probability", f"{ai_score:.2%}")
                col2.metric("Human Probability", f"{human_score:.2%}")

                with st.expander("Technical Details"):
                    st.write(f"Sigmoid output: `{confidence:.6f}`")
                    st.write("Input shape: `(224, 224, 3)`")
