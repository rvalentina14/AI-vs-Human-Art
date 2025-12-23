import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
import gdown np

# PAGE CONFIG
st.set_page_config(
    page_title="AI vs Human Art Detector",
    page_icon="🎨",
    layout="centered"
)

# MODEL CONFIG
model_path = "inceptionv3_aiart_detector_final.h5"
FILE_ID = "12dT7IBR6UjebfBv-v_TE_jhcZp8irH_7"

def download_model():
    if not os.path.exists(model_path):
        with st.spinner("Downloading model... (first time only)"):
            url = f"https://drive.google.com/uc?id={FILE_ID}"
            gdown.download(url, model_path, quiet=False)

# LOAD MODEL
@st.cache_resource
def load_trained_model():
    download_model()
    return load_model(model_path)

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
st.markdown("This model is powered by InceptionV3, trained to intelligently distinguish between AI-generated and human-created artworks.")

with st.sidebar:
    st.header("⚙️ Model Information")
    model_path = st.text_input(
        "Path model (.h5)",
        value="inceptionv3_aiart_detector_final.h5"
    )

    st.markdown("---")
    st.markdown("""
    **Interpretation**
    Output ∈ [0,1]
    - > 0.5 → **AI Generated**
    - < 0.5 → **Human Made**
    """)

# LOAD MODEL
try:
    model = load_trained_model(model_path)
    st.success("Model succesfully loaded")
except:
    model = None
    st.error("Model failed to load")

# IMAGE UPLOAD
if model:
    uploaded_file = st.file_uploader(
        "Upload your art here!",
        type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Input Image", use_container_width=True)

        if st.button("🔍 Detect now", use_container_width=True):
            with st.spinner("Analyzing..."):
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

                with st.expander("Technical Detail"):
                    st.write(f"Output sigmoid: {confidence:.6f}")

                    st.write("Input shape: (224, 224, 3)")