"""Beginner-friendly Streamlit interface for the Cat vs Dog CNN model."""

from io import BytesIO
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, UnidentifiedImageError


IMG_SIZE = 128
MODEL_PATH = Path(__file__).resolve().parent / "cat_dog_model.keras"


@st.cache_resource(show_spinner="Loading the trained CNN model...")
def load_model():
    """Load the model once per Streamlit server session."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH.name}. "
            "Keep it in the same folder as app.py."
        )
    return tf.keras.models.load_model(MODEL_PATH)


def prepare_image(uploaded_bytes: bytes) -> Image.Image:
    """Convert an uploaded image to RGB and resize it like the training data."""
    with Image.open(BytesIO(uploaded_bytes)) as image:
        return image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))


def predict_image(model, image: Image.Image) -> tuple[str, float, float]:
    """Return the top class and the model's cat/dog output scores."""
    image_array = np.asarray(image, dtype=np.float32) / 255.0
    batch = np.expand_dims(image_array, axis=0)

    dog_score = float(model.predict(batch, verbose=0).squeeze())
    dog_score = float(np.clip(dog_score, 0.0, 1.0))
    cat_score = 1.0 - dog_score
    label = "Dog 🐶" if dog_score >= 0.5 else "Cat 🐱"
    return label, cat_score, dog_score


st.set_page_config(page_title="Cat vs Dog Classifier", page_icon="🐾", layout="centered")

st.title("🐾 Cat vs Dog Classifier")
st.write("Upload a photo and let your trained CNN choose between **cat** and **dog**.")
st.caption("This is a learning project. The result is the model's best guess, not a guarantee.")

with st.sidebar:
    st.header("How it works")
    st.write(
        "The app converts your photo to RGB, resizes it to 128 × 128 pixels, "
        "and scales pixel values from 0–255 to 0–1 before prediction."
    )
    st.info("For best results, use a clear photo with one visible cat or dog.")

uploaded_file = st.file_uploader(
    "Choose an image", type=["jpg", "jpeg", "png"], help="JPG and PNG images are supported."
)

if uploaded_file is None:
    st.info("Start by uploading a JPG or PNG image.")
else:
    try:
        uploaded_bytes = uploaded_file.getvalue()
        display_image = Image.open(BytesIO(uploaded_bytes)).convert("RGB")
        prepared_image = prepare_image(uploaded_bytes)

        st.image(display_image, caption=f"Uploaded image: {uploaded_file.name}", use_container_width=True)

        if st.button("Classify image", type="primary"):
            model = load_model()
            label, cat_score, dog_score = predict_image(model, prepared_image)

            st.subheader(f"Prediction: {label}")
            st.write("The model assigned the higher output score to this class.")

            left_column, right_column = st.columns(2)
            left_column.metric("Cat score", f"{cat_score:.1%}")
            right_column.metric("Dog score", f"{dog_score:.1%}")
            st.progress(int(max(cat_score, dog_score) * 100))

            st.caption(
                "Scores show the model's sigmoid output, not a calibrated measure of certainty. "
                "Images unlike the training data can still receive a high score."
            )
    except (UnidentifiedImageError, OSError) as error:
        st.error(f"That file could not be read as an image: {error}")
    except FileNotFoundError as error:
        st.error(str(error))
    except Exception as error:
        st.error("The app could not classify this image.")
        st.exception(error)
