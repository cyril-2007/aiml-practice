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
st.markdown(
    "### AI-powered image recognition\n"
    "Upload a clear photo of a cat or dog and let your trained CNN classify it."
)
st.caption(
    "A beginner-friendly deep learning project. Predictions are model estimates, "
    "not guarantees of correctness."
)

with st.expander("🔍 How does this classifier work?"):
    st.markdown(
        """
        1. **Upload:** Choose a JPG or PNG image.
        2. **Preprocess:** The image is converted to RGB and resized to 128 × 128 pixels.
        3. **Normalize:** Pixel values are scaled from 0–255 to 0–1.
        4. **Predict:** The trained convolutional neural network (CNN) produces a score.
        5. **Display:** The app shows the predicted class and both Cat and Dog scores.

        **Important:** A high score does not guarantee that a prediction is correct.
        """
    )

with st.sidebar:
    st.header("Tips for best results")
    st.info("Use a clear photo with one visible cat or dog.")
    st.write("Supported formats: JPG, JPEG, and PNG.")

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

            
            
            confidence = max(cat_score, dog_score)

            st.divider()
            with st.container(border=True):
                st.subheader("🔎 Classification result")
                st.title(label)
                st.metric("Prediction confidence", f"{confidence:.1%}")

                st.caption(
                    "This is the model's highest output score, "
                    "not a guarantee that the prediction is correct."
                )


            
            left_column, right_column = st.columns(2)

            with left_column:
                st.metric("🐱 Cat score", f"{cat_score:.1%}")
                st.progress(cat_score)

            with right_column:
                st.metric("🐶 Dog score", f"{dog_score:.1%}")
                st.progress(dog_score)



            st.caption(
                st.caption(
    "The scores are model estimates, not calibrated probabilities. "
    "Unfamiliar images may still receive high scores."
)
            )
    except (UnidentifiedImageError, OSError) as error:
        st.error(f"That file could not be read as an image: {error}")
    except FileNotFoundError as error:
        st.error(str(error))
    except Exception as error:
        st.error("The app could not classify this image.")
        st.exception(error)
