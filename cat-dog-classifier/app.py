
"""Streamlit Cat vs Dog classifier using MobileNetV2 transfer learning."""

from io import BytesIO
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, UnidentifiedImageError


IMG_SIZE = 160
MODEL_PATH = Path(__file__).resolve().parent / "cat_dog_mobilenet.keras"


st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="🐾",
    layout="centered",
)


@st.cache_resource(show_spinner="Loading the MobileNetV2 model...")
def load_model():
    """Load the trained MobileNetV2 model once."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH.name}. "
            "Keep it in the same folder as app.py."
        )

    return tf.keras.models.load_model(MODEL_PATH)


def prepare_image(uploaded_bytes: bytes) -> Image.Image:
    """Convert the uploaded image to RGB and resize to 160 x 160."""
    with Image.open(BytesIO(uploaded_bytes)) as image:
        return image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))


def predict_image(model, image: Image.Image) -> tuple[str, float, float]:
    """Predict Cat or Dog using MobileNetV2 preprocessing."""
    image_array = np.asarray(image, dtype=np.float32)

    # MobileNetV2 expects pixel values scaled to the range [-1, 1].
    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(
        image_array
    )

    batch = np.expand_dims(image_array, axis=0)
    dog_score = float(model.predict(batch, verbose=0).squeeze())
    dog_score = float(np.clip(dog_score, 0.0, 1.0))
    cat_score = 1.0 - dog_score

    label = "Dog 🐶" if dog_score >= 0.5 else "Cat 🐱"

    return label, cat_score, dog_score


st.title("🐾 Cat vs Dog Classifier")

st.markdown(
    "### AI-powered image recognition\n"
    "Upload a clear photo of a cat or dog and let MobileNetV2 "
    "classify the image."
)

st.caption(
    "A deep learning project using transfer learning. "
    "Predictions are model estimates, not guarantees of correctness."
)

with st.expander("🔍 How does this classifier work?"):
    st.markdown(
        """
        1. **Upload:** Choose a JPG or PNG image.
        2. **Preprocess:** Convert the image to RGB and resize it to
           160 × 160 pixels.
        3. **Normalize:** MobileNetV2 scales pixel values to the range -1 to 1.
        4. **Predict:** The trained MobileNetV2 model estimates whether
           the image contains a cat or dog.
        5. **Display:** View the predicted class and both model scores.

        **Important:** The model is designed for cats and dogs.
        It can misclassify unrelated images, even with a high score.
        """
    )

with st.sidebar:
    st.header("Tips for best results")
    st.info("Use a clear photo with a visible cat or dog.")
    st.write("Supported formats: JPG, JPEG, and PNG.")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"],
    help="JPG and PNG images are supported.",
)

if uploaded_file is None:
    st.info("Start by uploading a JPG or PNG image.")

else:
    try:
        uploaded_bytes = uploaded_file.getvalue()

        with Image.open(BytesIO(uploaded_bytes)) as original:
            display_image = original.convert("RGB")

        prepared_image = prepare_image(uploaded_bytes)

        st.image(
            display_image,
            caption=f"Uploaded image: {uploaded_file.name}",
            width="stretch",
        )

        if st.button("Classify image", type="primary"):
            model = load_model()
            label, cat_score, dog_score = predict_image(
                model, prepared_image
            )

            confidence = max(cat_score, dog_score)

            st.divider()

            with st.container(border=True):
                st.subheader("🔎 Classification result")
                st.title(label)
                st.metric(
                    "Prediction confidence",
                    f"{confidence:.1%}",
                )

                if confidence < 0.70:
                    st.warning(
                        "Low-confidence prediction. The model may be "
                        "unsure which class fits this image."
                    )

                st.info(
                    "This model is designed for cats and dogs. "
                    "It may misclassify unrelated images, even with "
                    "a high score."
                )

                st.caption(
                    "The highest output score is not a guarantee "
                    "that the prediction is correct."
                )

            left_column, right_column = st.columns(2)

            with left_column:
                st.metric("🐱 Cat score", f"{cat_score:.1%}")
                st.progress(cat_score)

            with right_column:
                st.metric("🐶 Dog score", f"{dog_score:.1%}")
                st.progress(dog_score)

            st.caption(
                "Scores are model estimates, not calibrated probabilities. "
                "Unfamiliar images may still receive high scores."
            )

    except (UnidentifiedImageError, OSError) as error:
        st.error(f"That file could not be read as an image: {error}")

    except FileNotFoundError as error:
        st.error(str(error))

    except Exception as error:
        st.error("The app could not classify this image.")
        st.exception(error)
