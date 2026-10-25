
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import sys

IMG_SIZE = 128

# Load the trained model
model = keras.models.load_model("cat_dog_model.keras")

# Get the image path
if len(sys.argv) > 1:
    image_path = Path(sys.argv[1])
else:
    image_path = Path("test_images/my_pet.jpg")

if not image_path.exists():
    print("Image not found:", image_path)
    print("Please check the image filename and path.")
    sys.exit(1)

# Load and prepare the image
image = tf.keras.utils.load_img(
    image_path,
    target_size=(IMG_SIZE, IMG_SIZE)
)

image_array = tf.keras.utils.img_to_array(image)
image_array = image_array / 255.0
image_array = tf.expand_dims(image_array, axis=0)

# Predict
prediction = model.predict(image_array, verbose=0)[0][0]

print("\n--- Cat vs Dog Prediction ---")

if prediction >= 0.5:
    print("Prediction: DOG 🐶")
    confidence = prediction * 100
else:
    print("Prediction: CAT 🐱")
    confidence = (1 - prediction) * 100

print(f"Model confidence score: {confidence:.2f}%")
