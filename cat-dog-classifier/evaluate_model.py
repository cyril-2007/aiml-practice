
"""Evaluate the saved Cat vs Dog CNN model."""

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

IMG_SIZE = 128
BATCH_SIZE = 32

print("Loading the validation dataset...")

(_, val_ds), info = tfds.load(
    "cats_vs_dogs",
    split=["train[:80%]", "train[80%:]"],
    as_supervised=True,
    with_info=True,
)

def prepare_image(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

val_ds = (
    val_ds
    .map(prepare_image, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

print("Loading the saved model...")
model = tf.keras.models.load_model("cat_dog_model.keras")

true_labels = []
predicted_labels = []

print("Evaluating images...")

for images, labels in val_ds:
    scores = model.predict(images, verbose=0).ravel()
    predictions = (scores >= 0.5).astype(int)

    true_labels.extend(labels.numpy().astype(int))
    predicted_labels.extend(predictions)

true_labels = np.array(true_labels)
predicted_labels = np.array(predicted_labels)

print("\n--- Evaluation Results ---")
print(f"Images evaluated: {len(true_labels)}")
print(f"Accuracy:  {accuracy_score(true_labels, predicted_labels):.4f}")
print(
    f"Precision: {precision_score(true_labels, predicted_labels, zero_division=0):.4f}"
)
print(
    f"Recall:    {recall_score(true_labels, predicted_labels, zero_division=0):.4f}"
)
print(
    f"F1-score:  {f1_score(true_labels, predicted_labels, zero_division=0):.4f}"
)

print("\nConfusion matrix:")
print(confusion_matrix(true_labels, predicted_labels, labels=[0, 1]))

print("\nClassification report:")
print(
    classification_report(
        true_labels,
        predicted_labels,
        labels=[0, 1],
        target_names=["Cat (0)", "Dog (1)"],
        zero_division=0,
    )
)
