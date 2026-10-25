import tensorflow_datasets as tfds

print("Downloading Cats vs Dogs dataset...")

dataset, info = tfds.load(
    "cats_vs_dogs",
    split="train",
    with_info=True,
    as_supervised=True
)

print("\nDataset downloaded successfully!")
print("Number of examples:", info.splits["train"].num_examples)