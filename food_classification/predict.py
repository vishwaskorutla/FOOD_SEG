"""
Run predictions with a trained Indian food classifier.

Usage:
    python predict.py path/to/image.jpg

Before running, edit the CONFIG section below to point at your local
model file and the dataset folder (used only to recover the class names).
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import kagglehub
from tensorflow.keras.preprocessing import image

# ----------------------------- CONFIG ------------------------------------
MODEL_PATH = "./indian_food_model_v2.keras"  # <-- change to your local path
DATASET_DIR = "./data/Indian Food Images"
if not os.path.exists(DATASET_DIR):
    try:
        print("Local dataset not found. Locating/downloading via kagglehub...")
        _download_path = kagglehub.dataset_download("iamsouravbanerjee/indian-food-images-dataset")
        DATASET_DIR = os.path.join(_download_path, "Indian Food Images")
        print(f"Dataset path: {DATASET_DIR}")
    except Exception as e:
        print(f"Warning: Failed to load dataset via kagglehub: {e}")
IMG_SIZE = (300, 300)  # must match the size the model was trained on
# ---------------------------------------------------------------------------


def load_class_labels(dataset_dir: str):
    if os.path.exists(dataset_dir):
        labels = sorted(
            d for d in os.listdir(dataset_dir)
            if os.path.isdir(os.path.join(dataset_dir, d))
        )
        print(f"Loaded {len(labels)} class labels.")
        return labels
    print("Dataset directory not found — predictions will show class IDs instead of names.")
    return []


def detect_food(model, class_labels, image_path: str, show_plot: bool = True):
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return None

    img = image.load_img(image_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_batch, verbose=0)
    predicted_index = int(np.argmax(predictions[0]))
    confidence = float(np.max(predictions[0]) * 100)

    if class_labels:
        predicted_food = class_labels[predicted_index]
    else:
        predicted_food = f"Class ID {predicted_index}"

    print(f"Predicted: {predicted_food} ({confidence:.2f}%)")

    if show_plot:
        plt.figure(figsize=(6, 6))
        plt.imshow(img_array.astype("uint8"))
        plt.axis("off")
        plt.title(f"Detected: {predicted_food} ({confidence:.2f}%)", fontsize=16, color="darkblue")
        plt.show()

    return predicted_food


def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py path/to/image.jpg")
        sys.exit(1)

    image_path = sys.argv[1]

    print(f"Loading model from {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully!")

    class_labels = load_class_labels(DATASET_DIR)
    detect_food(model, class_labels, image_path)


if __name__ == "__main__":
    main()
