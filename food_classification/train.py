"""
Train an EfficientNetB3-based Indian food classifier.

Usage:
    python train.py

Before running, edit the CONFIG section below to point at your local
copy of the "Indian Food Images" dataset folder (a directory containing
one sub-folder per food class, each full of images of that food).
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import kagglehub
from tensorflow.keras import regularizers
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ----------------------------- CONFIG ------------------------------------
DATASET_DIR = "./data/Indian Food Images"
if not os.path.exists(DATASET_DIR):
    try:
        print("Local dataset not found. Locating/downloading via kagglehub...")
        _download_path = kagglehub.dataset_download("iamsouravbanerjee/indian-food-images-dataset")
        DATASET_DIR = os.path.join(_download_path, "Indian Food Images")
        print(f"Dataset path: {DATASET_DIR}")
    except Exception as e:
        print(f"Warning: Failed to load dataset via kagglehub: {e}")
IMG_SIZE = (300, 300)
BATCH_SIZE = 16
INITIAL_LR = 3e-4
FINE_TUNE_LR = 1e-5
INITIAL_EPOCHS = 150
FINE_TUNE_EPOCHS = 50
MODEL_OUT_PATH = "indian_food_model_v2.keras"
# ---------------------------------------------------------------------------


def build_dataframe(dataset_dir: str) -> pd.DataFrame:
    """Walk the dataset folder and build a filepaths/labels dataframe."""
    filepaths, labels = [], []
    for folder in os.listdir(dataset_dir):
        folder_path = os.path.join(dataset_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        for file in os.listdir(folder_path):
            filepaths.append(os.path.join(folder_path, file))
            labels.append(folder)

    df = pd.DataFrame({"filepaths": filepaths, "labels": labels})
    return df.sample(frac=1).reset_index(drop=True)


def split_dataframe(df: pd.DataFrame):
    train_df = df.sample(frac=0.8)
    temp_df = df.drop(train_df.index)
    valid_df = temp_df.sample(frac=0.5)
    test_df = temp_df.drop(valid_df.index)
    return train_df, valid_df, test_df


def make_generators(train_df, valid_df, test_df, all_classes):
    train_datagen = ImageDataGenerator(
        rotation_range=30,
        zoom_range=0.3,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
    )
    eval_datagen = ImageDataGenerator()

    train_gen = train_datagen.flow_from_dataframe(
        train_df,
        x_col="filepaths",
        y_col="labels",
        target_size=IMG_SIZE,
        class_mode="categorical",
        batch_size=BATCH_SIZE,
        classes=all_classes,
    )
    valid_gen = eval_datagen.flow_from_dataframe(
        valid_df,
        x_col="filepaths",
        y_col="labels",
        target_size=IMG_SIZE,
        class_mode="categorical",
        batch_size=BATCH_SIZE,
        classes=all_classes,
    )
    test_gen = eval_datagen.flow_from_dataframe(
        test_df,
        x_col="filepaths",
        y_col="labels",
        target_size=IMG_SIZE,
        class_mode="categorical",
        batch_size=BATCH_SIZE,
        classes=all_classes,
        shuffle=False,
    )
    return train_gen, valid_gen, test_gen


def make_model(img_size, lr, class_count):
    base_model = EfficientNetB3(
        include_top=False,
        weights="imagenet",
        input_shape=(img_size[0], img_size[1], 3),
        pooling="avg",
    )
    base_model.trainable = False

    x = base_model.output
    x = BatchNormalization()(x)
    x = Dense(256, activation="relu", kernel_regularizer=regularizers.l2(0.01))(x)
    x = Dropout(0.4)(x)
    output = Dense(class_count, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    df = build_dataframe(DATASET_DIR)
    all_classes = sorted(df["labels"].unique().tolist())
    train_df, valid_df, test_df = split_dataframe(df)

    train_gen, valid_gen, test_gen = make_generators(
        train_df, valid_df, test_df, all_classes
    )

    class_count = len(train_gen.class_indices)
    
    # Matching Kaggle make_model exactly
    model = make_model(IMG_SIZE, INITIAL_LR, class_count)
    model.summary()

    checkpoint = ModelCheckpoint(
        filepath=MODEL_OUT_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    )
    callbacks = [
        ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, verbose=1),
        EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
        ),
        checkpoint,
    ]

    # Stage 1: Train the model head exactly as in Kaggle code for 200 epochs
    model.fit(
        train_gen,
        validation_data=valid_gen,
        epochs=INITIAL_EPOCHS + FINE_TUNE_EPOCHS,  # Total 200 epochs
        callbacks=callbacks,
    )

    # Final evaluation on the held-out test set
    test_loss, test_accuracy = model.evaluate(test_gen)
    print(f"Final Test Accuracy: {test_accuracy * 100:.2f}%")

    model.save(MODEL_OUT_PATH)
    print(f"Model saved to {MODEL_OUT_PATH}")


if __name__ == "__main__":
    main()
