"""
ML Image Classifier - Training Script
Trains a CNN on the CIFAR-10 dataset with data augmentation and callbacks.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense, Flatten,
    Dropout, BatchNormalization
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
)
from tensorflow.keras.optimizers import Adam

from config import CONFIG
from utils.data_loader import load_and_preprocess_data
from utils.visualization import plot_training_history


def build_model(input_shape, num_classes):
    """
    Build a CNN with three convolutional blocks.
    Each block has two Conv2D layers, batch normalization,
    max pooling, and dropout.
    """
    model = Sequential()

    # Block 1
    model.add(Conv2D(32, (3, 3), padding="same", activation="relu",
                     input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Block 2
    model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Block 3
    model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Dense layers
    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="softmax"))

    return model


def create_data_augmentation():
    """Set up real-time data augmentation for training."""
    return ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )


def get_callbacks():
    """Set up training callbacks."""
    os.makedirs(CONFIG["model_dir"], exist_ok=True)
    model_path = os.path.join(CONFIG["model_dir"], "best_model.keras")

    callbacks = [
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=model_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        )
    ]
    return callbacks


def main():
    print("Loading and preprocessing CIFAR-10 data...")
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_and_preprocess_data(
        validation_split=CONFIG["validation_split"]
    )

    print(f"Training samples: {len(x_train)}")
    print(f"Validation samples: {len(x_val)}")
    print(f"Test samples: {len(x_test)}")

    print("\nBuilding model...")
    model = build_model(
        input_shape=CONFIG["input_shape"],
        num_classes=CONFIG["num_classes"]
    )
    model.summary()

    optimizer = Adam(learning_rate=CONFIG["learning_rate"])
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("\nSetting up data augmentation...")
    datagen = create_data_augmentation()
    datagen.fit(x_train)

    print("\nTraining model...")
    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=CONFIG["batch_size"]),
        epochs=CONFIG["epochs"],
        validation_data=(x_val, y_val),
        callbacks=get_callbacks(),
        steps_per_epoch=len(x_train) // CONFIG["batch_size"],
        verbose=1
    )

    print("\nEvaluating on test set...")
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    print("\nSaving training history plots...")
    os.makedirs(CONFIG["results_dir"], exist_ok=True)
    plot_training_history(
        history,
        save_path=os.path.join(CONFIG["results_dir"], "training_history.png")
    )

    print("\nDone! Model saved to:", os.path.join(CONFIG["model_dir"], "best_model.keras"))


if __name__ == "__main__":
    main()