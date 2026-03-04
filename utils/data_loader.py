"""
Data loading and preprocessing utilities for CIFAR-10.
"""

import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical


def load_and_preprocess_data(validation_split=0.1):
    """
    Load CIFAR-10 data, normalize pixel values, one-hot encode labels,
    and split off a validation set from the training data.

    Returns:
        (x_train, y_train), (x_val, y_val), (x_test, y_test)
    """
    (x_train_full, y_train_full), (x_test, y_test) = cifar10.load_data()

    # Normalize pixel values to 0-1
    x_train_full = x_train_full.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # One-hot encode labels
    y_train_full = to_categorical(y_train_full, 10)
    y_test = to_categorical(y_test, 10)

    # Split validation set from training data
    num_val = int(len(x_train_full) * validation_split)
    indices = np.random.RandomState(42).permutation(len(x_train_full))

    val_indices = indices[:num_val]
    train_indices = indices[num_val:]

    x_train = x_train_full[train_indices]
    y_train = y_train_full[train_indices]
    x_val = x_train_full[val_indices]
    y_val = y_train_full[val_indices]

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def get_class_names():
    """Return the CIFAR-10 class names."""
    return [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]


def get_class_distribution(y, class_names=None):
    """Get the distribution of classes in a dataset."""
    if class_names is None:
        class_names = get_class_names()

    if len(y.shape) > 1:
        y = np.argmax(y, axis=1)

    unique, counts = np.unique(y, return_counts=True)
    distribution = {}
    for idx, count in zip(unique, counts):
        distribution[class_names[idx]] = count

    return distribution