"""
Central configuration for the ML Image Classifier.
All hyperparameters and file paths are defined here.
"""

CONFIG = {
    # Data
    "input_shape": (32, 32, 3),
    "num_classes": 10,
    "validation_split": 0.1,

    # Training
    "batch_size": 64,
    "epochs": 100,
    "learning_rate": 0.001,

    # Paths
    "model_dir": "models",
    "results_dir": "results",
}
