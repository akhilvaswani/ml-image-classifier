"""
ML Image Classifier - Evaluation Script
Generates detailed evaluation metrics, confusion matrix, and classification report.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from config import CONFIG
from utils.data_loader import load_and_preprocess_data
from utils.visualization import plot_confusion_matrix, plot_per_class_accuracy


CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def evaluate_model(model, x_test, y_test):
    """Run full evaluation on the test set."""
    # Get overall metrics
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

    # Get predictions
    y_pred_proba = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # Classification report
    report = classification_report(
        y_true, y_pred,
        target_names=CLASS_NAMES,
        output_dict=True
    )
    report_text = classification_report(
        y_true, y_pred,
        target_names=CLASS_NAMES
    )

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Per-class accuracy
    per_class_acc = {}
    for i, name in enumerate(CLASS_NAMES):
        mask = y_true == i
        class_acc = np.mean(y_pred[mask] == y_true[mask])
        per_class_acc[name] = class_acc

    return {
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "classification_report": report,
        "classification_report_text": report_text,
        "confusion_matrix": cm,
        "per_class_accuracy": per_class_acc,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_pred_proba": y_pred_proba
    }


def print_results(results):
    """Print evaluation results to console."""
    print("=" * 60)
    print("MODEL EVALUATION RESULTS")
    print("=" * 60)
    print(f"\nTest Loss:     {results['test_loss']:.4f}")
    print(f"Test Accuracy: {results['test_accuracy']:.4f}")

    print(f"\n{'=' * 60}")
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(results["classification_report_text"])

    print(f"\n{'=' * 60}")
    print("PER-CLASS ACCURACY")
    print("=" * 60)
    sorted_classes = sorted(
        results["per_class_accuracy"].items(),
        key=lambda x: x[1],
        reverse=True
    )
    for name, acc in sorted_classes:
        bar = "#" * int(acc * 40)
        print(f"  {name:12s} {acc:6.1%} {bar}")

    # Find most confused pairs
    cm = results["confusion_matrix"]
    np.fill_diagonal(cm, 0)
    worst_idx = np.unravel_index(cm.argmax(), cm.shape)
    print(f"\nMost confused pair: {CLASS_NAMES[worst_idx[0]]} -> {CLASS_NAMES[worst_idx[1]]} "
          f"({cm[worst_idx[0], worst_idx[1]]} misclassifications)")


def save_results(results, results_dir):
    """Save all evaluation visualizations."""
    os.makedirs(results_dir, exist_ok=True)

    # Confusion matrix
    plot_confusion_matrix(
        results["confusion_matrix"],
        CLASS_NAMES,
        save_path=os.path.join(results_dir, "confusion_matrix.png")
    )

    # Per-class accuracy
    plot_per_class_accuracy(
        results["per_class_accuracy"],
        save_path=os.path.join(results_dir, "per_class_accuracy.png")
    )

    # Save misclassified examples
    save_misclassified_examples(
        results,
        save_path=os.path.join(results_dir, "misclassified_examples.png")
    )

    print(f"\nResults saved to: {results_dir}/")


def save_misclassified_examples(results, save_path, num_examples=16):
    """Save a grid of misclassified images."""
    # Reload test data for visualization
    (_, _), (_, _), (x_test, _) = load_and_preprocess_data(
        validation_split=CONFIG["validation_split"]
    )

    misclassified_idx = np.where(results["y_true"] != results["y_pred"])[0]

    if len(misclassified_idx) == 0:
        print("No misclassified examples found!")
        return

    sample_idx = np.random.choice(
        misclassified_idx,
        size=min(num_examples, len(misclassified_idx)),
        replace=False
    )

    rows = int(np.ceil(num_examples / 4))
    fig, axes = plt.subplots(rows, 4, figsize=(12, 3 * rows))
    axes = axes.flatten()

    for i, idx in enumerate(sample_idx):
        axes[i].imshow(x_test[idx])
        true_label = CLASS_NAMES[results["y_true"][idx]]
        pred_label = CLASS_NAMES[results["y_pred"][idx]]
        conf = results["y_pred_proba"][idx][results["y_pred"][idx]]
        axes[i].set_title(
            f"True: {true_label}\nPred: {pred_label} ({conf:.0%})",
            fontsize=9, color="red"
        )
        axes[i].axis("off")

    # Hide extra subplots
    for i in range(len(sample_idx), len(axes)):
        axes[i].axis("off")

    plt.suptitle("Misclassified Examples", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved misclassified examples to: {save_path}")


def main():
    model_path = os.path.join(CONFIG["model_dir"], "best_model.keras")

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Run train.py first to train the model.")
        return

    print("Loading model...")
    model = load_model(model_path)

    print("Loading test data...")
    (_, _), (_, _), (x_test, y_test) = load_and_preprocess_data(
        validation_split=CONFIG["validation_split"]
    )

    print("Evaluating model...")
    results = evaluate_model(model, x_test, y_test)

    print_results(results)
    save_results(results, CONFIG["results_dir"])


if __name__ == "__main__":
    main()