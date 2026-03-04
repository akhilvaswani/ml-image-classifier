"""
Visualization utilities for training history, confusion matrices,
and prediction results.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_training_history(history, save_path=None):
    """Plot training and validation accuracy/loss curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    ax1.plot(history.history["accuracy"], label="Training", linewidth=2)
    ax1.plot(history.history["val_accuracy"], label="Validation", linewidth=2)
    ax1.set_title("Model Accuracy", fontsize=14)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)

    # Loss
    ax2.plot(history.history["loss"], label="Training", linewidth=2)
    ax2.plot(history.history["val_loss"], label="Validation", linewidth=2)
    ax2.set_title("Model Loss", fontsize=14)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Training history saved to: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_confusion_matrix(cm, class_names, save_path=None):
    """Plot a confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )

    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Confusion matrix saved to: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_per_class_accuracy(per_class_acc, save_path=None):
    """Plot a horizontal bar chart of per-class accuracy."""
    sorted_items = sorted(per_class_acc.items(), key=lambda x: x[1])
    classes = [item[0] for item in sorted_items]
    accuracies = [item[1] for item in sorted_items]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.RdYlGn(np.array(accuracies))
    bars = ax.barh(classes, accuracies, color=colors)

    for bar, acc in zip(bars, accuracies):
        ax.text(acc + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{acc:.1%}", va="center", fontsize=10)

    ax.set_xlabel("Accuracy", fontsize=12)
    ax.set_title("Per-Class Accuracy", fontsize=14)
    ax.set_xlim(0, 1.05)
    ax.axvline(x=np.mean(accuracies), color="gray",
               linestyle="--", alpha=0.7, label=f"Mean: {np.mean(accuracies):.1%}")
    ax.legend(fontsize=11)
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Per-class accuracy chart saved to: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_sample_predictions(images, true_labels, pred_labels, confidences,
                            class_names, num_samples=16, save_path=None):
    """Plot a grid of sample predictions."""
    rows = int(np.ceil(num_samples / 4))
    fig, axes = plt.subplots(rows, 4, figsize=(12, 3 * rows))
    axes = axes.flatten()

    for i in range(min(num_samples, len(images))):
        axes[i].imshow(images[i])
        correct = true_labels[i] == pred_labels[i]
        color = "green" if correct else "red"
        title = f"True: {class_names[true_labels[i]]}\n"
        title += f"Pred: {class_names[pred_labels[i]]} ({confidences[i]:.0%})"
        axes[i].set_title(title, fontsize=9, color=color)
        axes[i].axis("off")

    for i in range(len(images), len(axes)):
        axes[i].axis("off")

    plt.suptitle("Sample Predictions", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Sample predictions saved to: {save_path}")
    else:
        plt.show()

    plt.close()