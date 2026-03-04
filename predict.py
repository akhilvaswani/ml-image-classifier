"""
ML Image Classifier - Prediction Script
Load a trained model and classify images.
"""

import argparse
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt

from config import CONFIG


CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def preprocess_image(image_path):
    """Load and preprocess a single image for prediction."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((32, 32))
    img_array = np.array(img).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array, img


def predict_image(model, image_path):
    """Run prediction on a single image."""
    img_array, original_img = preprocess_image(image_path)
    predictions = model.predict(img_array, verbose=0)[0]

    predicted_class = np.argmax(predictions)
    confidence = predictions[predicted_class]

    return {
        "class": CLASS_NAMES[predicted_class],
        "confidence": float(confidence),
        "all_probabilities": {
            CLASS_NAMES[i]: float(predictions[i])
            for i in range(len(CLASS_NAMES))
        },
        "original_image": original_img
    }


def visualize_prediction(result, save_path=None):
    """Show the image alongside prediction probabilities."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Show the image
    ax1.imshow(result["original_image"])
    ax1.set_title(
        f"Predicted: {result['class']} ({result['confidence']:.1%})",
        fontsize=14
    )
    ax1.axis("off")

    # Show probability bar chart
    classes = list(result["all_probabilities"].keys())
    probs = list(result["all_probabilities"].values())
    colors = ["#2196F3" if p < max(probs) else "#4CAF50" for p in probs]

    ax2.barh(classes, probs, color=colors)
    ax2.set_xlabel("Confidence")
    ax2.set_title("Prediction Probabilities")
    ax2.set_xlim(0, 1)

    for i, prob in enumerate(probs):
        if prob > 0.01:
            ax2.text(prob + 0.01, i, f"{prob:.1%}", va="center", fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Visualization saved to: {save_path}")
    else:
        plt.show()

    plt.close()


def predict_batch(model, image_dir):
    """Run predictions on all images in a directory."""
    supported_formats = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
    results = []

    for filename in sorted(os.listdir(image_dir)):
        ext = os.path.splitext(filename)[1].lower()
        if ext in supported_formats:
            image_path = os.path.join(image_dir, filename)
            result = predict_image(model, image_path)
            result["filename"] = filename
            results.append(result)
            print(f"  {filename}: {result['class']} ({result['confidence']:.1%})")

    return results


def main():
    parser = argparse.ArgumentParser(description="Classify images using trained CNN model")
    parser.add_argument("--image", type=str, help="Path to a single image to classify")
    parser.add_argument("--dir", type=str, help="Path to a directory of images to classify")
    parser.add_argument("--model", type=str,
                        default=os.path.join(CONFIG["model_dir"], "best_model.keras"),
                        help="Path to the trained model file")
    parser.add_argument("--visualize", action="store_true",
                        help="Show visualization of prediction probabilities")
    parser.add_argument("--output", type=str, default=None,
                        help="Save visualization to this path instead of showing it")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        print("Run train.py first to train the model.")
        return

    print(f"Loading model from {args.model}...")
    model = load_model(args.model)

    if args.image:
        if not os.path.exists(args.image):
            print(f"Error: Image not found at {args.image}")
            return

        print(f"\nClassifying: {args.image}")
        result = predict_image(model, args.image)
        print(f"  Prediction: {result['class']}")
        print(f"  Confidence: {result['confidence']:.1%}")
        print(f"\n  All probabilities:")
        for cls, prob in sorted(result["all_probabilities"].items(),
                                key=lambda x: x[1], reverse=True):
            bar = "#" * int(prob * 40)
            print(f"    {cls:12s} {prob:6.1%} {bar}")

        if args.visualize:
            visualize_prediction(result, save_path=args.output)

    elif args.dir:
        if not os.path.isdir(args.dir):
            print(f"Error: Directory not found at {args.dir}")
            return

        print(f"\nClassifying all images in: {args.dir}")
        results = predict_batch(model, args.dir)
        print(f"\nClassified {len(results)} images")

    else:
        print("Please provide --image or --dir argument")
        parser.print_help()


if __name__ == "__main__":
    main()