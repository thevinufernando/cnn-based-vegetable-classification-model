"""
Inference script to classify any single vegetable image or folder of images.

Usage:
    python scripts/predict.py --image "data/processed/test/Tomato/test_0001.png" --model b
    python scripts/predict.py --image "path/to/any_vegetable.jpg" --model a
    python scripts/predict.py --image "path/to/any_vegetable.jpg" --model mobilenet
"""

import argparse
import sys
import time
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf
from tensorflow import keras

CLASS_NAMES = [
    "Bean", "Bitter_Gourd", "Bottle_Gourd", "Brinjal", "Broccoli",
    "Cabbage", "Capsicum", "Carrot", "Cauliflower", "Cucumber",
    "Papaya", "Potato", "Pumpkin", "Radish", "Tomato"
]

MODEL_MAP = {
    "a": "model_a_standard.keras",
    "b": "model_b_separable.keras",
    "mobilenet": "mobilenet_v2.keras",
    "efficientnet": "efficientnet_b0.keras",
}


def load_and_preprocess_image(image_path: Path | str, target_size=(64, 64)) -> np.ndarray:
    """Load image from disk, convert to RGB, resize to target_size, return batch tensor."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size, Image.Resampling.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    # Shape: (1, 64, 64, 3)
    return np.expand_dims(arr, axis=0)


def predict(image_path: Path | str, model_key: str = "b", top_k: int = 3):
    """Run prediction on an image and print top-k results with probabilities."""
    model_file = MODEL_MAP.get(model_key.lower())
    if not model_file:
        raise ValueError(f"Unknown model '{model_key}'. Available: {list(MODEL_MAP.keys())}")

    model_path = ROOT / "models" / model_file
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}. Train the model first.")

    print(f"\nLoading model: {model_path.name}...")
    model = keras.models.load_model(model_path)

    img_tensor = load_and_preprocess_image(image_path)

    # Benchmark prediction latency
    t0 = time.perf_counter()
    preds = model.predict(img_tensor, verbose=0)[0]
    latency_ms = (time.perf_counter() - t0) * 1000.0

    top_indices = np.argsort(preds)[::-1][:top_k]

    print("\n" + "=" * 50)
    print(f"Classification Results for: {Path(image_path).name}")
    print("=" * 50)
    print(f"Predicted Class : {CLASS_NAMES[top_indices[0]]} ({preds[top_indices[0]] * 100:.2f}%)")
    print(f"Inference Time  : {latency_ms:.2f} ms")
    print("\nTop Candidates:")
    for rank, idx in enumerate(top_indices, 1):
        print(f"  {rank}. {CLASS_NAMES[idx]:<15} {preds[idx] * 100:>6.2f}%")
    print("=" * 50 + "\n")

    return CLASS_NAMES[top_indices[0]], preds[top_indices[0]]


def main():
    parser = argparse.ArgumentParser(description="Classify a vegetable image using trained CNNs.")
    parser.add_argument("--image", type=Path, required=True, help="Path to input image")
    parser.add_argument("--model", choices=["a", "b", "mobilenet", "efficientnet"], default="b",
                        help="Model architecture to use (default: b)")
    parser.add_argument("--top-k", type=int, default=3, help="Number of top classes to display")
    args = parser.parse_args()

    predict(args.image, model_key=args.model, top_k=args.top_k)


if __name__ == "__main__":
    main()
