"""Single-image inference utility for the Streamlit app and the CLI."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from src.settings import CLASS_DISPLAY, CLASS_NAMES, IMG_SIZE, MODELS_DIR


def find_model() -> Path:
    for name in (
        "tomato_mobilenetv2_final.keras",
        "tomato_mobilenetv2_finetune.keras",
        "tomato_mobilenetv2_head.keras",
    ):
        p = MODELS_DIR / name
        if p.exists():
            return p
    raise FileNotFoundError("No trained model found in models/.")


_loaded_model: tf.keras.Model | None = None
_loaded_path: Path | None = None


def load_model() -> tf.keras.Model:
    global _loaded_model, _loaded_path
    path = find_model()
    if _loaded_model is None or _loaded_path != path:
        _loaded_model = tf.keras.models.load_model(str(path), safe_mode=False)
        _loaded_path = path
    return _loaded_model


def preprocess(image: Image.Image) -> np.ndarray:
    img = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(img, dtype=np.float32)
    return np.expand_dims(arr, axis=0)


def predict(image: Image.Image, top_k: int = 3) -> dict:
    model = load_model()
    x = preprocess(image)
    probs = model.predict(x, verbose=0)[0]
    order = np.argsort(probs)[::-1]
    top = [
        {
            "class_id": CLASS_NAMES[i],
            "label": CLASS_DISPLAY[CLASS_NAMES[i]],
            "confidence": float(probs[i]),
        }
        for i in order[:top_k]
    ]
    best = top[0]
    return {
        "predicted_class_id": best["class_id"],
        "predicted_label": best["label"],
        "confidence": best["confidence"],
        "top_k": top,
    }


def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Predict tomato leaf disease from a single image.")
    parser.add_argument("image", help="Path to a JPEG or PNG image")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    image = Image.open(args.image)
    result = predict(image, top_k=args.top_k)
    print(f"Predicted: {result['predicted_label']} ({result['confidence'] * 100:.2f}% confidence)")
    print("Top candidates:")
    for entry in result["top_k"]:
        print(f"  - {entry['label']}: {entry['confidence'] * 100:.2f}%")


if __name__ == "__main__":
    _cli()
