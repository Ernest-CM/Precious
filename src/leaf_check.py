"""
Heuristic gate that rejects images that clearly are not tomato leaves.

The trained classifier is closed-set (10 classes, softmax-normalised) so it
has no way to say "this isn't a leaf at all" - it will confidently misclassify
anything you give it. This module computes a simple leaf-likeness score from
the image's colour distribution so the UI can refuse to run inference on
obvious non-leaves.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def leaf_color_ratio(image: Image.Image) -> float:
    """Fraction of pixels in the leaf colour range (greens, yellows, browns)
    with non-trivial saturation and brightness."""
    arr = np.asarray(image.convert("RGB"))
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    mask = (h >= 15) & (h <= 95) & (s >= 30) & (v >= 30)
    return float(mask.mean())


def assess(image: Image.Image) -> dict:
    ratio = leaf_color_ratio(image)
    if ratio < 0.15:
        verdict = "reject"
    elif ratio < 0.30:
        verdict = "warn"
    else:
        verdict = "ok"
    return {"ratio": ratio, "verdict": verdict}
