"""
Training-time augmentation pipeline matching revised.md s3.20:
rotation +/-30, H/V flips, brightness +/-20%, contrast +/-20%,
random crop with resize, low-probability blur.

A heavier `field_distortion_pipeline` is used by prepare_field_split.py
to synthesize a field-style test set when no real field images are
provided.
"""
from __future__ import annotations

import cv2
import numpy as np

from src.settings import IMG_SIZE


def _rand(a: float, b: float, rng: np.random.Generator) -> float:
    return float(rng.uniform(a, b))


def train_augment(img: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    h, w = img.shape[:2]

    if rng.random() < 0.5:
        img = cv2.flip(img, 1)
    if rng.random() < 0.3:
        img = cv2.flip(img, 0)

    angle = _rand(-30.0, 30.0, rng)
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    img = cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)

    crop_frac = _rand(0.80, 1.0, rng)
    ch, cw = int(h * crop_frac), int(w * crop_frac)
    y = rng.integers(0, max(1, h - ch + 1))
    x = rng.integers(0, max(1, w - cw + 1))
    img = img[y:y + ch, x:x + cw]
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    brightness = _rand(-0.20, 0.20, rng)
    contrast = _rand(0.80, 1.20, rng)
    img = img.astype(np.float32) / 255.0
    img = np.clip((img - 0.5) * contrast + 0.5 + brightness, 0.0, 1.0)
    img = (img * 255.0).astype(np.uint8)

    if rng.random() < 0.15:
        k = int(rng.choice([3, 5]))
        img = cv2.GaussianBlur(img, (k, k), 0)

    return img


def field_distortion(img: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    """Heavier transform used to synthesize a field-style test set."""
    if rng is None:
        rng = np.random.default_rng()
    h, w = img.shape[:2]

    angle = _rand(-45.0, 45.0, rng)
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    img = cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REFLECT)

    brightness = _rand(-0.35, 0.35, rng)
    contrast = _rand(0.6, 1.4, rng)
    f = img.astype(np.float32) / 255.0
    f = np.clip((f - 0.5) * contrast + 0.5 + brightness, 0.0, 1.0)
    img = (f * 255.0).astype(np.uint8)

    if rng.random() < 0.7:
        k = int(rng.choice([5, 7, 9]))
        img = cv2.GaussianBlur(img, (k, k), 0)

    if rng.random() < 0.4:
        noise = rng.normal(0, 12, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    if rng.random() < 0.5:
        oy = rng.integers(0, h // 3)
        ox = rng.integers(0, w // 3)
        oh = rng.integers(h // 6, h // 3)
        ow = rng.integers(w // 6, w // 3)
        img[oy:oy + oh, ox:ox + ow] = rng.integers(0, 80, (oh, ow, 3))

    return img
