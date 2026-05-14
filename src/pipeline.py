"""tf.data pipelines built from the prepared index.csv."""
from __future__ import annotations

import pandas as pd
import numpy as np
import tensorflow as tf

from src.augment import train_augment
from src.settings import CFG, CLASS_NAMES, IMG_SIZE, PREPARED_DIR

AUTOTUNE = tf.data.AUTOTUNE
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}


def _read_index() -> pd.DataFrame:
    path = PREPARED_DIR / "index.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"index.csv not found at {path}. Run: python -m src.dataset"
        )
    return pd.read_csv(path)


def _decode_image(filepath: tf.Tensor) -> tf.Tensor:
    raw = tf.io.read_file(filepath)
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    return tf.cast(img, tf.uint8)


def _augment_tf(img: tf.Tensor) -> tf.Tensor:
    def _np(x):
        return train_augment(x)
    out = tf.numpy_function(_np, [img], tf.uint8)
    out.set_shape([IMG_SIZE, IMG_SIZE, 3])
    return out


def _build_ds(df: pd.DataFrame, training: bool, batch_size: int) -> tf.data.Dataset:
    filepaths = df["filepath"].astype(str).tolist()
    labels = [CLASS_TO_IDX[c] for c in df["class"].tolist()]
    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    if training:
        ds = ds.shuffle(buffer_size=min(2048, len(filepaths)), seed=42, reshuffle_each_iteration=True)
    def _map(fp, y):
        img = _decode_image(fp)
        if training:
            img = _augment_tf(img)
        img = tf.cast(img, tf.float32)
        return img, y
    ds = ds.map(_map, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds


def get_datasets(batch_size: int | None = None) -> dict:
    if batch_size is None:
        batch_size = int(CFG["training"]["batch_size"])
    df = _read_index()
    splits = {
        "train": df[df["split"] == "train"],
        "val": df[df["split"] == "val"],
        "test_controlled": df[df["split"] == "test_controlled"],
    }
    field_real = df[df["split"] == "test_field"]
    field_sim = df[df["split"] == "test_field_simulated"]

    out = {
        "train": _build_ds(splits["train"], training=True, batch_size=batch_size),
        "val": _build_ds(splits["val"], training=False, batch_size=batch_size),
        "test_controlled": _build_ds(splits["test_controlled"], training=False, batch_size=batch_size),
        "counts": {k: len(v) for k, v in splits.items()},
        "field_kind": None,
    }
    if len(field_real) > 0:
        out["test_field"] = _build_ds(field_real, training=False, batch_size=batch_size)
        out["counts"]["test_field"] = len(field_real)
        out["field_kind"] = "real"
    elif len(field_sim) > 0:
        out["test_field"] = _build_ds(field_sim, training=False, batch_size=batch_size)
        out["counts"]["test_field"] = len(field_sim)
        out["field_kind"] = "simulated"
    return out


def compute_class_weights(df: pd.DataFrame | None = None) -> dict[int, float]:
    if df is None:
        df = _read_index()
    train = df[df["split"] == "train"]
    counts = train["class"].value_counts().to_dict()
    total = sum(counts.values())
    num = len(CLASS_NAMES)
    weights = {}
    for cls, idx in CLASS_TO_IDX.items():
        c = counts.get(cls, 0)
        weights[idx] = (total / (num * c)) if c > 0 else 1.0
    return weights
