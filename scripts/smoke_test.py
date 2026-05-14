"""
Quick sanity check that the pipeline wiring is correct without needing
the full dataset. Builds a tiny synthetic dataset of 3 classes x 12 images
in data/raw/_smoke, swaps it in, and runs one forward pass through the
training pipeline + a single batch fit.

Run:
  python scripts\\smoke_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.settings import CLASS_NAMES, IMG_SIZE, PREPARED_DIR, PROJECT_ROOT  # noqa: E402


SMOKE_DIR = PROJECT_ROOT / "data" / "raw" / "_smoke"


def make_dataset() -> None:
    rng = np.random.default_rng(0)
    SMOKE_DIR.mkdir(parents=True, exist_ok=True)
    for cls in CLASS_NAMES[:3]:
        d = SMOKE_DIR / cls
        d.mkdir(parents=True, exist_ok=True)
        for i in range(12):
            arr = rng.integers(0, 255, (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            Image.fromarray(arr).save(d / f"{i:02d}.jpg")
    print(f"[ok] wrote synthetic images to {SMOKE_DIR}")


def main() -> None:
    make_dataset()

    import src.settings as s
    s.RAW_DIR = SMOKE_DIR
    s.CLASS_NAMES = CLASS_NAMES[:3]
    s.NUM_CLASSES = 3

    import importlib
    import src.dataset as dataset_mod
    importlib.reload(dataset_mod)
    dataset_mod.CLASS_NAMES = CLASS_NAMES[:3]
    dataset_mod.RAW_DIR = SMOKE_DIR
    dataset_mod.build_index()

    import src.pipeline as pl
    importlib.reload(pl)
    pl.CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES[:3])}
    ds = pl.get_datasets(batch_size=4)
    print(f"[ok] split counts: {ds['counts']}")

    import src.model_builder as mb
    importlib.reload(mb)
    model = mb.build_model(num_classes=3)
    import tensorflow as tf
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(ds["train"], validation_data=ds["val"], epochs=1, verbose=2)
    print("[ok] smoke test passed")

    index = PREPARED_DIR / "index.csv"
    if index.exists():
        index.unlink()


if __name__ == "__main__":
    main()
