"""
Optional helper: synthesise a 'field-style' test set by applying heavy
distortion to a held-out PlantVillage slice. Writes images into
data/field/<class>/ so the index builder treats them as a real field set.

If you have genuine field photos, drop them directly into
  data/field/<class_name>/<image.jpg>
and skip this script.

Run from the Anaconda Prompt:
  python scripts\\prepare_field_split.py --per-class 50
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.augment import field_distortion  # noqa: E402
from src.settings import CLASS_NAMES, FIELD_DIR, IMG_SIZE, RAW_DIR  # noqa: E402

VALID_EXT = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def synthesise(per_class: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    FIELD_DIR.mkdir(parents=True, exist_ok=True)
    for cls in CLASS_NAMES:
        src_dir = RAW_DIR / cls
        if not src_dir.exists():
            print(f"[warn] {cls} missing in raw dataset; skipping")
            continue
        files = sorted(p for p in src_dir.iterdir() if p.suffix in VALID_EXT)
        rng.shuffle(files)
        out_dir = FIELD_DIR / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        n = min(per_class, len(files))
        for i, f in enumerate(files[:n]):
            img = cv2.imread(str(f))
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            distorted = field_distortion(img, rng=rng)
            cv2.imwrite(str(out_dir / f"sim_{i:04d}.jpg"), distorted)
        print(f"[ok] wrote {n} simulated field images for {cls}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    synthesise(args.per_class, args.seed)


if __name__ == "__main__":
    main()
