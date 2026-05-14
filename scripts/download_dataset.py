"""
Downloads the PlantVillage tomato subset from Kaggle and lays it out as:

  data/raw/PlantVillage/<class_name>/<image.jpg>

Prerequisite: Kaggle API credentials. Place kaggle.json at
  Windows: %USERPROFILE%\\.kaggle\\kaggle.json
Then run from the Anaconda Prompt:

  python scripts\\download_dataset.py
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.settings import CLASS_NAMES, RAW_DIR  # noqa: E402

DATASET_SLUG = "abdallahalidev/plantvillage-dataset"
DOWNLOAD_DIR = ROOT / "data" / "raw" / "_kaggle"


def run() -> None:
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[info] downloading {DATASET_SLUG} to {DOWNLOAD_DIR}")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "kaggle",
            "datasets",
            "download",
            "-d",
            DATASET_SLUG,
            "-p",
            str(DOWNLOAD_DIR),
            "--unzip",
        ],
        check=True,
    )

    candidates = [DOWNLOAD_DIR, DOWNLOAD_DIR / "plantvillage dataset"]
    color_root = None
    for c in candidates:
        guess = c / "color"
        if guess.exists():
            color_root = guess
            break
    if color_root is None:
        for sub in DOWNLOAD_DIR.rglob("color"):
            if sub.is_dir():
                color_root = sub
                break
    if color_root is None:
        raise RuntimeError("Could not locate the 'color' folder in the Kaggle download.")

    print(f"[info] copying tomato classes from {color_root}")
    copied = 0
    for cls in CLASS_NAMES:
        src = color_root / cls
        if not src.exists():
            print(f"[warn] missing source class: {cls}")
            continue
        dst = RAW_DIR / cls
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        copied += len(list(dst.iterdir()))
        print(f"[ok] {cls}: {len(list(dst.iterdir()))} images")

    print(f"[done] total images copied: {copied}")
    print("Now run: python -m src.dataset")


if __name__ == "__main__":
    run()
