"""
Builds train / val / test_controlled / test_field index files from the
PlantVillage tomato subset plus an optional supplementary field set.

If `data/field/<class>/...` exists it is used as the field-style test split.
Otherwise a stratified slice of PlantVillage is reserved and tagged as
`simulated_field` so the report can disclose this honestly.
"""
from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import List, Tuple

from src.settings import (
    CFG,
    CLASS_NAMES,
    FIELD_DIR,
    PREPARED_DIR,
    RAW_DIR,
)

VALID_EXT = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
SEED = 42


def _collect(root: Path) -> dict[str, List[Path]]:
    out: dict[str, List[Path]] = {c: [] for c in CLASS_NAMES}
    if not root.exists():
        return out
    for cls in CLASS_NAMES:
        cls_dir = root / cls
        if not cls_dir.exists():
            continue
        out[cls] = sorted(p for p in cls_dir.iterdir() if p.suffix in VALID_EXT)
    return out


def _stratified_split(
    items: List[Path], fractions: Tuple[float, ...]
) -> List[List[Path]]:
    rng = random.Random(SEED)
    pool = list(items)
    rng.shuffle(pool)
    n = len(pool)
    cuts = []
    running = 0.0
    for f in fractions[:-1]:
        running += f
        cuts.append(int(round(running * n)))
    out: List[List[Path]] = []
    prev = 0
    for c in cuts:
        out.append(pool[prev:c])
        prev = c
    out.append(pool[prev:])
    return out


def _write_csv(path: Path, rows: List[Tuple[str, str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "class", "split"])
        writer.writerows(rows)


def build_index() -> dict:
    raw = _collect(RAW_DIR)
    field = _collect(FIELD_DIR)
    has_real_field = any(len(v) > 0 for v in field.values())

    splits_cfg = CFG["splits"]
    if has_real_field:
        fractions = (
            splits_cfg["train"],
            splits_cfg["val"],
            splits_cfg["test_controlled"] + splits_cfg["test_field"],
        )
    else:
        fractions = (
            splits_cfg["train"],
            splits_cfg["val"],
            splits_cfg["test_controlled"],
            splits_cfg["test_field"],
        )

    rows: List[Tuple[str, str, str]] = []
    summary = {"train": 0, "val": 0, "test_controlled": 0, "test_field": 0}

    for cls, paths in raw.items():
        if not paths:
            print(f"[warn] no images for class {cls}")
            continue
        parts = _stratified_split(paths, fractions)
        if has_real_field:
            train, val, test_ctrl = parts
            for p in train:
                rows.append((str(p), cls, "train"))
            for p in val:
                rows.append((str(p), cls, "val"))
            for p in test_ctrl:
                rows.append((str(p), cls, "test_controlled"))
            summary["train"] += len(train)
            summary["val"] += len(val)
            summary["test_controlled"] += len(test_ctrl)
        else:
            train, val, test_ctrl, test_field = parts
            for p in train:
                rows.append((str(p), cls, "train"))
            for p in val:
                rows.append((str(p), cls, "val"))
            for p in test_ctrl:
                rows.append((str(p), cls, "test_controlled"))
            for p in test_field:
                rows.append((str(p), cls, "test_field_simulated"))
            summary["train"] += len(train)
            summary["val"] += len(val)
            summary["test_controlled"] += len(test_ctrl)
            summary["test_field"] += len(test_field)

    if has_real_field:
        for cls, paths in field.items():
            for p in paths:
                rows.append((str(p), cls, "test_field"))
            summary["test_field"] += len(paths)

    index_path = PREPARED_DIR / "index.csv"
    _write_csv(index_path, rows)
    print(f"[ok] wrote {index_path}")
    print(f"[ok] split totals: {summary}")
    return {"index": index_path, "summary": summary, "real_field": has_real_field}


if __name__ == "__main__":
    build_index()
