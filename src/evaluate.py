"""Evaluate the trained model on the controlled and field-style test splits."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.pipeline import CLASS_TO_IDX, get_datasets
from src.settings import CLASS_DISPLAY, CLASS_NAMES, MODELS_DIR, REPORTS_DIR

DISPLAY_LABELS = [CLASS_DISPLAY[c] for c in CLASS_NAMES]


def _load_model() -> tf.keras.Model:
    final = MODELS_DIR / "tomato_mobilenetv2_final.keras"
    fine = MODELS_DIR / "tomato_mobilenetv2_finetune.keras"
    head = MODELS_DIR / "tomato_mobilenetv2_head.keras"
    for p in (final, fine, head):
        if p.exists():
            print(f"[info] loading model: {p.name}")
            return tf.keras.models.load_model(str(p), safe_mode=False)
    raise FileNotFoundError("No trained model found in models/.")


def _gather(model: tf.keras.Model, ds: tf.data.Dataset) -> tuple[np.ndarray, np.ndarray]:
    y_true, y_pred = [], []
    for batch_x, batch_y in ds:
        probs = model.predict(batch_x, verbose=0)
        y_pred.extend(np.argmax(probs, axis=1))
        y_true.extend(batch_y.numpy())
    return np.array(y_true), np.array(y_pred)


def _save_confusion(y_true, y_pred, name: str, out: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=DISPLAY_LABELS,
        yticklabels=DISPLAY_LABELS,
    )
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def _metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": float((y_true == y_pred).mean()),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def main() -> None:
    model = _load_model()
    ds = get_datasets()

    report = {"field_kind": ds["field_kind"], "splits": {}}

    for split_name, key in (("controlled", "test_controlled"), ("field", "test_field")):
        if key not in ds:
            print(f"[skip] no dataset for {key}")
            continue
        y_true, y_pred = _gather(model, ds[key])
        m = _metrics(y_true, y_pred)
        report["splits"][split_name] = m
        print(f"[{split_name}] {m}")

        _save_confusion(
            y_true,
            y_pred,
            name=split_name,
            out=REPORTS_DIR / f"confusion_{split_name}.png",
        )

        txt = classification_report(
            y_true,
            y_pred,
            labels=list(range(len(CLASS_NAMES))),
            target_names=DISPLAY_LABELS,
            zero_division=0,
            digits=4,
        )
        (REPORTS_DIR / f"classification_report_{split_name}.txt").write_text(txt)

    (REPORTS_DIR / "evaluation.json").write_text(json.dumps(report, indent=2))
    print("[ok] wrote evaluation.json + confusion plots + per-class reports")


if __name__ == "__main__":
    main()
