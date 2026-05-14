"""Two-stage training: (1) train classifier head only, (2) fine-tune backbone."""
from __future__ import annotations

import json
import time
from datetime import datetime

import tensorflow as tf

from src.model_builder import build_model, unfreeze_for_finetune
from src.pipeline import compute_class_weights, get_datasets
from src.settings import CFG, MODELS_DIR, REPORTS_DIR


def _compile(model: tf.keras.Model, lr: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def _callbacks(stage: str) -> list:
    ckpt_path = MODELS_DIR / f"tomato_mobilenetv2_{stage}.keras"
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_path),
            save_best_only=True,
            monitor="val_accuracy",
            mode="max",
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=int(CFG["training"]["early_stopping_patience"]),
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-7,
        ),
        tf.keras.callbacks.CSVLogger(
            filename=str(REPORTS_DIR / f"history_{stage}.csv"),
        ),
    ]


def main() -> None:
    started = time.time()
    train_cfg = CFG["training"]
    ds = get_datasets()
    weights = compute_class_weights()
    print(f"[info] split sizes: {ds['counts']}")
    print(f"[info] field-style source: {ds['field_kind']}")

    model = build_model(trainable_backbone=False)
    _compile(model, float(train_cfg["learning_rate_head"]))
    model.summary()

    print("[stage 1] training classification head...")
    h1 = model.fit(
        ds["train"],
        validation_data=ds["val"],
        epochs=int(train_cfg["epochs_head"]),
        class_weight=weights,
        callbacks=_callbacks("head"),
    )

    print("[stage 2] fine-tuning backbone...")
    unfreeze_for_finetune(model)
    _compile(model, float(train_cfg["learning_rate_finetune"]))
    h2 = model.fit(
        ds["train"],
        validation_data=ds["val"],
        epochs=int(train_cfg["epochs_finetune"]),
        class_weight=weights,
        callbacks=_callbacks("finetune"),
    )

    head_best = max(h1.history.get("val_accuracy", [0.0]))
    finetune_best = max(h2.history.get("val_accuracy", [0.0]))
    head_path = MODELS_DIR / "tomato_mobilenetv2_head.keras"
    finetune_path = MODELS_DIR / "tomato_mobilenetv2_finetune.keras"

    if finetune_best >= head_best and finetune_path.exists():
        print(f"[ok] fine-tune improved val_acc {head_best:.4f} -> {finetune_best:.4f}")
        chosen = tf.keras.models.load_model(str(finetune_path), safe_mode=False)
        chosen_stage = "finetune"
    else:
        print(
            f"[warn] fine-tune did not improve (head {head_best:.4f} vs finetune {finetune_best:.4f}); "
            "using the stage-1 head checkpoint as the final model"
        )
        chosen = tf.keras.models.load_model(str(head_path), safe_mode=False)
        chosen_stage = "head"

    final_path = MODELS_DIR / "tomato_mobilenetv2_final.keras"
    chosen.save(str(final_path))
    print(f"[ok] saved final model ({chosen_stage}) to {final_path}")
    model = chosen

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite = converter.convert()
    tflite_path = MODELS_DIR / "tomato_mobilenetv2.tflite"
    tflite_path.write_bytes(tflite)
    print(f"[ok] saved TFLite model to {tflite_path} ({tflite_path.stat().st_size / 1024:.1f} KB)")

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "duration_seconds": round(time.time() - started, 1),
        "counts": ds["counts"],
        "field_kind": ds["field_kind"],
        "final_val_accuracy": float(max(h2.history.get("val_accuracy", [0.0]) + h1.history.get("val_accuracy", [0.0]))),
    }
    (REPORTS_DIR / "training_summary.json").write_text(json.dumps(summary, indent=2))
    print("[ok] wrote training_summary.json")


if __name__ == "__main__":
    main()
