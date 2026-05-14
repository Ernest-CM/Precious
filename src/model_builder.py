"""MobileNetV2 transfer-learning model for 10-class tomato leaf classification."""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

from src.settings import CFG, IMG_SIZE, NUM_CLASSES


def build_model(num_classes: int = NUM_CLASSES, trainable_backbone: bool = False) -> tf.keras.Model:
    train_cfg = CFG["training"]

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="leaf_image")
    x = layers.Lambda(preprocess_input, name="mobilenet_preprocess")(inputs)

    backbone = MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_tensor=x,
        pooling=None,
    )
    backbone.trainable = trainable_backbone

    feat = layers.GlobalAveragePooling2D(name="gap")(backbone.output)
    feat = layers.BatchNormalization(name="head_bn")(feat)
    feat = layers.Dense(
        int(train_cfg["dense_units"]),
        activation="relu",
        name="head_dense",
    )(feat)
    feat = layers.Dropout(float(train_cfg["dropout"]), name="head_dropout")(feat)
    outputs = layers.Dense(num_classes, activation="softmax", name="probs")(feat)

    model = models.Model(inputs=inputs, outputs=outputs, name="tomato_mobilenetv2")
    return model


def unfreeze_for_finetune(model: tf.keras.Model, from_layer_index: int | None = None) -> None:
    """Unfreeze layers from `start` onward, but keep BatchNorm frozen.

    Keeping the pretrained BN statistics is the standard recipe for fine-tuning
    MobileNetV2: unfreezing BN with a small batch size destabilises training.
    """
    train_cfg = CFG["training"]
    start = (
        int(train_cfg["finetune_from_layer"])
        if from_layer_index is None
        else int(from_layer_index)
    )
    for i, layer in enumerate(model.layers):
        if i < start:
            layer.trainable = False
            continue
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True
