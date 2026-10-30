"""MobileNetV2 model architecture for pneumonia classification."""

import tensorflow as tf
from tensorflow.keras import layers, models

from src.config import IMAGE_SHAPE, MOBILENET_WEIGHTS_PATH


def build_mobilenetv2_model(
    input_shape: tuple[int, int, int] = IMAGE_SHAPE,
) -> tf.keras.Model:
    """Build a MobileNetV2 transfer learning model."""
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights=None,
    )

    base_model.load_weights(str(MOBILENET_WEIGHTS_PATH))
    base_model.trainable = False

    inputs = layers.Input(shape=input_shape)

    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.Precision(name="precision"),
        ],
    )

    return model
