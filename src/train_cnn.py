"""Train the MobileNetV2 pneumonia classifier."""

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.utils.class_weight import compute_class_weight

from src.config import (
    CNN_MODEL_PATH,
    CNN_HISTORY_PATH,
    SEED,
)
from src.cnn_data import create_dataset
from src.cnn_model import build_mobilenetv2_model
from src.utils import get_logger, save_json, set_seed

LOGGER = get_logger(__name__)

EPOCHS = 15
BATCH_SIZE = 32


def main():
    set_seed(SEED)

    train_csv = "data/splits/train.csv"
    validation_csv = "data/splits/validation.csv"

    train_frame = pd.read_csv(train_csv)
    labels = train_frame["label"].to_numpy()

    classes = np.array([0, 1])
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=labels,
    )
    class_weights = {
        int(class_id): float(weight)
        for class_id, weight in zip(classes, weights)
    }

    LOGGER.info("Class weights: %s", class_weights)

    train_dataset = create_dataset(
        train_csv,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    validation_dataset = create_dataset(
        validation_csv,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = build_mobilenetv2_model()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(CNN_MODEL_PATH),
            monitor="val_recall",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_recall",
            mode="max",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    save_json(
        {
            key: [float(value) for value in values]
            for key, values in history.history.items()
        },
        CNN_HISTORY_PATH,
    )

    LOGGER.info("Training completed.")
    LOGGER.info("Best model saved to %s", CNN_MODEL_PATH)


if __name__ == "__main__":
    main()