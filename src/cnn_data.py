
"""Data loading pipeline for MobileNetV2 training."""

import pandas as pd
import tensorflow as tf

from src.config import IMAGE_SIZE, IMAGE_CHANNELS
from src.utils import get_logger

LOGGER = get_logger(__name__)

BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

# Mild augmentation suitable for chest X-ray images.
DATA_AUGMENTATION = tf.keras.Sequential(
    [
        tf.keras.layers.RandomRotation(
            factor=0.014,
            fill_mode="nearest",
        ),
        tf.keras.layers.RandomTranslation(
            height_factor=0.05,
            width_factor=0.05,
            fill_mode="nearest",
        ),
        tf.keras.layers.RandomZoom(
            height_factor=0.08,
            width_factor=0.08,
            fill_mode="nearest",
        ),
        tf.keras.layers.RandomContrast(factor=0.10),
    ],
    name="data_augmentation",
)


def load_image(
    image_path: tf.Tensor,
    label: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Load and resize an image for MobileNetV2."""

    image = tf.io.read_file(image_path)

    image = tf.io.decode_image(
        image,
        channels=IMAGE_CHANNELS,
        expand_animations=False,
    )

    image.set_shape([None, None, IMAGE_CHANNELS])

    image = tf.image.resize(
        image,
        [IMAGE_SIZE, IMAGE_SIZE],
    )

    image = tf.cast(image, tf.float32)

    return image, tf.cast(label, tf.float32)


def create_dataset(
    csv_path: str,
    batch_size: int = BATCH_SIZE,
    shuffle: bool = False,
    augment: bool = False,
) -> tf.data.Dataset:
    """Create a batched TensorFlow dataset from a split CSV."""

    frame = pd.read_csv(csv_path)

    paths = frame["image_path"].astype(str).to_numpy()
    labels = frame["label"].to_numpy(dtype="float32")

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=len(frame),
            seed=42,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=AUTOTUNE,
    )

    dataset = dataset.batch(batch_size)

    if augment:
        dataset = dataset.map(
            lambda images, labels: (
                tf.clip_by_value(DATA_AUGMENTATION(images, training=True), 0.0, 255.0),
                labels,
            ),
            num_parallel_calls=AUTOTUNE,
        )

    dataset = dataset.prefetch(AUTOTUNE)

    LOGGER.info(
        "Created dataset from %s with %d images (augment=%s)",
        csv_path,
        len(frame),
        augment,
    )

    return dataset

