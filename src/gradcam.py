"""Generate Grad-CAM visualizations for the trained MobileNetV2 model."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

from src.config import (
    CNN_NIH_MODEL_PATH,
    GRADCAM_DIR,
    IMAGE_SIZE,
    ensure_directories,
)
from src.utils import get_logger

LOGGER = get_logger(__name__)


def load_xray(image_path):
    """Load and resize an X-ray image."""
    image = tf.io.read_file(str(image_path))
    image = tf.io.decode_image(
        image, channels=3, expand_animations=False
    )
    image = tf.image.resize(
        image, [IMAGE_SIZE, IMAGE_SIZE]
    )
    return tf.cast(image, tf.float32)


def make_gradcam_heatmap(image, model):
    """Compute Grad-CAM for the pneumonia output."""

    base_model = model.get_layer("mobilenetv2_1.00_224")
    last_conv_layer = base_model.get_layer("out_relu")

    conv_model = tf.keras.Model(
        inputs=base_model.inputs,
        outputs=[
            last_conv_layer.output,
            base_model.output,
        ],
    )

    preprocessed_image = (
        tf.keras.applications.mobilenet_v2.preprocess_input(
            tf.expand_dims(image, axis=0)
        )
    )

    with tf.GradientTape() as tape:
        conv_outputs, base_outputs = conv_model(
            preprocessed_image, training=False
        )

        pooled = model.get_layer(
            "global_average_pooling2d"
        )(base_outputs)

        dropped = model.get_layer("dropout")(
            pooled, training=False
        )

        predictions = model.get_layer("pneumonia_probability")(dropped)
        loss = predictions[:, 0]

    gradients = tape.gradient(loss, conv_outputs)

    if gradients is None:
        raise RuntimeError("Grad-CAM gradients are None.")

    weights = tf.reduce_mean(
        gradients, axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(
        conv_outputs * weights, axis=-1
    )

    heatmap = tf.maximum(heatmap, 0)

    heatmap = tf.math.divide_no_nan(
        heatmap,
        tf.reduce_max(heatmap),
    )

    return heatmap.numpy(), float(predictions[0, 0].numpy())


def save_gradcam(image, heatmap, probability, output_path):
    """Save the original X-ray and Grad-CAM overlay."""

    image_array = image.numpy().astype("uint8")

    heatmap_resized = tf.image.resize(
        heatmap[..., tf.newaxis],
        [IMAGE_SIZE, IMAGE_SIZE],
    ).numpy().squeeze()

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(image_array)
    axes[0].set_title("Original X-ray")
    axes[0].axis("off")

    axes[1].imshow(image_array)
    axes[1].imshow(
        heatmap_resized,
        cmap="jet",
        alpha=0.4,
    )
    axes[1].set_title(
        f"Grad-CAM | Pneumonia probability: {probability:.3f}"
    )
    axes[1].axis("off")

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


def generate_gradcam(image_path):
    """Generate and save Grad-CAM for one image."""

    ensure_directories()

    image_path = Path(image_path)

    if not CNN_NIH_MODEL_PATH.exists():
        raise FileNotFoundError(
            f'Model not found: {CNN_NIH_MODEL_PATH}'
        )

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    model = tf.keras.models.load_model(
        str(CNN_NIH_MODEL_PATH)
    )

    image = load_xray(image_path)

    heatmap, probability = make_gradcam_heatmap(
        image, model
    )

    output_path = (
        GRADCAM_DIR / f"{image_path.stem}_gradcam.png"
    )

    save_gradcam(
        image,
        heatmap,
        probability,
        output_path,
    )

    predicted_class = (
        "PNEUMONIA" if probability >= 0.5 else "NORMAL"
    )

    LOGGER.info("Prediction: %s", predicted_class)
    LOGGER.info(
        "Pneumonia probability: %.4f", probability
    )
    LOGGER.info("Grad-CAM saved to: %s", output_path)

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a Grad-CAM visualization for an X-ray."
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to the X-ray image.",
    )

    args = parser.parse_args()
    generate_gradcam(args.image)


