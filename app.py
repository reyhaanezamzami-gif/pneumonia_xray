import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path
import tempfile

from src.config import CNN_MODEL_PATH, IMAGE_SIZE
from src.gradcam import make_gradcam_heatmap, save_gradcam


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Chest X-ray Pneumonia Classifier",
    page_icon="ðŸ©»",
    layout="wide",
)

# --------------------------------------------------
# Load model
# --------------------------------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(str(CNN_MODEL_PATH))


# --------------------------------------------------
# Image preprocessing
# --------------------------------------------------

def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))

    image_array = np.array(image, dtype=np.float32)

    return image_array


# --------------------------------------------------
# Main interface
# --------------------------------------------------

st.title("Chest X-ray Pneumonia Classification")

st.markdown(
    """
    Upload a chest X-ray image to obtain a prediction
    from the trained MobileNetV2 model.
    """
)

st.warning(
    "Educational and research use only. "
    "This model is not a medical diagnostic system. "
    "Do not use its predictions for clinical decisions."
)

uploaded_file = st.file_uploader(
    "Upload a chest X-ray image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file).convert("RGB")

        image_array = preprocess_image(image)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Uploaded X-ray")
            st.image(image, use_column_width=True)

        with st.spinner("Loading model and analyzing image..."):

            model = load_model()

            input_tensor = np.expand_dims(
                image_array,
                axis=0,
            )

            prediction = model.predict(
                input_tensor,
                verbose=0,
            )[0][0]

            pneumonia_probability = float(prediction)
            normal_probability = 1.0 - pneumonia_probability

            predicted_class = (
                "PNEUMONIA"
                if pneumonia_probability >= 0.5
                else "NORMAL"
            )

        with col2:
            st.subheader("Prediction")

            if predicted_class == "PNEUMONIA":
                st.error("Prediction: PNEUMONIA")
            else:
                st.success("Prediction: NORMAL")

            st.metric(
                "Pneumonia probability",
                f"{pneumonia_probability * 100:.2f}%",
            )

            st.metric(
                "Normal probability",
                f"{normal_probability * 100:.2f}%",
            )

        st.divider()

        st.subheader("Grad-CAM Visualization")

        with st.spinner("Generating Grad-CAM visualization..."):

            image_tensor = tf.convert_to_tensor(
                image_array,
                dtype=tf.float32,
            )

            heatmap, gradcam_probability = make_gradcam_heatmap(
                image_tensor,
                model,
            )

            with tempfile.TemporaryDirectory() as temp_dir:

                output_path = (
                    Path(temp_dir) / "gradcam_result.png"
                )

                save_gradcam(
                    image_tensor,
                    heatmap,
                    gradcam_probability,
                    output_path,
                )

                st.image(
                    str(output_path),
                    caption="Original X-ray and Grad-CAM heatmap",
                    use_column_width=True,
                )

        st.caption(
            "Grad-CAM highlights image regions that contribute "
            "to the model's prediction. These regions are not "
            "proof of disease."
        )

    except Exception as error:
        st.error(f"An error occurred: {error}")

else:
    st.info("Upload an X-ray image to begin.")

# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "MobileNetV2 | Chest X-ray Pneumonia Classification | "
    "Research Project"
)
