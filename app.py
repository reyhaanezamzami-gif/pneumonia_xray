import streamlit as st
import tensorflow as tf
import numpy as np
import tempfile
import os
import hashlib
from pathlib import Path
from PIL import Image

from src.config import CNN_NIH_MODEL_PATH, GRADCAM_DIR
from src.gradcam import generate_gradcam

st.set_page_config(
    page_title="AURA | X-ray Analysis",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="collapsed"
)

MODEL_PATH = CNN_NIH_MODEL_PATH
THRESHOLD = 0.5

# -------------------- STYLE --------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --navy: #173b55;
    --blue: #2878a8;
    --muted: #7b8d9c;
    --border: #e5edf2;
    --canvas: #f7fafc;
    --green: #16866b;
    --red: #c65361;
}

.stApp {
    background: var(--canvas);
    color: var(--navy);
    font-family: Inter, "Segoe UI", sans-serif;
}

.block-container {
    max-width: 1220px;
    padding: 2rem 2rem 2.5rem;
}

header[data-testid="stHeader"] {
    background: transparent;
}

#MainMenu, footer {
    visibility: hidden;
}

h1, h2, h3, p {
    font-family: Inter, "Segoe UI", sans-serif;
}

.brand {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 30px;
}

.brand-icon {
    width: 43px;
    height: 43px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 13px;
    background: #e8f2f8;
    font-size: 23px;
}

.brand-name {
    color: var(--navy);
    font-size: 21px;
    font-weight: 800;
    letter-spacing: 1.5px;
}

.brand-caption {
    color: var(--muted);
    font-size: 10px;
    letter-spacing: 1.4px;
    font-weight: 600;
}

.page-title {
    color: var(--navy);
    font-size: 30px;
    font-weight: 800;
    letter-spacing: -1px;
    margin-bottom: 5px;
}

.page-subtitle {
    color: var(--muted);
    font-size: 13px;
    margin-bottom: 25px;
}

.panel {
    background: white;
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 4px 16px rgba(26, 61, 85, .035);
}

.panel-heading {
    font-size: 14px;
    font-weight: 700;
    color: var(--navy);
    margin-bottom: 5px;
}

.panel-caption {
    color: var(--muted);
    font-size: 11px;
    margin-bottom: 15px;
}

.result-card {
    background: white;
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 24px;
    min-height: 150px;
    box-shadow: 0 4px 16px rgba(26, 61, 85, .035);
}

.result-caption {
    color: var(--muted);
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.4px;
    margin-bottom: 12px;
}

.result-value {
    font-size: 29px;
    font-weight: 800;
    letter-spacing: -.5px;
}

.normal {
    color: var(--green);
}

.pneumonia {
    color: var(--red);
}

.result-note {
    color: var(--muted);
    font-size: 11px;
    line-height: 1.6;
    margin-top: 10px;
}

.section-title {
    color: var(--navy);
    font-size: 17px;
    font-weight: 750;
    margin: 30px 0 13px;
}

.score-card {
    background: white;
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 17px;
}

.score-label {
    color: var(--muted);
    font-size: 11px;
    margin-bottom: 7px;
}

.score-value {
    color: var(--navy);
    font-size: 24px;
    font-weight: 800;
}

.stButton > button {
    background: var(--blue);
    color: white !important;
    border: 0;
    border-radius: 10px;
    min-height: 44px;
    font-weight: 700;
}

.stButton > button:hover {
    background: #1d628d;
    color: white !important;
}

[data-testid="stFileUploader"] {
    background: #f8fbfd;
    border: 1px dashed #b8cbd8;
    border-radius: 12px;
    padding: 12px;
}

[data-testid="stFileUploader"] button {
    background: #eaf3f8 !important;
    color: var(--navy) !important;
    border: 1px solid #d6e4ec !important;
}

[data-testid="stImage"] img {
    border-radius: 12px;
}

[data-testid="stProgress"] > div {
    background: #e8eef2;
    border-radius: 20px;
}

[data-testid="stProgress"] > div > div {
    background: var(--blue);
    border-radius: 20px;
}

[data-testid="stAlert"] {
    border-radius: 11px;
    font-size: 12px;
}

div[data-testid="stExpander"] {
    background: white;
    border: 1px solid var(--border);
    border-radius: 12px;
}

.small-note {
    color: var(--muted);
    font-size: 10px;
    line-height: 1.6;
}

.footer {
    border-top: 1px solid var(--border);
    margin-top: 38px;
    padding-top: 17px;
    text-align: center;
    color: #93a2ad;
    font-size: 10px;
    letter-spacing: .5px;
}

@media (max-width: 700px) {
    .block-container {
        padding: 1.2rem 1rem 2rem;
    }
    .page-title {
        font-size: 25px;
    }
}
</style>
""", unsafe_allow_html=True)


# -------------------- MODEL --------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(str(MODEL_PATH))


def predict_image(model, image):
    resized = image.resize((224, 224))
    array = np.array(resized).astype(np.float32)
    array = np.expand_dims(array, axis=0)
    return float(model.predict(array, verbose=0)[0][0])


def clear_old_result():
    for key in [
        "prediction",
        "pneumonia_probability",
        "normal_probability",
        "gradcam_path",
        "image_hash"
    ]:
        st.session_state.pop(key, None)


# -------------------- HEADER --------------------

st.markdown("""
<div class="brand">
    <div class="brand-icon">🩻</div>
    <div>
        <div class="brand-name">AURA</div>
        <div class="brand-caption">CHEST X-RAY AI</div>
    </div>
</div>

<div class="page-title">Chest X-ray Analysis</div>
<div class="page-subtitle">AI image classification and visual interpretation</div>
""", unsafe_allow_html=True)

if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()


# -------------------- IMAGE UPLOAD --------------------

upload_col, preview_col = st.columns([0.85, 1.15], gap="large")

with upload_col:
    st.markdown("""
    <div class="panel">
        <div class="panel-heading">Upload image</div>
        <div class="panel-caption">Select a chest X-ray image</div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

with preview_col:
    st.markdown("""
    <div class="panel-heading" style="margin-bottom:12px;">
        Image preview
    </div>
    """, unsafe_allow_html=True)

    if uploaded_file is not None:
        image_bytes = uploaded_file.getvalue()
        current_hash = hashlib.md5(image_bytes).hexdigest()

        if st.session_state.get("image_hash") != current_hash:
            clear_old_result()
            st.session_state["image_hash"] = current_hash

        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_column_width=True)
    else:
        st.markdown("""
        <div style="
            height:230px;
            display:flex;
            flex-direction:column;
            align-items:center;
            justify-content:center;
            background:#ffffff;
            border:1px dashed #cbd9e2;
            border-radius:14px;
            color:#8b9ba8;">
            <div style="font-size:38px;">🩻</div>
            <div style="font-size:13px;font-weight:600;margin-top:10px;">
                No image selected
            </div>
        </div>
        """, unsafe_allow_html=True)


# -------------------- ANALYSIS --------------------

if uploaded_file is not None:
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    if st.button("Analyze X-ray", type="primary", use_container_width=True):
        with st.spinner("Analyzing image..."):
            try:
                model = load_model()

                pneumonia_probability = predict_image(model, image)
                normal_probability = 1.0 - pneumonia_probability

                prediction = (
                    "PNEUMONIA"
                    if pneumonia_probability >= THRESHOLD
                    else "NORMAL"
                )

                st.session_state["prediction"] = prediction
                st.session_state["pneumonia_probability"] = pneumonia_probability
                st.session_state["normal_probability"] = normal_probability

                GRADCAM_DIR.mkdir(parents=True, exist_ok=True)

                fd, temp_name = tempfile.mkstemp(
                    suffix=".jpg",
                    dir=str(GRADCAM_DIR)
                )
                os.close(fd)
                temp_path = Path(temp_name)

                try:
                    image.save(temp_path, format="JPEG")
                    gradcam_path = generate_gradcam(temp_path)
                    st.session_state["gradcam_path"] = str(gradcam_path)
                finally:
                    temp_path.unlink(missing_ok=True)

            except Exception as error:
                st.error(f"Analysis failed: {error}")


# -------------------- RESULTS --------------------

if "prediction" in st.session_state:
    prediction = st.session_state["prediction"]
    pneumonia_probability = st.session_state["pneumonia_probability"]
    normal_probability = st.session_state["normal_probability"]

    st.markdown('<div class="section-title">Analysis result</div>', unsafe_allow_html=True)

    result_col, scores_col = st.columns([1, 1.2], gap="large")

    with result_col:
        result_class = "pneumonia" if prediction == "PNEUMONIA" else "normal"

        st.markdown(f"""
        <div class="result-card">
            <div class="result-caption">MODEL CLASSIFICATION</div>
            <div class="result-value {result_class}">{prediction}</div>
            <div class="result-note">Decision threshold: 0.50</div>
        </div>
        """, unsafe_allow_html=True)

    with scores_col:
        score_a, score_b = st.columns(2)

        with score_a:
            st.markdown(f"""
            <div class="score-card">
                <div class="score-label">Pneumonia score</div>
                <div class="score-value">{pneumonia_probability * 100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        with score_b:
            st.markdown(f"""
            <div class="score-card">
                <div class="score-label">Normal score</div>
                <div class="score-value">{normal_probability * 100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        st.progress(pneumonia_probability)

    # -------------------- GRAD-CAM --------------------

    st.markdown('<div class="section-title">Grad-CAM</div>', unsafe_allow_html=True)

    gradcam_path = st.session_state.get("gradcam_path")

    if gradcam_path and Path(gradcam_path).exists():
        st.image(
            gradcam_path,
            caption="Model attention visualization",
            use_column_width=True
        )
    else:
        st.info("Grad-CAM visualization is unavailable.")

    with st.expander("About this result"):
        st.write(
            "Grad-CAM highlights image regions that influenced the model output. "
            "It does not prove the presence, location or cause of disease."
        )

# -------------------- MEDICAL NOTICE --------------------

st.markdown("""
<div class="footer">
    Research prototype · Educational use only<br>
    Not intended for clinical diagnosis or treatment decisions.
</div>
""", unsafe_allow_html=True)
