from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# ── Paths ────────────────────────────────────────────────────────────────────
# Inside the HF Space Docker container, app.py and models/ sit together at
# WORKDIR (/app), so we stay at APP_DIR — not APP_DIR.parent.
APP_DIR = Path(__file__).resolve().parent
LOCAL_MODEL_PATH = APP_DIR / "models" / "best_model.joblib"
LOCAL_METADATA_PATH = APP_DIR / "models" / "model_metadata.json"

FEATURE_COLUMNS = [
    "engine_rpm",
    "lub_oil_pressure",
    "fuel_pressure",
    "coolant_pressure",
    "lub_oil_temp",
    "coolant_temp",
]
FEATURE_LABELS = {
    "engine_rpm": "Engine RPM",
    "lub_oil_pressure": "Lub Oil Pressure",
    "fuel_pressure": "Fuel Pressure",
    "coolant_pressure": "Coolant Pressure",
    "lub_oil_temp": "Lub Oil Temperature",
    "coolant_temp": "Coolant Temperature",
}


# ── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    hf_model_repo = os.getenv("HF_MODEL_REPO")
    try:
        if hf_model_repo:
            from huggingface_hub import hf_hub_download
            model_path = hf_hub_download(
                repo_id=hf_model_repo, filename="best_model.joblib"
            )
            return joblib.load(model_path)
        if LOCAL_MODEL_PATH.exists():
            return joblib.load(LOCAL_MODEL_PATH)
        return None
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None


def load_metadata() -> dict:
    try:
        if LOCAL_METADATA_PATH.exists():
            with LOCAL_METADATA_PATH.open("r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Engine Predictive Maintenance",
    page_icon=":wrench:",
    layout="centered",
)
st.title("Engine Predictive Maintenance")
st.caption(
    "Classifies whether an engine is likely to need maintenance "
    "based on sensor readings."
)

# ── Load model ────────────────────────────────────────────────────────────────
model = load_model()
metadata = load_metadata()

if model is None:
    st.error(
        "Model not found. The pipeline may still be running or the model "
        "has not been pushed to this Space yet. Please check back shortly."
    )
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Model Info")
    st.write(metadata.get("best_model", "Trained classifier"))
    metrics = metadata.get("test_metrics", {})
    if metrics:
        st.metric("Test F1", f"{metrics.get('f1', 0):.3f}")
        st.metric("Test Recall", f"{metrics.get('recall', 0):.3f}")

# ── Inputs ────────────────────────────────────────────────────────────────────
inputs = {
    "engine_rpm":       st.slider("Engine RPM", 50, 2300, 790, 10),
    "lub_oil_pressure": st.slider("Lub Oil Pressure", 0.0, 8.0, 3.3, 0.1),
    "fuel_pressure":    st.slider("Fuel Pressure", 0.0, 22.0, 6.7, 0.1),
    "coolant_pressure": st.slider("Coolant Pressure", 0.0, 8.0, 2.3, 0.1),
    "lub_oil_temp":     st.slider("Lub Oil Temperature", 70.0, 92.0, 77.6, 0.1),
    "coolant_temp":     st.slider("Coolant Temperature", 60.0, 200.0, 78.4, 0.1),
}

record = pd.DataFrame([inputs], columns=FEATURE_COLUMNS)

# ── Prediction ────────────────────────────────────────────────────────────────
prediction = int(model.predict(record)[0])
probability = None
if hasattr(model, "predict_proba"):
    probability = float(model.predict_proba(record)[0, 1])

st.subheader("Prediction")
if prediction == 1:
    st.error("⚠️ Maintenance recommended")
else:
    st.success("✅ Engine appears normal")

if probability is not None:
    st.progress(probability)
    st.write(f"Estimated maintenance probability: **{probability:.2%}**")

# ── Input record ──────────────────────────────────────────────────────────────
st.subheader("Input Record")
display_record = record.rename(columns=FEATURE_LABELS)
st.dataframe(display_record, use_container_width=True, hide_index=True)
