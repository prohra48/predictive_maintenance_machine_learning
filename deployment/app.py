from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
LOCAL_MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"
LOCAL_METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"

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


@st.cache_resource
def load_model():
    hf_model_repo = os.getenv("HF_MODEL_REPO")
    if hf_model_repo:
        from huggingface_hub import hf_hub_download

        model_path = hf_hub_download(repo_id=hf_model_repo, filename="best_model.joblib")
        return joblib.load(model_path)

    return joblib.load(LOCAL_MODEL_PATH)


def load_metadata() -> dict:
    if LOCAL_METADATA_PATH.exists():
        with LOCAL_METADATA_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


st.set_page_config(page_title="Engine Predictive Maintenance", page_icon=":wrench:", layout="centered")
st.title("Engine Predictive Maintenance")
st.caption("Classifies whether an engine is likely to need maintenance based on sensor readings.")

model = load_model()
metadata = load_metadata()

with st.sidebar:
    st.header("Model")
    st.write(metadata.get("best_model", "Trained classifier"))
    metrics = metadata.get("test_metrics", {})
    if metrics:
        st.metric("Test F1", f"{metrics.get('f1', 0):.3f}")
        st.metric("Test Recall", f"{metrics.get('recall', 0):.3f}")

inputs = {
    "engine_rpm": st.slider("Engine RPM", 50, 2300, 790, 10),
    "lub_oil_pressure": st.slider("Lub Oil Pressure", 0.0, 8.0, 3.3, 0.1),
    "fuel_pressure": st.slider("Fuel Pressure", 0.0, 22.0, 6.7, 0.1),
    "coolant_pressure": st.slider("Coolant Pressure", 0.0, 8.0, 2.3, 0.1),
    "lub_oil_temp": st.slider("Lub Oil Temperature", 70.0, 92.0, 77.6, 0.1),
    "coolant_temp": st.slider("Coolant Temperature", 60.0, 200.0, 78.4, 0.1),
}

record = pd.DataFrame([inputs], columns=FEATURE_COLUMNS)
prediction = int(model.predict(record)[0])
probability = None
if hasattr(model, "predict_proba"):
    probability = float(model.predict_proba(record)[0, 1])

st.subheader("Prediction")
if prediction == 1:
    st.error("Maintenance recommended")
else:
    st.success("Engine appears normal")

if probability is not None:
    st.progress(probability)
    st.write(f"Estimated maintenance probability: {probability:.2%}")

st.subheader("Input Record")
display_record = record.rename(columns=FEATURE_LABELS)
st.dataframe(display_record, use_container_width=True, hide_index=True)
