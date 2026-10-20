"""EMIPredict AI - home page (Streamlit entry point)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import streamlit as st

from app_common import load_models, page_setup, read_json
from config import REPORTS_DIR

page_setup("Home", "💳")

st.title("💳 EMIPredict AI")
st.caption("Intelligent financial risk assessment for EMI eligibility and maximum affordable EMI")

st.markdown(
    "Instant, data-driven loan decisions. Two models work together: a **classifier** that rates an application "
    "*Eligible*, *High risk* or *Not eligible*, and a **regressor** that estimates the **maximum safe monthly EMI** "
    "the applicant can carry."
)

try:
    _, _, meta = load_models()
    ready = True
except Exception as e:
    ready, meta = False, {}
    st.warning(f"Models are not available yet: {e}")

quality = read_json(REPORTS_DIR / "data_quality.json")
c1, c2, c3, c4 = st.columns(4)
rows = quality.get("after", {}).get("rows")
c1.metric("Records analysed", f"{rows:,}" if rows else "-")
if ready and "clf" in meta:
    c2.metric("Eligibility accuracy", f"{meta['clf'].get('test_accuracy', 0) * 100:.2f}%",
              help=f"Champion: {meta['clf']['best_algorithm']} (held-out test set)")
    c3.metric("Max-EMI RMSE", f"₹{meta['reg'].get('test_rmse', 0):,.0f}",
              help=f"Champion: {meta['reg']['best_algorithm']} (held-out test set)")
    c4.metric("Max-EMI R²", f"{meta['reg'].get('test_r2', 0):.3f}")

st.divider()
left, right = st.columns(2)
with left:
    st.subheader("What you can do")
    st.markdown(
        "- **🎯 Prediction** - assess a single applicant in real time with a lending recommendation\n"
        "- **📊 Data Explorer** - slice the 400K-record portfolio by scenario, income, credit and demographics\n"
        "- **🧪 Model Performance** - compare every trained model and browse the MLflow experiments and registry\n"
        "- **🗂️ Data Management** - create, view, edit and delete applicant records; batch-score a CSV"
    )
with right:
    st.subheader("How it works")
    st.markdown(
        "1. Raw applicant data → cleaned and validated\n"
        "2. ~20 engineered features: debt-to-income, expense ratio, affordability, savings buffer, composite risk score\n"
        "3. Four candidate models per task, tracked in **MLflow**; champion chosen on a validation split\n"
        "4. Champions served from the **model registry** pipeline - the same code path used in training"
    )

st.info("Open **Prediction** in the sidebar to try it.", icon="👈")
st.caption("Decision support only - final lending decisions require human underwriting and regulatory checks.")
