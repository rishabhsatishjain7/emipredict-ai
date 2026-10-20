"""Shared helpers for the Streamlit app: path setup, cached model/data loaders, inference."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier, XGBRegressor

from config import (CLASS_LABELS, MODELS_DIR, PROCESSED_DIR, RAW_FEATURES, REPORTS_DIR, SCENARIO_RATE)

CLASS_COLORS = {"Eligible": "#2a9d8f", "High_Risk": "#e9a23b", "Not_Eligible": "#e76f51"}
CLASS_DISPLAY = {"Eligible": "Eligible", "High_Risk": "High risk", "Not_Eligible": "Not eligible"}


def page_setup(title: str, icon: str = "💳"):
    st.set_page_config(page_title=f"{title} · EMIPredict AI", page_icon=icon, layout="wide")
    st.markdown(
        """<style>
        .block-container {padding-top: 2rem; max-width: 1250px;}
        div[data-testid="stMetric"] {background: rgba(128,128,128,0.08); padding: 0.8rem 1rem; border-radius: 10px;}
        @media (max-width: 640px) {.block-container {padding-left: 0.8rem; padding-right: 0.8rem;}}
        </style>""",
        unsafe_allow_html=True,
    )


TRAIN_HINT = "Run `python src/train.py` to (re)train them."


def load_pipeline(task: str, models_dir: Path | None = None):
    """Rebuild the serving pipeline (prep + model) for `task` ("clf" / "reg") from the manifest.

    Reads models/model_manifest.json, which records for each task the format
    used by train.py:
      - "xgboost_native":  prep-only pipeline (.joblib) + native XGBoost
        booster (.ubj); the two are reassembled into one Pipeline here.
      - "joblib_pipeline": a single full-pipeline .joblib, returned as is.
    Raises with a message telling the user to run `python src/train.py`.
    """
    d = Path(models_dir) if models_dir is not None else MODELS_DIR
    manifest_p = d / "model_manifest.json"
    if not manifest_p.exists():
        raise FileNotFoundError(f"Model manifest not found ({manifest_p}). {TRAIN_HINT}")
    try:
        manifest = json.loads(manifest_p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise RuntimeError(f"Unreadable {manifest_p.name}: {e}. {TRAIN_HINT}") from e
    entry = manifest.get(task)
    if entry is None:
        raise FileNotFoundError(f"No '{task}' entry in {manifest_p.name}. {TRAIN_HINT}")
    files = entry.get("files", {})
    missing = [n for n in files.values() if not (d / n).exists()]
    if missing:
        raise FileNotFoundError(f"Missing model file(s): {', '.join(missing)}. {TRAIN_HINT}")
    fmt = entry.get("format")
    try:
        if fmt == "xgboost_native":
            prep = joblib.load(d / files["prep"])
            model = (XGBClassifier if task == "clf" else XGBRegressor)()
            model.load_model(str(d / files["model"]))
            return Pipeline([("prep", prep), ("model", model)])
        if fmt == "joblib_pipeline":
            return joblib.load(d / files["pipeline"])
    except Exception as e:  # corrupted files / version mismatch - no stack trace in the app
        raise RuntimeError(f"Could not load the '{task}' model: {e}. {TRAIN_HINT}") from e
    raise ValueError(f"Unknown model format {fmt!r} for '{task}'. {TRAIN_HINT}")


@st.cache_resource(show_spinner="Loading models…")
def load_models():
    """Returns (classifier_pipeline, regressor_pipeline, metadata) or raises with a friendly message."""
    clf, reg = load_pipeline("clf"), load_pipeline("reg")
    meta_p = MODELS_DIR / "metadata.json"
    meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
    return clf, reg, meta


def models_or_stop():
    try:
        return load_models()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()
    except Exception as e:  # corrupted / version-mismatched pickles
        st.error(f"Could not load models: {e}")
        st.stop()


@st.cache_data(show_spinner="Loading data…")
def load_sample() -> pd.DataFrame:
    for name in ("clean.parquet", "sample_30k.parquet"):
        p = PROCESSED_DIR / name
        if p.exists():
            df = pd.read_parquet(p)
            return df.sample(min(len(df), 60_000), random_state=1).reset_index(drop=True)
    st.error("No dataset found. Run `python src/generate_data.py` and `python src/preprocess.py`.")
    st.stop()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def predict_df(df: pd.DataFrame) -> pd.DataFrame:
    """Run both models on a frame containing the raw input columns."""
    clf, reg, _ = models_or_stop()
    X = df[RAW_FEATURES]
    proba = clf.predict_proba(X)
    out = pd.DataFrame({
        "pred_eligibility": [CLASS_LABELS[i] for i in proba.argmax(axis=1)],
        "pred_max_emi": np.clip(reg.predict(X), 500, 50_000).round(0),
        "confidence": proba.max(axis=1).round(4),
    }, index=df.index)
    for i, c in enumerate(CLASS_LABELS):
        out[f"p_{c}"] = proba[:, i]
    return out


def adjusted_rate(scenario: str, credit_score: float) -> float:
    return SCENARIO_RATE[scenario] + (700 - credit_score) / 700 * 0.04


def max_affordable_principal(emi: float, annual_rate: float, months: int) -> float:
    r = annual_rate / 12
    return float(emi * ((1 + r) ** months - 1) / (r * (1 + r) ** months))


def inr(x: float) -> str:
    return f"₹{x:,.0f}"
