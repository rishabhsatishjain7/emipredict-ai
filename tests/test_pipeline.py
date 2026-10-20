import json

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

import crud
from config import CLASS_LABELS, MODELS_DIR, PROCESSED_DIR, RAW_FEATURES
from features import ALL_FEATURES, amortised_emi, engineer_features
from preprocess import clean_data, split_data

SAMPLE = PROCESSED_DIR / "sample_30k.parquet"


@pytest.fixture(scope="module")
def sample():
    if not SAMPLE.exists():
        pytest.skip("sample data not generated")
    return pd.read_parquet(SAMPLE).head(2000)


def test_amortised_emi_known_value():
    # 100,000 at 12% p.a. over 12 months -> 8,884.88 (standard EMI formula)
    assert amortised_emi(100_000, 0.12, 12) == pytest.approx(8884.88, abs=0.01)


def test_engineer_features_complete_and_finite(sample):
    out = engineer_features(sample)
    assert set(ALL_FEATURES) <= set(out.columns)
    assert not np.isinf(out[ALL_FEATURES].select_dtypes("number").to_numpy()).any()
    assert (out["total_expenses"] >= 0).all()


def test_clean_data_fixes_dirty_values():
    raw = pd.read_parquet(SAMPLE).head(200).copy() if SAMPLE.exists() else pytest.skip("no sample")
    raw.loc[0, "gender"] = " male "
    raw.loc[1, "age"] = -1
    raw = pd.concat([raw, raw.iloc[[5]]], ignore_index=True)
    clean, rep = clean_data(raw)
    assert clean["gender"].iloc[0] == "Male"
    assert pd.isna(clean["age"].iloc[1])
    assert rep["duplicates_removed"] >= 1


def test_split_is_disjoint_and_stratified(sample):
    tr, va, te = split_data(sample)
    assert len(tr) + len(va) + len(te) == len(sample)
    assert abs(tr["emi_eligibility"].eq("Eligible").mean() - te["emi_eligibility"].eq("Eligible").mean()) < 0.05


def test_crud_roundtrip(tmp_path, monkeypatch, sample):
    monkeypatch.setattr(crud, "DB_PATH", tmp_path / "t.db")
    crud.init_db()
    assert crud.count() == 0
    crud.create_many(sample[RAW_FEATURES].head(5))
    assert crud.count() == 5
    first = int(crud.read(limit=1).iloc[0]["id"])
    crud.update(first, {"credit_score": 777})
    assert crud.read(limit=10).query("id == @first").iloc[0]["credit_score"] == 777
    assert crud.delete([first]) == 1
    assert crud.count() == 4


@pytest.mark.skipif(not (MODELS_DIR / "model_manifest.json").exists(), reason="models not trained")
def test_models_predict_valid_outputs(sample):
    from app_common import load_pipeline
    clf = load_pipeline("clf")
    reg = load_pipeline("reg")
    X = sample[RAW_FEATURES].head(50)
    proba = clf.predict_proba(X)
    assert proba.shape == (50, 3) and np.allclose(proba.sum(axis=1), 1, atol=1e-5)
    assert (reg.predict(X) > 0).all()


def test_xgboost_roundtrip_through_loader(tmp_path, sample):
    """A tiny XGBoost champion saved by train.save_champion reloads through the
    app loader with predictions identical to the original pipeline."""
    from app_common import load_pipeline
    from train import make_pre, save_champion

    df = sample.groupby("emi_eligibility", group_keys=False).head(60)
    X = df[RAW_FEATURES]
    y = df["emi_eligibility"].map({c: i for i, c in enumerate(CLASS_LABELS)})
    pipe = Pipeline([
        ("prep", make_pre()),
        ("model", XGBClassifier(n_estimators=5, eval_metric="mlogloss", random_state=42)),
    ]).fit(X, y)

    save_champion("clf", pipe, "XGBoost", models_dir=tmp_path)
    manifest = json.loads((tmp_path / "model_manifest.json").read_text(encoding="utf-8"))
    assert manifest["clf"]["format"] == "xgboost_native"
    assert manifest["clf"]["algorithm"] == "XGBoost"
    assert set(manifest["clf"]["versions"]) == {"python", "scikit-learn", "xgboost"}

    reloaded = load_pipeline("clf", models_dir=tmp_path)
    assert np.array_equal(pipe.predict(X), reloaded.predict(X))
    assert np.array_equal(pipe.predict_proba(X), reloaded.predict_proba(X))
