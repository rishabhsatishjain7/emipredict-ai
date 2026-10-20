# EMIPredict AI - Intelligent Financial Risk Assessment Platform

End-to-end FinTech ML project: two prediction problems on 400,000 loan-applicant records, tracked in MLflow and served through a multi-page Streamlit app.

**Live app:** <paste URL after deployment>

| Problem | Target | Champion | Test result |
|---|---|---|---|
| Classification | `emi_eligibility` (Eligible / High_Risk / Not_Eligible) | XGBoost | 98.3% accuracy, weighted F1 0.983, ROC-AUC 0.992 |
| Regression | `max_monthly_emi` (₹500-50,000) | XGBoost | RMSE ₹318, MAE ₹121, R² 0.997 |

> **Important - data caveat.** The official `EMI_dataset` was not available when this was built, so `src/generate_data.py` creates a **synthetic** 400K-row dataset that follows the schema and domain rules in the brief. Labels come from affordability rules plus ~1.5% noise, so the scores above are optimistic and say nothing about real-world performance. To use the real dataset, place it at `data/emi_dataset.csv` (same column names) and rerun steps 2-4 below.

## Quick start

```bash
pip install -r requirements.txt

python src/generate_data.py     # 1. synthetic data (skip if you have the official CSV)
python src/preprocess.py        # 2. clean, validate, split 70/15/15 -> data/processed/
python src/eda.py               #    EDA figures + reports/eda_report.md
python src/train.py             # 3. train 4 classifiers + 4 regressors, log to MLflow, register champions (~10 min on 2 CPUs)
python src/make_report.py       #    reports/model_report.md
streamlit run streamlit_app.py  # 4. launch the app
mlflow ui --backend-store-uri sqlite:///mlflow.db   # optional: MLflow dashboard
pytest                          # tests
```

`python src/train.py --quick` runs a 60K-row smoke test in ~1.5 minutes.

## Project layout

```
streamlit_app.py          Home page (entry point)
pages/                    1 Prediction · 2 Data Explorer · 3 Model Performance (MLflow) · 4 Data Management (CRUD)
src/
  config.py               paths, column lists, scenario limits/rates
  generate_data.py        synthetic 400K dataset (5 scenarios x 80K)
  preprocess.py           cleaning, quality report, stratified splits
  features.py             feature engineering + sklearn preprocessor
  eda.py                  EDA figures and report
  train.py                training, MLflow tracking, registry, champion export
  crud.py                 SQLite CRUD layer
  app_common.py           cached loaders and inference helpers
models/                   portable champion models (.ubj + prep .joblib, described by model_manifest.json), comparison tables, metadata
reports/                  data quality, EDA, model report, figures
tests/                    pytest suite
docs/                     methodology, deployment, business impact
```

## What was built (mapped to the brief)

- **Preprocessing:** whitespace/case normalisation, impossible values → NaN, duplicate removal (400 found), missing-value handling via train-fit imputers (no leakage), quality report before/after, stratified 70/15/15 split.
- **Feature engineering (20 derived + 2 ordinal):** expense-to-income, debt-to-income, disposable income, amortised requested EMI (credit-adjusted rate), affordability ratio, post-loan DTI, savings buffer, composite risk score, interaction terms. One-hot + scaling inside the pipeline. Feature engineering is part of the saved pipeline, so the app takes raw inputs.
- **Models:** classification - Logistic Regression, Decision Tree, Random Forest, XGBoost. Regression - Linear Regression, Decision Tree, Random Forest, XGBoost.
- **Metrics:** accuracy/precision/recall/F1/ROC-AUC (classification); RMSE/MAE/R²/MAPE (regression); train/val/test for each.
- **MLflow:** two experiments, 8 runs with params, metrics, confusion matrices / residual plots, feature importances, data-quality artifact, signed models; both registered models carry 4 versions each, with a `champion` alias on the winner chosen on the **validation** split.
- **Streamlit app:** real-time prediction with recommendation and "what would be affordable" guidance, interactive EDA with filters, model/MLflow dashboard, full CRUD plus CSV batch scoring.
- **Deployment:** see `docs/DEPLOYMENT.md`.

## Known limitations

- The brief lists "22 input features" but names 25 input columns; all 25 are used.
- No hyperparameter search was run; models use sensible fixed settings with early stopping for XGBoost.
- CRUD data lives in SQLite; on Streamlit Community Cloud the filesystem is ephemeral, so edits reset on restart.
- Rate premiums in the app's recommendations are illustrative assumptions, not a lending policy.
- The originally shipped `models/*.joblib` (pickled on Linux) failed to load on Windows with `XGBoostError: input stream corrupted` even under the pinned versions — pickling an XGBoost estimator inside a pipeline is not portable across platforms. Champions are now exported portably: the XGBoost booster in its native `.ubj` format plus a joblib-only prep pipeline, recorded in `models/model_manifest.json`; `src/app_common.py` rebuilds the serving Pipeline from it. After changing platform or library versions, regenerate the files with `python src/train.py`.
