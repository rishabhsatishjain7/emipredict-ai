# Methodology and Architecture

## Architecture

```
emi_dataset.csv (400K) → preprocess.py → data/processed/{clean,train,val,test}.parquet
        → features.py (engineer + impute + scale + one-hot)  ← part of every saved pipeline
        → train.py → MLflow (sqlite backend, 2 experiments, registry)
        → models/ (portable champion files + model_manifest.json) + comparison tables
        → Streamlit (streamlit_app.py + pages/) → Streamlit Community Cloud
```

| Layer | Implementation |
|---|---|
| Data | CSV → parquet; SQLite for admin CRUD |
| Processing | pandas, scikit-learn `Pipeline` / `ColumnTransformer` |
| Model | sklearn + XGBoost, tracked in MLflow 3 (SQLite store, file artifacts) |
| Application | Streamlit multi-page, Plotly charts |
| Deployment | GitHub → Streamlit Community Cloud; GitHub Actions runs tests |

## Data quality

`preprocess.assess_quality` reports missing values, duplicates, out-of-range values and categorical inconsistencies before and after cleaning (`reports/data_quality.json`). Cleaning is deterministic; **statistical imputation (median / most-frequent) happens inside the sklearn pipeline and is fit on training data only**, preventing leakage into validation/test.

## Feature engineering

| Group | Features |
|---|---|
| Ratios | `expense_to_income`, `debt_to_income`, `disposable_income`, `disposable_to_income`, `emi_to_income`, `affordability_ratio` (requested EMI ÷ disposable), `dti_after_loan`, `loan_to_annual_income` |
| Pricing | `scenario_rate` + credit adjustment: `rate + (700 − score)/700 × 4pts`; `est_requested_emi` via the standard amortisation formula |
| Liquidity | `savings_buffer_months`, `liquid_assets_to_loan` |
| Risk | `credit_risk_score` (0-1), `employment_stability` (years × type weight), `risk_score` (weighted composite: 35% credit, 25% post-loan DTI, 20% expense ratio, 10% stability, 10% buffer) |
| Interactions | `income_x_credit`, `emi_x_tenure_burden`, `income_per_member`, `dependents_ratio` |
| Encoding | one-hot for nominal categoricals; ordinal `education_level`, `company_tier` |

The weights in `risk_score` and the scenario interest rates are **business assumptions**, defined in `features.py` / `config.py`, to be replaced with the institution's own policy.

## Modelling and selection

- Splits: stratified 70/15/15 on `emi_eligibility` (`random_state=42`).
- XGBoost uses the validation split for early stopping (30 rounds).
- **Selection uses validation metrics** (weighted F1 for classification, RMSE for regression). Test metrics are logged for all models but were not used to choose, keeping the reported test score unbiased.
- Champions are exported portably to `models/` (native `.ubj` + prep-only `.joblib` for XGBoost, full `.joblib` otherwise — see `model_manifest.json`) and tagged `champion` in the MLflow registry.

## MLflow layout

- Experiments: `EMI_Eligibility_Classification`, `EMI_MaxEMI_Regression`.
- Per run: hyperparameters, train/val/test metrics, fit time, confusion matrix or residual plot, feature importance (PNG + CSV), `data_quality.json`, signed model with input example.
- Registry: `EMIPredict_Classifier`, `EMIPredict_Regressor`; each model variant is a version tagged with its algorithm; `champion` alias marks production.
- Models are logged with `serialization_format="cloudpickle"` because the pipeline contains a custom function (`features.engineer_features`); loading them requires `src/` on `PYTHONPATH`.

## Synthetic data design

`generate_data.py` produces 5 × 80,000 rows within the brief's amount/tenure ranges, with correlated demographics, income, credit score and liquidity. `max_monthly_emi = disposable income × f(credit, stability, buffers)`, clipped to ₹500-50,000. `emi_eligibility` follows thresholds on EMI-to-capacity load, credit score and post-loan DTI. ~1.5% of labels are randomised, and ~1% of values are injected as missing/dirty so the cleaning stage has real work. Overall class mix ≈ 41% Not_Eligible, 37% High_Risk, 22% Eligible.
