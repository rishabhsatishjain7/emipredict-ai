"""Feature engineering + sklearn preprocessing (shared by training and the Streamlit app)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import RAW_CATEGORICAL, RAW_NUMERIC, SCENARIO_RATE

EXPENSE_COLS = ["monthly_rent", "school_fees", "college_fees", "travel_expenses",
                "groceries_utilities", "other_monthly_expenses"]

ENGINEERED_NUMERIC = [
    "total_expenses", "expense_to_income", "debt_to_income", "disposable_income",
    "disposable_to_income", "scenario_rate", "est_requested_emi", "emi_to_income",
    "affordability_ratio", "dti_after_loan", "loan_to_annual_income", "savings_buffer_months",
    "liquid_assets_to_loan", "credit_risk_score", "employment_stability", "dependents_ratio",
    "risk_score", "income_x_credit", "income_per_member", "emi_x_tenure_burden",
]
ORDINAL_FROM_CATEGORICAL = ["education_level", "company_tier"]

NUMERIC_FEATURES = RAW_NUMERIC + ENGINEERED_NUMERIC + ORDINAL_FROM_CATEGORICAL
CATEGORICAL_FEATURES = RAW_CATEGORICAL
ALL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

_EDU = {"High School": 0, "Graduate": 1, "Post Graduate": 2, "Professional": 3}
_COMPANY = {"Startup": 0, "Small": 1, "Mid-size": 2, "Large": 3, "MNC": 4}
_EMP_WEIGHT = {"Government": 1.0, "Private": 0.8, "Self-employed": 0.6}


def _safe_div(a, b, fill=np.nan):
    b = pd.Series(b).replace(0, np.nan)
    out = pd.Series(a) / b
    return out.replace([np.inf, -np.inf], np.nan).fillna(fill)


def amortised_emi(principal, annual_rate, months):
    r = np.asarray(annual_rate, dtype=float) / 12.0
    n = np.asarray(months, dtype=float)
    p = np.asarray(principal, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        emi = p * r * (1 + r) ** n / ((1 + r) ** n - 1)
    return emi


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived financial ratios, risk scores and interaction terms. NaNs are preserved so
    the downstream imputer (fit on train only) can handle them."""
    d = df.copy()
    sal = d["monthly_salary"]

    d["total_expenses"] = d[EXPENSE_COLS].sum(axis=1, min_count=1)
    d["expense_to_income"] = _safe_div(d["total_expenses"], sal)
    d["debt_to_income"] = _safe_div(d["current_emi_amount"], sal)
    d["disposable_income"] = sal - d["total_expenses"] - d["current_emi_amount"]
    d["disposable_to_income"] = _safe_div(d["disposable_income"], sal)

    d["scenario_rate"] = d["emi_scenario"].map(SCENARIO_RATE)
    # Credit-adjusted pricing: weaker scores pay up to 4 pts more (mirrors risk-based pricing)
    adj_rate = d["scenario_rate"] + ((700 - d["credit_score"]) / 700 * 0.04)
    d["est_requested_emi"] = amortised_emi(d["requested_amount"], adj_rate, d["requested_tenure"])
    d["emi_to_income"] = _safe_div(d["est_requested_emi"], sal)
    d["affordability_ratio"] = _safe_div(d["est_requested_emi"], d["disposable_income"].clip(lower=1))
    d["dti_after_loan"] = _safe_div(d["current_emi_amount"] + d["est_requested_emi"], sal)
    d["loan_to_annual_income"] = _safe_div(d["requested_amount"], sal * 12)

    outflow = d["total_expenses"] + d["current_emi_amount"]
    d["savings_buffer_months"] = _safe_div(d["bank_balance"] + d["emergency_fund"], outflow + 1)
    d["liquid_assets_to_loan"] = _safe_div(d["bank_balance"] + d["emergency_fund"], d["requested_amount"])

    d["credit_risk_score"] = (d["credit_score"] - 300) / 550
    d["employment_stability"] = d["years_of_employment"] * d["employment_type"].map(_EMP_WEIGHT)
    d["dependents_ratio"] = _safe_div(d["dependents"], d["family_size"])
    d["income_per_member"] = _safe_div(sal, d["family_size"])

    # Composite 0-1 risk score (higher = riskier), documented in docs/METHODOLOGY.md
    d["risk_score"] = (
        0.35 * (1 - d["credit_risk_score"].clip(0, 1))
        + 0.25 * d["dti_after_loan"].clip(0, 1.5) / 1.5
        + 0.20 * d["expense_to_income"].clip(0, 1.5) / 1.5
        + 0.10 * (1 - (d["employment_stability"] / 20).clip(0, 1))
        + 0.10 * (1 - d["savings_buffer_months"].clip(0, 12) / 12)
    )

    # Interactions
    d["income_x_credit"] = sal * d["credit_risk_score"]
    d["emi_x_tenure_burden"] = d["emi_to_income"] * d["requested_tenure"]

    # Ordinal encodings of ordered categoricals
    d["education_level"] = d["education"].map(_EDU)
    d["company_tier"] = d["company_type"].map(_COMPANY)

    d = d.replace([np.inf, -np.inf], np.nan)
    return d


def build_preprocessor(scale: bool = True) -> ColumnTransformer:
    """Impute -> (scale numerics) -> one-hot categoricals. Fit on training data only."""
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    numeric = Pipeline(num_steps)
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        [("num", numeric, NUMERIC_FEATURES), ("cat", categorical, CATEGORICAL_FEATURES)],
        remainder="drop", verbose_feature_names_out=False,
    )


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    return list(preprocessor.get_feature_names_out())
