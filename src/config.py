"""Shared paths, column definitions and business constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "emi_dataset.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
EDA_DIR = REPORTS_DIR / "eda"
MLRUNS_DIR = ROOT / "mlruns"
TRACKING_URI = f"sqlite:///{ROOT / 'mlflow.db'}"
ARTIFACT_ROOT = str(MLRUNS_DIR)

RANDOM_STATE = 42

CLASS_TARGET = "emi_eligibility"
REG_TARGET = "max_monthly_emi"
CLASS_LABELS = ["Eligible", "High_Risk", "Not_Eligible"]

RAW_CATEGORICAL = [
    "gender", "marital_status", "education", "employment_type", "company_type",
    "house_type", "existing_loans", "emi_scenario",
]
RAW_NUMERIC = [
    "age", "monthly_salary", "years_of_employment", "monthly_rent", "family_size", "dependents",
    "school_fees", "college_fees", "travel_expenses", "groceries_utilities",
    "other_monthly_expenses", "current_emi_amount", "credit_score", "bank_balance",
    "emergency_fund", "requested_amount", "requested_tenure",
]
RAW_FEATURES = RAW_CATEGORICAL + RAW_NUMERIC

# Indicative annual interest rate per scenario (business assumption used for EMI estimates)
SCENARIO_RATE = {
    "E-commerce Shopping EMI": 0.14,
    "Home Appliances EMI": 0.12,
    "Vehicle EMI": 0.095,
    "Personal Loan EMI": 0.13,
    "Education EMI": 0.10,
}
SCENARIO_LIMITS = {  # amount_min, amount_max, tenure_min, tenure_max
    "E-commerce Shopping EMI": (10_000, 200_000, 3, 24),
    "Home Appliances EMI": (20_000, 300_000, 6, 36),
    "Vehicle EMI": (80_000, 1_500_000, 12, 84),
    "Personal Loan EMI": (50_000, 1_000_000, 12, 60),
    "Education EMI": (50_000, 500_000, 6, 48),
}

CATEGORY_VALUES = {
    "gender": ["Male", "Female"],
    "marital_status": ["Single", "Married"],
    "education": ["High School", "Graduate", "Post Graduate", "Professional"],
    "employment_type": ["Private", "Government", "Self-employed"],
    "company_type": ["Startup", "Small", "Mid-size", "Large", "MNC"],
    "house_type": ["Rented", "Own", "Family"],
    "existing_loans": ["No", "Yes"],
    "emi_scenario": list(SCENARIO_RATE),
}

# Business targets from the brief
TARGET_ACCURACY = 0.90
TARGET_RMSE = 2000
