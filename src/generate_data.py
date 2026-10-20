"""Synthetic EMI dataset generator (400,000 records, 5 scenarios x 80,000).

The real `EMI_dataset` was not supplied, so this script produces a dataset that follows
the schema and domain rules from the project brief. If you have the official CSV, drop it
at data/emi_dataset.csv and skip this step - every downstream module reads that path.

Usage:  python src/generate_data.py [--n-per-scenario 80000] [--seed 42]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "emi_dataset.csv"

SCENARIOS = {
    # name: (amount_min, amount_max, tenure_min, tenure_max, annual_rate)
    "E-commerce Shopping EMI": (10_000, 200_000, 3, 24, 0.14),
    "Home Appliances EMI": (20_000, 300_000, 6, 36, 0.12),
    "Vehicle EMI": (80_000, 1_500_000, 12, 84, 0.095),
    "Personal Loan EMI": (50_000, 1_000_000, 12, 60, 0.13),
    "Education EMI": (50_000, 500_000, 6, 48, 0.10),
}


def emi_formula(principal, annual_rate, months):
    r = annual_rate / 12.0
    return principal * r * (1 + r) ** months / ((1 + r) ** months - 1)


def generate_scenario(name: str, n: int, rng: np.random.Generator) -> pd.DataFrame:
    amin, amax, tmin, tmax, rate = SCENARIOS[name]

    age = rng.integers(25, 61, n)
    gender = rng.choice(["Male", "Female"], n, p=[0.55, 0.45])
    marital = np.where(rng.random(n) < np.clip((age - 22) / 25, 0.1, 0.85), "Married", "Single")
    education = rng.choice(
        ["High School", "Graduate", "Post Graduate", "Professional"], n, p=[0.18, 0.42, 0.25, 0.15]
    )
    edu_boost = pd.Series(education).map(
        {"High School": 0.0, "Graduate": 0.25, "Post Graduate": 0.45, "Professional": 0.65}
    ).to_numpy()

    employment = rng.choice(["Private", "Government", "Self-employed"], n, p=[0.58, 0.17, 0.25])
    company = rng.choice(["Startup", "Small", "Mid-size", "Large", "MNC"], n, p=[0.12, 0.2, 0.28, 0.2, 0.2])
    max_exp = np.clip(age - 22, 0, 38)
    years_emp = np.round(rng.random(n) * max_exp * 0.95, 1)

    # Salary: lognormal, lifted by education/experience/company, clipped to 15K-200K
    comp_boost = pd.Series(company).map(
        {"Startup": 0.0, "Small": 0.05, "Mid-size": 0.15, "Large": 0.25, "MNC": 0.35}
    ).to_numpy()
    log_sal = 9.9 + edu_boost * 0.45 + comp_boost * 0.35 + np.log1p(years_emp) * 0.12 + rng.normal(0, 0.35, n)
    salary = np.clip(np.exp(log_sal), 15_000, 200_000).round(-2)

    house = rng.choice(["Rented", "Own", "Family"], n, p=[0.45, 0.25, 0.30])
    rent = np.where(house == "Rented", np.clip(salary * rng.uniform(0.12, 0.32, n), 2_000, 60_000), 0).round(-2)

    family_size = np.clip(1 + (marital == "Married") + rng.poisson(1.3, n), 1, 8)
    dependents = np.clip(family_size - 1 - (marital == "Married") + rng.integers(-1, 2, n), 0, 6)
    dependents = np.minimum(dependents, family_size - 1)

    school_fees = np.where(dependents > 0, dependents * rng.uniform(1_000, 6_000, n) * rng.binomial(1, 0.6, n), 0).round(-2)
    college_fees = np.where(dependents > 0, rng.binomial(1, 0.18, n) * rng.uniform(3_000, 15_000, n), 0).round(-2)
    travel = np.clip(salary * rng.uniform(0.02, 0.07, n), 500, 12_000).round(-2)
    groceries = np.clip(family_size * rng.uniform(2_500, 4_500, n), 3_000, 30_000).round(-2)
    other = np.clip(salary * rng.uniform(0.02, 0.08, n), 500, 15_000).round(-2)

    existing_loans = rng.random(n) < (0.15 + 0.35 * (salary > 40_000) * 0.6)
    current_emi = np.where(existing_loans, np.clip(salary * rng.uniform(0.05, 0.30, n), 1_000, 50_000), 0).round(-2)

    # Credit score: higher with experience/govt employment, lower with heavy existing EMI burden
    cs_base = 560 + years_emp * 3.2 + (employment == "Government") * 35 + edu_boost * 30
    cs_base -= (current_emi / salary) * 180
    credit = np.clip(cs_base + rng.normal(0, 55, n), 300, 850).round().astype(int)

    bank = np.clip(salary * rng.uniform(0.3, 9.0, n) * (0.6 + 0.4 * (credit - 300) / 550), 1_000, 2_500_000).round(-2)
    emergency = np.clip(salary * rng.uniform(0.0, 5.0, n) * (0.5 + 0.5 * (credit - 300) / 550), 0, 1_000_000).round(-2)

    # Loan application: requested amount scales with salary within scenario range
    u = np.clip(rng.beta(1.1, 4.0, n) * (0.35 + 0.9 * np.clip(salary / 120_000, 0, 1)), 0, 1)
    amount = (amin + u * (amax - amin)).round(-3)
    tenure = rng.integers(tmin, tmax + 1, n)

    # ---- Targets ---------------------------------------------------------------
    obligations = rent + school_fees + college_fees + travel + groceries + other + current_emi
    disposable = salary - obligations
    credit_factor = 0.30 + 0.35 * (credit - 300) / 550
    stability = 0.04 * (employment == "Government") + 0.02 * np.clip(years_emp / 15, 0, 1) - 0.03 * (employment == "Self-employed")
    buffer = 0.03 * np.clip(np.log1p(emergency / salary) / 2, 0, 1) + 0.02 * np.clip(np.log1p(bank / salary) / 3, 0, 1)
    ratio = np.clip(credit_factor + stability + buffer, 0.2, 0.75)
    max_emi = np.where(disposable > 0, disposable * ratio, 0) * (1 + rng.normal(0, 0.015, n))
    max_emi = np.clip(max_emi, 500, 50_000).round(-1)

    req_emi = emi_formula(amount, rate + (700 - credit) / 700 * 0.04, tenure)
    load = req_emi / max_emi
    dti_after = (current_emi + req_emi) / salary

    label = np.full(n, "Not_Eligible", dtype=object)
    high_risk = (load <= 1.6) & (credit >= 520) & (dti_after <= 0.75)
    eligible = (load <= 1.0) & (credit >= 620) & (dti_after <= 0.55) & (disposable > 0)
    label[high_risk] = "High_Risk"
    label[eligible] = "Eligible"
    # small amount of label noise (borderline human-underwriter disagreement)
    flip = rng.random(n) < 0.015
    shifted = rng.choice(["Eligible", "High_Risk", "Not_Eligible"], n)
    label = np.where(flip, shifted, label)

    return pd.DataFrame({
        "age": age, "gender": gender, "marital_status": marital, "education": education,
        "monthly_salary": salary, "employment_type": employment, "years_of_employment": years_emp,
        "company_type": company, "house_type": house, "monthly_rent": rent,
        "family_size": family_size, "dependents": dependents,
        "school_fees": school_fees, "college_fees": college_fees, "travel_expenses": travel,
        "groceries_utilities": groceries, "other_monthly_expenses": other,
        "existing_loans": np.where(existing_loans, "Yes", "No"), "current_emi_amount": current_emi,
        "credit_score": credit, "bank_balance": bank, "emergency_fund": emergency,
        "emi_scenario": name, "requested_amount": amount, "requested_tenure": tenure,
        "emi_eligibility": label, "max_monthly_emi": max_emi,
    })


def inject_dirt(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Real-world messiness so the cleaning pipeline has something to do."""
    n = len(df)
    for col, frac in [("education", 0.012), ("monthly_salary", 0.006), ("credit_score", 0.008),
                      ("bank_balance", 0.01), ("years_of_employment", 0.007)]:
        df.loc[rng.random(n) < frac, col] = np.nan
    # inconsistent casing / whitespace in a categorical
    mask = rng.random(n) < 0.004
    df.loc[mask, "gender"] = df.loc[mask, "gender"].str.lower()
    mask = rng.random(n) < 0.003
    df.loc[mask, "employment_type"] = " " + df.loc[mask, "employment_type"] + " "
    # a few impossible values
    df.loc[rng.random(n) < 0.0005, "age"] = -1
    df.loc[rng.random(n) < 0.0005, "credit_score"] = 0
    # exact duplicates (~0.1%)
    dups = df.sample(frac=0.001, random_state=1)
    return pd.concat([df, dups], ignore_index=True).sample(frac=1, random_state=7).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-scenario", type=int, default=80_000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    parts = [generate_scenario(name, args.n_per_scenario, rng) for name in SCENARIOS]
    df = pd.concat(parts, ignore_index=True)
    df = inject_dirt(df, rng)
    OUT.parent.mkdir(exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Wrote {len(df):,} rows x {df.shape[1]} cols -> {OUT}")
    print(df["emi_eligibility"].value_counts(normalize=True).round(3).to_string())
    print(df.groupby("emi_scenario")["emi_eligibility"].value_counts(normalize=True).unstack().round(3).to_string())
    print(df["max_monthly_emi"].describe().round(0).to_string())


if __name__ == "__main__":
    main()
