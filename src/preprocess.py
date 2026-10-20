"""Data loading, cleaning, quality assessment and train/validation/test splitting."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import (CATEGORY_VALUES, CLASS_TARGET, DATA_PATH, PROCESSED_DIR, RANDOM_STATE,
                    RAW_CATEGORICAL, RAW_NUMERIC, REG_TARGET, REPORTS_DIR)

VALID_RANGES = {
    "age": (18, 75),
    "monthly_salary": (1_000, 1_000_000),
    "credit_score": (300, 850),
    "years_of_employment": (0, 50),
    "family_size": (1, 15),
    "dependents": (0, 14),
}


def load_raw(path: Path = DATA_PATH) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python src/generate_data.py` or place the official dataset there."
        )
    return pd.read_csv(path)


def assess_quality(df: pd.DataFrame) -> dict:
    """Quality snapshot used before and after cleaning."""
    miss = df.isna().sum()
    report = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_by_column": {c: int(v) for c, v in miss.items() if v > 0},
        "missing_pct_total": round(float(df.isna().sum().sum() / df.size * 100), 4),
        "out_of_range": {},
        "categorical_inconsistencies": {},
    }
    for col, (lo, hi) in VALID_RANGES.items():
        if col in df:
            bad = int(((df[col] < lo) | (df[col] > hi)).sum())
            if bad:
                report["out_of_range"][col] = bad
    for col in RAW_CATEGORICAL:
        if col in df:
            s = df[col].dropna().astype(str)
            canon = set(CATEGORY_VALUES[col])
            bad = int((~s.isin(canon)).sum())
            if bad:
                report["categorical_inconsistencies"][col] = bad
    return report


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Deterministic cleaning. Statistical imputation is left to the sklearn pipeline so
    that medians/modes are learned on the training split only (no leakage)."""
    before = assess_quality(df)
    df = df.copy()

    # 1. Standardise categoricals: trim whitespace, canonical casing
    for col in RAW_CATEGORICAL:
        canon = {v.lower(): v for v in CATEGORY_VALUES[col]}
        df[col] = df[col].astype("string").str.strip().str.lower().map(canon)

    # 2. Impossible values -> NaN (to be imputed later)
    for col, (lo, hi) in VALID_RANGES.items():
        df.loc[(df[col] < lo) | (df[col] > hi), col] = np.nan

    # 3. Non-negative money columns
    for col in ["monthly_rent", "school_fees", "college_fees", "travel_expenses",
                "groceries_utilities", "other_monthly_expenses", "current_emi_amount",
                "bank_balance", "emergency_fund"]:
        df.loc[df[col] < 0, col] = np.nan

    # 4. Exact duplicates
    n_dups = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)

    # 5. Rows without targets are unusable for supervised learning
    df = df.dropna(subset=[CLASS_TARGET, REG_TARGET]).reset_index(drop=True)

    # 6. dependents cannot exceed family_size - 1
    mask = df["dependents"] > (df["family_size"] - 1)
    df.loc[mask, "dependents"] = (df.loc[mask, "family_size"] - 1).clip(lower=0)

    after = assess_quality(df)
    report = {"before": before, "after": after, "duplicates_removed": n_dups,
              "class_distribution": df[CLASS_TARGET].value_counts(normalize=True).round(4).to_dict()}
    return df, report


def split_data(df: pd.DataFrame, val_size: float = 0.15, test_size: float = 0.15):
    """Stratified 70/15/15 split on the classification target."""
    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df[CLASS_TARGET], random_state=RANDOM_STATE)
    rel_val = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=rel_val, stratify=train_val[CLASS_TARGET], random_state=RANDOM_STATE)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def run() -> dict:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_raw()
    clean, report = clean_data(raw)
    train, val, test = split_data(clean)
    for name, part in [("clean", clean), ("train", train), ("val", val), ("test", test)]:
        part.to_parquet(PROCESSED_DIR / f"{name}.parquet", index=False)
    clean.sample(30_000, random_state=RANDOM_STATE).to_parquet(PROCESSED_DIR / "sample_30k.parquet", index=False)
    report["splits"] = {"train": len(train), "val": len(val), "test": len(test)}
    (REPORTS_DIR / "data_quality.json").write_text(json.dumps(report, indent=2, default=str))
    return report


if __name__ == "__main__":
    rep = run()
    print(json.dumps(rep, indent=2, default=str))
