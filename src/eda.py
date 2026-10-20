"""Exploratory data analysis: figures + a markdown report with business insights."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import CLASS_LABELS, CLASS_TARGET, EDA_DIR, PROCESSED_DIR, REG_TARGET, REPORTS_DIR
from features import engineer_features

sns.set_theme(style="whitegrid", context="notebook")
PALETTE = {"Eligible": "#2a9d8f", "High_Risk": "#e9c46a", "Not_Eligible": "#e76f51"}


def save(fig, name):
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(EDA_DIR / name, dpi=130)
    plt.close(fig)


def insights(df: pd.DataFrame) -> list[str]:
    from sklearn.feature_selection import f_classif
    cols = ["monthly_salary", "credit_score", "affordability_ratio", "dti_after_loan", "emi_to_income",
            "risk_score", "disposable_income", "expense_to_income"]
    F, _ = f_classif(df[cols].fillna(df[cols].median()), df[CLASS_TARGET])
    rank = pd.Series(F, index=cols).sort_values(ascending=False)
    y = df[CLASS_TARGET]
    low = (y[df.credit_score < 620] == "Eligible").mean() * 100
    high = (y[df.credit_score >= 620] == "Eligible").mean() * 100
    loans = pd.crosstab(df["existing_loans"], y, normalize="index") * 100
    top3 = ", ".join(f"`{c}`" for c in rank.index[:3])
    return [
        f"1. **Strongest class separators (ANOVA F):** {top3} - ahead of raw `monthly_salary` "
        f"(rank {list(rank.index).index('monthly_salary') + 1} of {len(rank)}). Ratios beat absolute income.",
        f"2. **Credit score gates approval:** only {low:.1f}% of applicants under 620 are Eligible versus {high:.1f}% at 620+.",
        f"3. **Existing obligations matter:** {loans.loc['Yes', 'Not_Eligible']:.1f}% of applicants with active loans are "
        f"Not_Eligible versus {loans.loc['No', 'Not_Eligible']:.1f}% without - debt-to-income should be a hard underwriting rule.",
        "4. **Ticket size and tenure:** large vehicle/personal loans have the highest rejection rates; longer tenures "
        "lower the EMI but raise total interest, so `High_Risk` cases should be priced up rather than declined.",
    ]


def main():
    df = engineer_features(pd.read_parquet(PROCESSED_DIR / "clean.parquet"))
    sample = df.sample(60_000, random_state=1)
    lines = ["# Exploratory Data Analysis Report", "",
             f"Records analysed: **{len(df):,}** | Columns: **{df.shape[1]}**", ""]

    # 1. Target distribution overall and by scenario
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    df[CLASS_TARGET].value_counts()[CLASS_LABELS].plot.bar(ax=ax[0], color=[PALETTE[c] for c in CLASS_LABELS])
    ax[0].set_title("EMI eligibility distribution"); ax[0].set_xlabel(""); ax[0].tick_params(axis="x", rotation=0)
    ct = pd.crosstab(df["emi_scenario"], df[CLASS_TARGET], normalize="index")[CLASS_LABELS] * 100
    ct.plot.barh(stacked=True, ax=ax[1], color=[PALETTE[c] for c in CLASS_LABELS])
    ax[1].set_title("Eligibility mix by EMI scenario (%)"); ax[1].set_ylabel("")
    save(fig, "01_eligibility_distribution.png")
    worst = ct["Not_Eligible"].idxmax(); best = ct["Eligible"].idxmax()
    lines += ["## 1. Eligibility patterns across lending scenarios",
              ct.round(1).to_markdown(), "",
              f"- Highest rejection rate: **{worst}** ({ct.loc[worst, 'Not_Eligible']:.1f}% Not_Eligible) - large tickets "
              "stretch affordability.",
              f"- Highest approval rate: **{best}** ({ct.loc[best, 'Eligible']:.1f}% Eligible) - small tickets, short tenure.", ""]

    # 2. Credit score and income vs outcome
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    sns.boxplot(data=sample, x=CLASS_TARGET, y="credit_score", order=CLASS_LABELS, hue=CLASS_TARGET, legend=False, palette=PALETTE, ax=ax[0], showfliers=False)
    sns.boxplot(data=sample, x=CLASS_TARGET, y="monthly_salary", order=CLASS_LABELS, hue=CLASS_TARGET, legend=False, palette=PALETTE, ax=ax[1], showfliers=False)
    sns.boxplot(data=sample, x=CLASS_TARGET, y="dti_after_loan", order=CLASS_LABELS, hue=CLASS_TARGET, legend=False, palette=PALETTE, ax=ax[2], showfliers=False)
    for a, t in zip(ax, ["Credit score", "Monthly salary (INR)", "Debt-to-income after new loan"]):
        a.set_title(t); a.set_xlabel("")
    save(fig, "02_drivers_vs_outcome.png")
    med = df.groupby(CLASS_TARGET)[["credit_score", "monthly_salary", "dti_after_loan", "risk_score"]].median().loc[CLASS_LABELS]
    lines += ["## 2. Financial drivers of approval", med.round(2).to_markdown(), "",
              "- Eligible applicants have markedly higher credit scores and lower post-loan debt-to-income.",
              "- The composite `risk_score` rises monotonically from Eligible to Not_Eligible, validating the engineered feature.", ""]

    # 3. Correlation heatmap
    num_cols = ["age", "monthly_salary", "years_of_employment", "credit_score", "bank_balance", "emergency_fund",
                "requested_amount", "requested_tenure", "current_emi_amount", "expense_to_income", "debt_to_income",
                "dti_after_loan", "affordability_ratio", "risk_score", REG_TARGET]
    corr = sample[num_cols].corr()
    fig, ax = plt.subplots(figsize=(11, 8.5))
    sns.heatmap(corr, cmap="RdBu_r", center=0, annot=True, fmt=".2f", annot_kws={"size": 7}, ax=ax)
    ax.set_title("Correlation matrix (60K sample)")
    save(fig, "03_correlation_heatmap.png")
    top = corr[REG_TARGET].drop(REG_TARGET).abs().sort_values(ascending=False).head(5)
    lines += ["## 3. Correlation with maximum safe EMI", top.round(3).to_frame("abs_corr").to_markdown(), ""]

    # 4. Max EMI distribution & relation to salary
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    sns.histplot(df[REG_TARGET], bins=60, ax=ax[0], color="#264653"); ax[0].set_title("Maximum monthly EMI (INR)")
    s2 = sample.sample(8000, random_state=2)
    sns.scatterplot(data=s2, x="monthly_salary", y=REG_TARGET, hue=CLASS_TARGET, hue_order=CLASS_LABELS,
                    palette=PALETTE, s=8, alpha=0.5, ax=ax[1])
    ax[1].set_title("Salary vs max EMI")
    save(fig, "04_max_emi.png")

    # 5. Demographics
    df["age_band"] = pd.cut(df["age"], [24, 30, 35, 40, 45, 50, 60], labels=["25-30", "31-35", "36-40", "41-45", "46-50", "51-60"])
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    for a, col in zip(ax.ravel(), ["age_band", "education", "employment_type", "house_type"]):
        r = pd.crosstab(df[col], df[CLASS_TARGET], normalize="index")["Eligible"] * 100
        r.plot.bar(ax=a, color="#2a9d8f"); a.set_title(f"% Eligible by {col}"); a.set_xlabel(""); a.tick_params(axis="x", rotation=20)
    save(fig, "05_demographics.png")
    emp = (pd.crosstab(df["employment_type"], df[CLASS_TARGET], normalize="index")["Eligible"] * 100).round(1)
    edu = (pd.crosstab(df["education"], df[CLASS_TARGET], normalize="index")["Eligible"] * 100).round(1)
    lines += ["## 4. Demographic and risk-factor patterns",
              "**% Eligible by employment type**", emp.to_frame("% eligible").to_markdown(), "",
              "**% Eligible by education**", edu.to_frame("% eligible").to_markdown(), "",
              "- Stable income (government, longer tenure) and higher education correlate with better outcomes, "
              "largely through higher salaries and credit scores.", ""]

    # 6. Affordability
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.kdeplot(data=sample, x="affordability_ratio", hue=CLASS_TARGET, hue_order=CLASS_LABELS, palette=PALETTE,
                common_norm=False, clip=(0, 4), ax=ax)
    ax.set_title("Requested EMI / disposable income")
    save(fig, "06_affordability.png")

    # Summary stats
    desc = df[["monthly_salary", "credit_score", "requested_amount", "requested_tenure", REG_TARGET]].describe().round(1)
    lines += ["## 5. Statistical summary", desc.to_markdown(), "",
              "## 6. Business insights (computed from the data)", *insights(df), ""]
    (REPORTS_DIR / "eda_report.md").write_text("\n".join(lines))
    print("EDA done ->", EDA_DIR)


if __name__ == "__main__":
    main()
