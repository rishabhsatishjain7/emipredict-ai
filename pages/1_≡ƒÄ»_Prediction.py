"""Real-time EMI eligibility and maximum-EMI prediction."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app_common import (CLASS_COLORS, CLASS_DISPLAY, adjusted_rate, inr, max_affordable_principal, models_or_stop,
                        page_setup, predict_df)
from config import CATEGORY_VALUES, CLASS_LABELS, RAW_FEATURES, SCENARIO_LIMITS
from features import amortised_emi, engineer_features

page_setup("Prediction", "🎯")
models_or_stop()
st.title("🎯 Applicant Assessment")
st.caption("Fill in the applicant's profile. Predictions use the production champion models.")

scenario = st.selectbox("EMI scenario", CATEGORY_VALUES["emi_scenario"], index=3,
                        help="Amount and tenure limits adapt to the chosen scenario.")
amin, amax, tmin, tmax = SCENARIO_LIMITS[scenario]

with st.form("applicant"):
    t1, t2, t3, t4 = st.tabs(["👤 Personal", "💼 Employment & housing", "💸 Obligations & credit", "🏦 Loan request"])
    with t1:
        c = st.columns(2)
        age = c[0].number_input("Age", 25, 60, 34)
        gender = c[1].selectbox("Gender", CATEGORY_VALUES["gender"])
        marital = c[0].selectbox("Marital status", CATEGORY_VALUES["marital_status"], index=1)
        education = c[1].selectbox("Education", CATEGORY_VALUES["education"], index=1)
        family = c[0].number_input("Family size", 1, 10, 4)
        dependents = c[1].number_input("Dependents", 0, 8, 2)
    with t2:
        c = st.columns(2)
        salary = c[0].number_input("Monthly salary (₹)", 15_000, 200_000, 60_000, step=1_000)
        emp = c[1].selectbox("Employment type", CATEGORY_VALUES["employment_type"])
        years = c[0].number_input("Years of employment", 0.0, 40.0, 7.0, step=0.5)
        company = c[1].selectbox("Company type", CATEGORY_VALUES["company_type"], index=2)
        house = c[0].selectbox("House type", CATEGORY_VALUES["house_type"])
        rent = c[1].number_input("Monthly rent (₹)", 0, 100_000, 12_000 if house == "Rented" else 0, step=500)
    with t3:
        c = st.columns(3)
        school = c[0].number_input("School fees (₹)", 0, 100_000, 3_000, step=500)
        college = c[1].number_input("College fees (₹)", 0, 100_000, 0, step=500)
        travel = c[2].number_input("Travel expenses (₹)", 0, 50_000, 3_000, step=500)
        groc = c[0].number_input("Groceries & utilities (₹)", 0, 100_000, 10_000, step=500)
        other = c[1].number_input("Other monthly expenses (₹)", 0, 100_000, 3_000, step=500)
        loans = c[2].selectbox("Existing loans", CATEGORY_VALUES["existing_loans"])
        cur_emi = c[0].number_input("Current EMI amount (₹)", 0, 100_000, 0, step=500)
        credit = c[1].slider("Credit score", 300, 850, 690)
        bank = c[2].number_input("Bank balance (₹)", 0, 5_000_000, 150_000, step=10_000)
        emergency = c[0].number_input("Emergency fund (₹)", 0, 2_000_000, 60_000, step=5_000)
    with t4:
        c = st.columns(2)
        amount = c[0].number_input(f"Requested amount (₹) · {inr(amin)}–{inr(amax)}", amin, amax,
                                   min(max(amin, 300_000), amax), step=5_000)
        tenure = c[1].number_input(f"Requested tenure (months) · {tmin}–{tmax}", tmin, tmax, min(max(tmin, 24), tmax))
    submitted = st.form_submit_button("Assess applicant", type="primary", width="stretch")

if not submitted:
    st.info("Complete the form and press **Assess applicant**.")
    st.stop()

# ---- Input sanity checks (friendly feedback instead of silent nonsense)
issues = []
if dependents > family - 1:
    issues.append("Dependents cannot exceed family size minus one.")
if house != "Rented" and rent > 0:
    issues.append("Rent is entered but house type is not 'Rented'.")
if years > age - 18:
    issues.append("Years of employment look inconsistent with age.")
if loans == "No" and cur_emi > 0:
    issues.append("Current EMI is entered but 'Existing loans' is No.")
if loans == "Yes" and cur_emi == 0:
    issues.append("'Existing loans' is Yes but current EMI is 0.")
for msg in issues:
    st.warning(msg)
if dependents > family - 1:
    st.stop()

row = pd.DataFrame([{
    "age": age, "gender": gender, "marital_status": marital, "education": education, "monthly_salary": salary,
    "employment_type": emp, "years_of_employment": years, "company_type": company, "house_type": house,
    "monthly_rent": rent, "family_size": family, "dependents": dependents, "school_fees": school,
    "college_fees": college, "travel_expenses": travel, "groceries_utilities": groc, "other_monthly_expenses": other,
    "existing_loans": loans, "current_emi_amount": cur_emi, "credit_score": credit, "bank_balance": bank,
    "emergency_fund": emergency, "emi_scenario": scenario, "requested_amount": amount, "requested_tenure": tenure,
}])[RAW_FEATURES]

try:
    with st.spinner("Scoring…"):
        res = predict_df(row).iloc[0]
        feats = engineer_features(row).iloc[0]
except Exception as e:
    st.error(f"Prediction failed: {e}")
    st.stop()

label = res["pred_eligibility"]
max_emi = float(res["pred_max_emi"])
rate = adjusted_rate(scenario, credit)
req_emi = float(amortised_emi(amount, rate, tenure))

st.divider()
colA, colB, colC = st.columns([1.2, 1, 1])
with colA:
    st.markdown(
        f"<div style='padding:1rem 1.2rem;border-radius:12px;background:{CLASS_COLORS[label]}22;"
        f"border:2px solid {CLASS_COLORS[label]}'><div style='font-size:0.85rem;opacity:.7'>Eligibility</div>"
        f"<div style='font-size:1.9rem;font-weight:700;color:{CLASS_COLORS[label]}'>{CLASS_DISPLAY[label]}</div>"
        f"<div style='font-size:0.9rem'>Confidence {res['confidence'] * 100:.1f}%</div></div>",
        unsafe_allow_html=True,
    )
colB.metric("Maximum safe monthly EMI", inr(max_emi))
colC.metric("Requested EMI (est.)", inr(req_emi), delta=f"{inr(max_emi - req_emi)} headroom" if max_emi >= req_emi
            else f"{inr(req_emi - max_emi)} over capacity", delta_color="normal" if max_emi >= req_emi else "inverse")

g1, g2 = st.columns(2)
with g1:
    fig = go.Figure(go.Bar(x=[res[f"p_{c}"] * 100 for c in CLASS_LABELS], y=[CLASS_DISPLAY[c] for c in CLASS_LABELS],
                           orientation="h", marker_color=[CLASS_COLORS[c] for c in CLASS_LABELS],
                           text=[f"{res[f'p_{c}'] * 100:.1f}%" for c in CLASS_LABELS], textposition="outside"))
    fig.update_layout(title="Class probabilities", height=260, margin=dict(l=10, r=30, t=40, b=10),
                      xaxis=dict(range=[0, 115], title="%"), yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, width="stretch")
with g2:
    fig = go.Figure(go.Bar(x=["Max safe EMI", "Requested EMI", "Disposable income"],
                           y=[max_emi, req_emi, max(feats["disposable_income"], 0)],
                           marker_color=["#2a9d8f", "#e76f51" if req_emi > max_emi else "#457b9d", "#adb5bd"]))
    fig.update_layout(title="Capacity vs request (₹ / month)", height=260, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, width="stretch")

st.subheader("Recommendation")
if label == "Eligible":
    st.success(f"**Approve.** The requested EMI of {inr(req_emi)} fits within the safe capacity of {inr(max_emi)} "
               f"at an indicative {rate * 100:.1f}% p.a.")
elif label == "High_Risk":
    st.warning(f"**Approve with risk-based pricing or conditions.** Marginal affordability. Consider a rate premium "
               f"(illustrative: +2.0 pts → {(rate + 0.02) * 100:.1f}% p.a.), a smaller amount, or a co-applicant.")
else:
    st.error("**Not recommended as requested.**")
if label != "Eligible":
    alt_amount = max_affordable_principal(max_emi * 0.85, rate, tenure)
    longer = min(tmax, tenure + 12)
    alt_longer = max_affordable_principal(max_emi * 0.85, rate, longer)
    st.markdown(
        f"- A request of about **{inr(min(alt_amount, amax))}** over {tenure} months would sit at ~85% of safe capacity.\n"
        + (f"- Extending the tenure to {longer} months supports roughly **{inr(min(alt_longer, amax))}**.\n" if longer > tenure else "")
        + "- Reducing existing EMIs or improving the credit score are the fastest ways to raise eligibility."
    )

with st.expander("Financial profile analysis (engineered features)"):
    show = {
        "Total monthly expenses": inr(feats["total_expenses"]), "Disposable income": inr(feats["disposable_income"]),
        "Expense-to-income": f"{feats['expense_to_income']:.1%}", "Debt-to-income (current)": f"{feats['debt_to_income']:.1%}",
        "Debt-to-income after loan": f"{feats['dti_after_loan']:.1%}", "Requested EMI / disposable": f"{feats['affordability_ratio']:.2f}",
        "Savings buffer (months)": f"{feats['savings_buffer_months']:.1f}", "Composite risk score (0-1)": f"{feats['risk_score']:.2f}",
        "Indicative rate": f"{rate:.1%} p.a.",
    }
    st.table(pd.Series(show, name="Value"))
st.caption("Decision support only. Rates and conditions shown are illustrative assumptions, not an offer.")
