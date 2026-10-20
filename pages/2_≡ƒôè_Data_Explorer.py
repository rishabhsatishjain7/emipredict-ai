"""Interactive exploration of the financial portfolio."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import plotly.express as px
import streamlit as st

from app_common import CLASS_COLORS, load_sample, page_setup, read_json
from config import CLASS_LABELS, CLASS_TARGET, REG_TARGET, REPORTS_DIR
from features import engineer_features

page_setup("Data Explorer", "📊")
st.title("📊 Data Explorer")

df = load_sample()
df = engineer_features(df)

with st.sidebar:
    st.header("Filters")
    scen = st.multiselect("EMI scenario", sorted(df["emi_scenario"].unique()), default=[])
    emp = st.multiselect("Employment type", sorted(df["employment_type"].dropna().unique()), default=[])
    edu = st.multiselect("Education", sorted(df["education"].dropna().unique()), default=[])
    age_r = st.slider("Age", 25, 60, (25, 60))
    cs_r = st.slider("Credit score", 300, 850, (300, 850))

f = df[df["age"].between(*age_r) & df["credit_score"].between(*cs_r)]
if scen: f = f[f["emi_scenario"].isin(scen)]
if emp: f = f[f["employment_type"].isin(emp)]
if edu: f = f[f["education"].isin(edu)]

if f.empty:
    st.warning("No records match these filters.")
    st.stop()

k = st.columns(4)
k[0].metric("Records", f"{len(f):,}")
k[1].metric("% Eligible", f"{(f[CLASS_TARGET] == 'Eligible').mean():.1%}")
k[2].metric("Median salary", f"₹{f['monthly_salary'].median():,.0f}")
k[3].metric("Median max EMI", f"₹{f[REG_TARGET].median():,.0f}")
st.caption(f"Based on a {len(df):,}-record sample of the cleaned dataset.")

tabs = st.tabs(["Eligibility", "Financial drivers", "Demographics", "Correlations", "Data quality"])
cmap = {c: CLASS_COLORS[c] for c in CLASS_LABELS}
order = {CLASS_TARGET: CLASS_LABELS}

with tabs[0]:
    a, b = st.columns(2)
    ct = f.groupby(["emi_scenario", CLASS_TARGET]).size().reset_index(name="n")
    a.plotly_chart(px.bar(ct, x="emi_scenario", y="n", color=CLASS_TARGET, barmode="relative", color_discrete_map=cmap,
                          category_orders=order, title="Applications by scenario and outcome").update_layout(xaxis_title=None),
                   width="stretch")
    b.plotly_chart(px.pie(f, names=CLASS_TARGET, color=CLASS_TARGET, color_discrete_map=cmap, hole=0.5,
                          category_orders=order, title="Overall mix"), width="stretch")
    st.plotly_chart(px.box(f, x="emi_scenario", y="requested_amount", color=CLASS_TARGET, color_discrete_map=cmap,
                           category_orders=order, title="Requested amount by scenario and outcome"), width="stretch")

with tabs[1]:
    a, b = st.columns(2)
    a.plotly_chart(px.histogram(f, x="credit_score", color=CLASS_TARGET, nbins=50, barmode="overlay", opacity=0.65,
                                color_discrete_map=cmap, category_orders=order, title="Credit score distribution"), width="stretch")
    b.plotly_chart(px.box(f, x=CLASS_TARGET, y="dti_after_loan", color=CLASS_TARGET, color_discrete_map=cmap,
                          category_orders=order, title="Debt-to-income after new loan").update_yaxes(range=[0, 1.2]),
                   width="stretch")
    s = f.sample(min(6000, len(f)), random_state=0)
    st.plotly_chart(px.scatter(s, x="monthly_salary", y=REG_TARGET, color=CLASS_TARGET, color_discrete_map=cmap,
                               category_orders=order, opacity=0.5, title="Monthly salary vs maximum safe EMI"), width="stretch")

with tabs[2]:
    f = f.copy()
    f["age_band"] = f["age"].apply(lambda a: f"{int(a // 5 * 5)}-{int(a // 5 * 5 + 4)}" if a == a else None)
    dim = st.selectbox("Group by", ["age_band", "education", "employment_type", "house_type", "marital_status",
                                    "gender", "company_type"])
    g = (f.groupby(dim)[CLASS_TARGET].value_counts(normalize=True).rename("share").reset_index())
    st.plotly_chart(px.bar(g, x=dim, y="share", color=CLASS_TARGET, color_discrete_map=cmap, category_orders=order,
                           title=f"Outcome mix by {dim}").update_yaxes(tickformat=".0%"), width="stretch")
    st.plotly_chart(px.bar(f.groupby(dim)[REG_TARGET].median().reset_index(), x=dim, y=REG_TARGET,
                           title=f"Median maximum EMI by {dim}"), width="stretch")

with tabs[3]:
    cols = ["age", "monthly_salary", "years_of_employment", "credit_score", "bank_balance", "emergency_fund",
            "requested_amount", "requested_tenure", "expense_to_income", "debt_to_income", "dti_after_loan",
            "affordability_ratio", "risk_score", REG_TARGET]
    st.plotly_chart(px.imshow(f[cols].corr().round(2), text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                              aspect="auto", title="Correlation matrix").update_layout(height=650), width="stretch")

with tabs[4]:
    q = read_json(REPORTS_DIR / "data_quality.json")
    if q:
        a, b = st.columns(2)
        a.subheader("Before cleaning"); a.json(q["before"], expanded=False)
        b.subheader("After cleaning"); b.json(q["after"], expanded=False)
        st.write(f"Duplicates removed: **{q.get('duplicates_removed', 0):,}** · Splits: {q.get('splits')}")
    else:
        st.info("Run `python src/preprocess.py` to generate the data-quality report.")
    eda = REPORTS_DIR / "eda_report.md"
    if eda.exists():
        with st.expander("Full EDA report"):
            st.markdown(eda.read_text())
