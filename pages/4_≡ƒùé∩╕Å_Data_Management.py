"""Administrative CRUD interface for applicant records + batch scoring."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import streamlit as st

import crud
from app_common import inr, load_models, load_sample, page_setup, predict_df
from config import CATEGORY_VALUES, CLASS_LABELS, RAW_CATEGORICAL, RAW_FEATURES

page_setup("Data Management", "🗂️")
st.title("🗂️ Data Management")
st.caption("Create, read, update and delete applicant records. Stored in a local SQLite database.")

crud.init_db()
models_ready = True
try:
    load_models()
except Exception:
    models_ready = False

# Seed on first run
if crud.count() == 0:
    seed = load_sample().head(500)[RAW_FEATURES].copy()
    if models_ready:
        seed = seed.join(predict_df(seed)[crud.PRED_COLS])
    crud.create_many(seed)
    st.toast("Seeded 500 sample applicants")

tab_read, tab_create, tab_edit, tab_delete, tab_batch = st.tabs(
    ["📖 Browse", "➕ Add", "✏️ Edit", "🗑️ Delete", "📥 Batch score CSV"])

# ------------------------------------------------------------------ READ
with tab_read:
    c = st.columns(4)
    scen = c[0].multiselect("Scenario", CATEGORY_VALUES["emi_scenario"])
    emp = c[1].multiselect("Employment", CATEGORY_VALUES["employment_type"])
    verdict = c[2].multiselect("Predicted eligibility", CLASS_LABELS)
    limit = c[3].selectbox("Rows", [100, 250, 500, 1000], index=1)
    data = crud.read({"emi_scenario": scen, "employment_type": emp, "pred_eligibility": verdict}, limit=limit)
    st.write(f"**{len(data):,}** records shown · **{crud.count():,}** total")
    st.dataframe(data, hide_index=True, width="stretch", height=420)
    st.download_button("Download shown rows (CSV)", data.to_csv(index=False).encode(), "applicants.csv", "text/csv")

# ------------------------------------------------------------------ CREATE
with tab_create:
    st.write("Add a single applicant. Missing values aren't allowed here; use the batch tab for bulk loads.")
    with st.form("create"):
        cols = st.columns(4)
        rec = {}
        for i, f in enumerate(RAW_FEATURES):
            col = cols[i % 4]
            if f in RAW_CATEGORICAL:
                rec[f] = col.selectbox(f, CATEGORY_VALUES[f], key=f"c_{f}")
            else:
                default = {"age": 35, "monthly_salary": 50_000, "credit_score": 680, "family_size": 3,
                           "requested_amount": 200_000, "requested_tenure": 24, "years_of_employment": 5}.get(f, 0)
                rec[f] = col.number_input(f, value=float(default), key=f"c_{f}")
        auto = st.checkbox("Score with the models on save", value=models_ready, disabled=not models_ready)
        ok = st.form_submit_button("Create record", type="primary")
    if ok:
        if rec["dependents"] > rec["family_size"] - 1:
            st.error("Dependents cannot exceed family size minus one.")
        elif not 300 <= rec["credit_score"] <= 850:
            st.error("Credit score must be between 300 and 850.")
        elif rec["monthly_salary"] <= 0:
            st.error("Monthly salary must be positive.")
        else:
            row = pd.DataFrame([rec])
            if auto:
                row = row.join(predict_df(row)[crud.PRED_COLS])
            crud.create_many(row)
            st.success("Record created.")

# ------------------------------------------------------------------ UPDATE
with tab_edit:
    rid = st.number_input("Record ID to edit", min_value=1, step=1, value=1)
    found = crud.read(limit=1_000_000)
    rec_df = found[found["id"] == rid]
    if rec_df.empty:
        st.info("No record with that ID.")
    else:
        rec = rec_df.iloc[0]
        with st.form("edit"):
            cols = st.columns(4)
            new = {}
            for i, f in enumerate(RAW_FEATURES):
                col = cols[i % 4]
                if f in RAW_CATEGORICAL:
                    opts = CATEGORY_VALUES[f]
                    new[f] = col.selectbox(f, opts, index=opts.index(rec[f]) if rec[f] in opts else 0, key=f"e_{f}")
                else:
                    new[f] = col.number_input(f, value=float(rec[f]) if pd.notna(rec[f]) else 0.0, key=f"e_{f}")
            rescore = st.checkbox("Re-score after saving", value=models_ready, disabled=not models_ready)
            save = st.form_submit_button("Save changes", type="primary")
        if save:
            if new["dependents"] > new["family_size"] - 1:
                st.error("Dependents cannot exceed family size minus one.")
            else:
                if rescore:
                    new.update(predict_df(pd.DataFrame([new]))[crud.PRED_COLS].iloc[0].to_dict())
                crud.update(int(rid), new)
                st.success(f"Record {int(rid)} updated.")
                if rescore:
                    st.write(f"New prediction: **{new['pred_eligibility']}**, max EMI {inr(new['pred_max_emi'])}")

# ------------------------------------------------------------------ DELETE
with tab_delete:
    ids_text = st.text_input("Record IDs to delete (comma-separated)", placeholder="e.g. 3, 17, 42")
    confirm = st.checkbox("I understand this cannot be undone")
    if st.button("Delete", type="primary", disabled=not confirm):
        try:
            ids = [int(x) for x in ids_text.replace(" ", "").split(",") if x]
        except ValueError:
            st.error("IDs must be whole numbers.")
        else:
            n = crud.delete(ids)
            st.success(f"Deleted {n} record(s).") if n else st.info("No matching records.")

# ------------------------------------------------------------------ BATCH
with tab_batch:
    st.write("Upload a CSV with the 25 applicant columns to score it in bulk.")
    st.download_button("Download CSV template", pd.DataFrame(columns=RAW_FEATURES).to_csv(index=False).encode(),
                       "template.csv", "text/csv")
    up = st.file_uploader("CSV file", type="csv")
    if up is not None:
        try:
            df = pd.read_csv(up)
        except Exception as e:
            st.error(f"Could not read file: {e}")
            st.stop()
        missing = [c for c in RAW_FEATURES if c not in df.columns]
        if missing:
            st.error(f"Missing columns: {', '.join(missing)}")
        elif not models_ready:
            st.error("Models are not available.")
        else:
            if len(df) > 50_000:
                st.warning("Only the first 50,000 rows are scored.")
                df = df.head(50_000)
            scored = df.join(predict_df(df))
            st.dataframe(scored.head(200), hide_index=True, width="stretch")
            st.write(scored["pred_eligibility"].value_counts().rename("count").to_frame().T)
            a, b = st.columns(2)
            a.download_button("Download scored CSV", scored.to_csv(index=False).encode(), "scored.csv", "text/csv")
            if b.button("Save all to database"):
                crud.create_many(scored[RAW_FEATURES + crud.PRED_COLS])
                st.success(f"Saved {len(scored):,} records.")
