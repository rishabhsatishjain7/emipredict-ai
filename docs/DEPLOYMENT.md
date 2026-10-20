# Deployment: Streamlit Community Cloud

## What gets deployed

Committed to GitHub (small): code, `requirements.txt`, `models/` (champion model files described by `model_manifest.json` — native XGBoost `.ubj` boosters + prep-only `.joblib` files, ~2 MB, plus comparison CSVs), `reports/`, and `data/processed/sample_30k.parquet` (650 KB, powers the Data Explorer). Excluded by `.gitignore`: the 72 MB raw CSV, full parquet splits, `mlruns/` (247 MB), `mlflow.db`, `applicants.db`.

The app degrades gracefully without the MLflow database: the Model Performance page falls back to the saved comparison tables, and says so.

## Steps

1. Create a GitHub repo and push the project root.
   ```bash
   git init && git add . && git commit -m "EMIPredict AI"
   git branch -M main && git remote add origin https://github.com/<you>/emipredict-ai.git && git push -u origin main
   ```
2. Go to <https://share.streamlit.io>, sign in with GitHub, **Create app**.
3. Repository: `<you>/emipredict-ai`, branch `main`, main file path `streamlit_app.py`.
4. Advanced settings → choose a Python version close to the one used for training (3.12) so the prep pipeline (a joblib pickle) loads cleanly.
5. Deploy. Every push to `main` redeploys automatically (this is the "CI/CD" path); GitHub Actions (`.github/workflows/ci.yml`) runs the tests on each push.

## Gotchas

- **Model file compatibility:** `requirements.txt` pins `scikit-learn` and `xgboost` to the training versions. The XGBoost booster is stored in its native `.ubj` format and moves between platforms safely; only the prep pipeline is still a joblib pickle, so the pins still matter. If you retrain with other versions, update the pins and recommit the model files.
- **Ephemeral storage:** Community Cloud resets the filesystem on restart, so CRUD edits (SQLite) are lost. For persistence, swap `src/crud.py` for a hosted database (Postgres/Supabase) via `st.connection`.
- **Memory:** inference loads only the two small pipelines; the Data Explorer uses the 30K sample.
- Verified locally: all five pages load without exceptions, prediction submit works (including invalid-input and weak-credit paths), and the server reports healthy. **The Streamlit Cloud deployment itself was not performed** (requires your GitHub account), so no public URL exists yet.
