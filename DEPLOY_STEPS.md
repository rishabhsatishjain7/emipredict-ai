# Deploying to Streamlit Community Cloud

The repository is ready as-is — no configuration files need to be added. The exact clicks:

1. Go to <https://share.streamlit.io> and sign in with the GitHub account that owns the repository.
2. Click **New app** (top right).
3. Set: **Repository** `<you>/emipredict-ai` · **Branch** `main` · **Main file path** `streamlit_app.py`.
4. Expand **Advanced settings** → set the Python version to **3.12** (matches the training environment; the prep pipeline is a joblib pickle tied to the pinned `scikit-learn==1.9.1`).
5. Click **Deploy**. The first build installs `requirements.txt` (about 3–5 minutes).

Every push to `main` redeploys automatically; the GitHub Actions workflow (`.github/workflows/ci.yml`) runs the tests on each push.

## If the build fails, read the logs

- A failed deploy shows the build log right on the deploy page. Later: open the app → **Manage app** (⋯ menu, bottom-left) → **Logs** tab.
- Two streams matter: the **build log** (pip install failures) and the **runtime log** (Python tracebacks when the app serves requests).
- Common failures:
  - `Could not find a version that satisfies the requirement …` → a pinned package has no wheel for Linux / Python 3.12; re-pin it and push.
  - `ModuleNotFoundError` at runtime → a source file is missing from the commit (check it is not caught by `.gitignore`).
  - A friendly "Run `python src/train.py`" message on the Home/Prediction pages → the `models/` files (`.ubj`, `*_prep.joblib`, `model_manifest.json`) are not committed; they are required because the cloud never runs training.

## Known limits on Community Cloud

- The filesystem is **ephemeral**: CRUD edits (SQLite `applicants.db`) are lost on every restart or redeploy.
- The **first load may be slow** (≈10–30 s) while the two champion models load; results are cached in memory afterwards.
- MLflow UI is not deployed — the Model Performance page falls back to the committed comparison tables and says so.
