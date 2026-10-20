"""Model comparison, MLflow experiment browser and registry view."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import plotly.express as px
import streamlit as st

from app_common import page_setup, read_json
from config import MODELS_DIR, ROOT, TARGET_ACCURACY, TARGET_RMSE, TRACKING_URI

page_setup("Model Performance", "🧪")
st.title("🧪 Model Performance & MLflow")

meta = read_json(MODELS_DIR / "metadata.json")
clf_path, reg_path = MODELS_DIR / "classification_comparison.csv", MODELS_DIR / "regression_comparison.csv"
if not (clf_path.exists() and reg_path.exists()):
    st.error("No comparison results found. Run `python src/train.py`.")
    st.stop()
clf, reg = pd.read_csv(clf_path), pd.read_csv(reg_path)

tab1, tab2, tab3, tab4 = st.tabs(["Classification", "Regression", "MLflow runs", "Model registry"])

with tab1:
    st.caption(f"Champion selected on validation weighted-F1: **{meta.get('clf', {}).get('best_algorithm', '?')}** · "
               f"target accuracy ≥ {TARGET_ACCURACY:.0%}")
    cols = ["algorithm", "test_accuracy", "test_precision_weighted", "test_recall_weighted", "test_f1_weighted",
            "test_roc_auc_ovr", "val_f1_weighted", "train_accuracy", "fit_seconds"]
    st.dataframe(clf[cols].round(4), hide_index=True, width="stretch")
    m = clf.melt(id_vars="algorithm", value_vars=["test_accuracy", "test_f1_weighted", "test_roc_auc_ovr"], var_name="metric")
    st.plotly_chart(px.bar(m, x="algorithm", y="value", color="metric", barmode="group", title="Test-set metrics")
                    .update_yaxes(range=[max(0, m["value"].min() - 0.05), 1.0]), width="stretch")
    st.caption("A train-vs-test accuracy gap (columns above) is the overfitting check.")
    fi = MODELS_DIR / "clf_feature_importance.csv"
    if fi.exists():
        d = pd.read_csv(fi, index_col=0).head(15)[::-1]
        st.plotly_chart(px.bar(d, x="importance", y=d.index, orientation="h", title="Champion feature importance"), width="stretch")

with tab2:
    st.caption(f"Champion selected on validation RMSE: **{meta.get('reg', {}).get('best_algorithm', '?')}** · "
               f"target RMSE < ₹{TARGET_RMSE:,}")
    cols = ["algorithm", "test_rmse", "test_mae", "test_r2", "test_mape", "val_rmse", "train_rmse", "fit_seconds"]
    st.dataframe(reg[cols].round(4), hide_index=True, width="stretch")
    st.plotly_chart(px.bar(reg, x="algorithm", y="test_rmse", color="algorithm", title="Test RMSE (₹, lower is better)")
                    .add_hline(y=TARGET_RMSE, line_dash="dash", annotation_text="target"), width="stretch")
    fi = MODELS_DIR / "reg_feature_importance.csv"
    if fi.exists():
        d = pd.read_csv(fi, index_col=0).head(15)[::-1]
        st.plotly_chart(px.bar(d, x="importance", y=d.index, orientation="h", title="Champion feature importance"), width="stretch")

db_exists = (ROOT / "mlflow.db").exists()
with tab3:
    if not db_exists:
        st.info("The local MLflow database is not bundled in this deployment. Metrics above come from the saved "
                "comparison tables. Locally run `mlflow ui --backend-store-uri sqlite:///mlflow.db`.")
    else:
        import mlflow
        mlflow.set_tracking_uri(TRACKING_URI)
        runs = mlflow.search_runs(search_all_experiments=True, order_by=["start_time DESC"])
        if runs.empty:
            st.info("No runs logged yet.")
        else:
            runs["experiment"] = runs["experiment_id"].map({e.experiment_id: e.name for e in mlflow.search_experiments()})
            show = ["experiment", "tags.mlflow.runName", "status", "metrics.test_accuracy", "metrics.test_f1_weighted",
                    "metrics.test_rmse", "metrics.test_r2", "metrics.fit_seconds", "run_id"]
            show = [c for c in show if c in runs.columns]
            st.dataframe(runs[show].round(4), hide_index=True, width="stretch")
            st.code("mlflow ui --backend-store-uri sqlite:///mlflow.db", language="bash")

with tab4:
    if not db_exists:
        st.info("Registry data lives in the local MLflow database.")
    else:
        from mlflow import MlflowClient
        client = MlflowClient(tracking_uri=TRACKING_URI, registry_uri=TRACKING_URI)
        for rm in client.search_registered_models():
            st.subheader(rm.name)
            rows = []
            for v in client.search_model_versions(f"name='{rm.name}'"):
                rows.append({"version": int(v.version), "algorithm": v.tags.get("algorithm"),
                             "aliases": ", ".join(v.aliases) or "-", "stage tag": v.tags.get("stage", "-"),
                             "run_id": v.run_id[:8]})
            st.dataframe(pd.DataFrame(rows).sort_values("version"), hide_index=True, width="stretch")
