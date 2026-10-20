"""Train classification + regression models, track everything in MLflow, register the winners.

Usage:
    python src/train.py                # full run on all 400K records
    python src/train.py --quick        # 60K-row smoke test
    python src/train.py --task clf     # only classification (or: reg)
"""
from __future__ import annotations

import argparse
import json
import platform
import time
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import xgboost
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, f1_score,
                             mean_absolute_error, mean_absolute_percentage_error, mean_squared_error,
                             precision_score, r2_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor

from config import (ARTIFACT_ROOT, CLASS_LABELS, CLASS_TARGET, MODELS_DIR, PROCESSED_DIR, RANDOM_STATE,
                    RAW_FEATURES, REG_TARGET, REPORTS_DIR, TARGET_ACCURACY, TARGET_RMSE, TRACKING_URI)
from features import build_preprocessor, engineer_features, get_feature_names

warnings.filterwarnings("ignore")
LABEL_TO_ID = {c: i for i, c in enumerate(CLASS_LABELS)}
CLF_EXPERIMENT, REG_EXPERIMENT = "EMI_Eligibility_Classification", "EMI_MaxEMI_Regression"
CLF_REGISTRY, REG_REGISTRY = "EMIPredict_Classifier", "EMIPredict_Regressor"


# --------------------------------------------------------------------------- models
def classifiers(quick: bool):
    n = 60 if quick else 120
    return {
        "LogisticRegression": LogisticRegression(max_iter=400, C=1.0, random_state=RANDOM_STATE),
        "DecisionTree": DecisionTreeClassifier(max_depth=12, min_samples_leaf=20, random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(n_estimators=n, max_depth=16, min_samples_leaf=5, max_samples=0.5,
                                               n_jobs=2, random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(n_estimators=600, learning_rate=0.08, max_depth=7, subsample=0.8,
                                 colsample_bytree=0.8, tree_method="hist", n_jobs=2, eval_metric="mlogloss",
                                 early_stopping_rounds=30, random_state=RANDOM_STATE),
    }


def regressors(quick: bool):
    n = 60 if quick else 120
    return {
        "LinearRegression": LinearRegression(),
        "DecisionTree": DecisionTreeRegressor(max_depth=14, min_samples_leaf=20, random_state=RANDOM_STATE),
        "RandomForest": RandomForestRegressor(n_estimators=n, max_depth=16, min_samples_leaf=5, max_samples=0.5,
                                              n_jobs=2, random_state=RANDOM_STATE),
        "XGBoost": XGBRegressor(n_estimators=800, learning_rate=0.08, max_depth=7, subsample=0.8,
                                colsample_bytree=0.8, tree_method="hist", n_jobs=2, eval_metric="rmse",
                                early_stopping_rounds=30, random_state=RANDOM_STATE),
    }


# --------------------------------------------------------------------------- metrics
def clf_metrics(y_true, y_pred, proba, prefix):
    return {
        f"{prefix}_accuracy": accuracy_score(y_true, y_pred),
        f"{prefix}_precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        f"{prefix}_recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        f"{prefix}_f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        f"{prefix}_f1_macro": f1_score(y_true, y_pred, average="macro"),
        f"{prefix}_roc_auc_ovr": roc_auc_score(y_true, proba, multi_class="ovr", average="macro"),
    }


def reg_metrics(y_true, y_pred, prefix):
    return {
        f"{prefix}_rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        f"{prefix}_mae": mean_absolute_error(y_true, y_pred),
        f"{prefix}_r2": r2_score(y_true, y_pred),
        f"{prefix}_mape": mean_absolute_percentage_error(y_true, y_pred) * 100,
    }


# --------------------------------------------------------------------------- plots
def plot_confusion(y_true, y_pred, path):
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion matrix (test, row-normalised)")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def importances(model, names):
    if hasattr(model, "feature_importances_"):
        vals = model.feature_importances_
    elif hasattr(model, "coef_"):
        vals = np.abs(np.atleast_2d(model.coef_)).mean(axis=0)
    else:
        return None
    return pd.Series(vals, index=names).sort_values(ascending=False)


def plot_importance(imp, path, title):
    top = imp.head(15)[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top.index, top.values, color="#2a9d8f"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def plot_regression(y_true, y_pred, path):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    idx = np.random.default_rng(0).choice(len(y_true), min(6000, len(y_true)), replace=False)
    ax[0].scatter(np.asarray(y_true)[idx], np.asarray(y_pred)[idx], s=4, alpha=0.4)
    lim = [0, max(np.max(y_true), np.max(y_pred))]
    ax[0].plot(lim, lim, "r--", lw=1); ax[0].set_xlabel("Actual max EMI"); ax[0].set_ylabel("Predicted"); ax[0].set_title("Actual vs predicted (test)")
    ax[1].hist(np.asarray(y_true) - np.asarray(y_pred), bins=80, color="#264653"); ax[1].set_title("Residuals (INR)")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


# --------------------------------------------------------------------------- core
def make_pre():
    return Pipeline([
        ("engineer", FunctionTransformer(engineer_features, validate=False)),
        ("columns", build_preprocessor(scale=True)),
    ])


def setup_mlflow(name):
    mlflow.set_tracking_uri(TRACKING_URI)
    if mlflow.get_experiment_by_name(name) is None:
        mlflow.create_experiment(name, artifact_location=(Path(ARTIFACT_ROOT) / name).as_uri())
    mlflow.set_experiment(name)


def save_champion(task: str, pipe: Pipeline, algorithm: str, models_dir: Path = MODELS_DIR) -> dict:
    """Export the winning pipeline portably and record it in models/model_manifest.json.

    Pickling an XGBoost estimator inside a pipeline is not portable across
    platforms/builds (loading can fail with "input stream corrupted"), so
    XGBoost champions are split: the booster is written in its native .ubj
    format and only the prep-only pipeline goes through joblib. Non-XGBoost
    champions stay a single full-pipeline joblib file. The loader in
    src/app_common.py reads the manifest and rebuilds an equivalent Pipeline
    either way.
    """
    stem = "best_classifier" if task == "clf" else "best_regressor"
    model = pipe.named_steps["model"]
    models_dir.mkdir(exist_ok=True)
    if isinstance(model, (XGBClassifier, XGBRegressor)):
        fmt = "xgboost_native"
        files = {"prep": f"{stem}_prep.joblib", "model": f"{stem}.ubj"}
        joblib.dump(pipe.named_steps["prep"], models_dir / files["prep"], compress=3)
        model.save_model(str(models_dir / files["model"]))
    else:
        fmt = "joblib_pipeline"
        files = {"pipeline": f"{stem}.joblib"}
        joblib.dump(pipe, models_dir / files["pipeline"], compress=3)
    entry = {
        "algorithm": algorithm,
        "format": fmt,
        "files": files,
        "versions": {"python": platform.python_version(),
                     "scikit-learn": sklearn.__version__,
                     "xgboost": xgboost.__version__},
    }
    manifest_p = models_dir / "model_manifest.json"
    manifest = json.loads(manifest_p.read_text(encoding="utf-8")) if manifest_p.exists() else {}
    manifest[task] = entry
    manifest_p.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[{task}] saved champion ({fmt}) -> {', '.join(files.values())}")
    return entry


def run_task(task: str, train, val, test, quick: bool, tmp):
    is_clf = task == "clf"
    exp, registry = (CLF_EXPERIMENT, CLF_REGISTRY) if is_clf else (REG_EXPERIMENT, REG_REGISTRY)
    setup_mlflow(exp)
    client = MlflowClient()
    target = CLASS_TARGET if is_clf else REG_TARGET
    enc = (lambda s: s.map(LABEL_TO_ID).to_numpy()) if is_clf else (lambda s: s.to_numpy(dtype=float))
    Xtr, Xva, Xte = train[RAW_FEATURES], val[RAW_FEATURES], test[RAW_FEATURES]
    ytr, yva, yte = enc(train[target]), enc(val[target]), enc(test[target])

    pre = make_pre()
    t0 = time.time(); Xtr_t = pre.fit_transform(Xtr); Xva_t, Xte_t = pre.transform(Xva), pre.transform(Xte)
    names = get_feature_names(pre.named_steps["columns"])
    print(f"[{task}] preprocessing done: {Xtr_t.shape} in {time.time()-t0:.1f}s")
    sub = np.random.default_rng(1).choice(len(Xtr), min(50_000, len(Xtr)), replace=False)

    results = []
    for name, model in (classifiers(quick) if is_clf else regressors(quick)).items():
        with mlflow.start_run(run_name=name) as run:
            t0 = time.time()
            if name == "XGBoost":
                model.fit(Xtr_t, ytr, eval_set=[(Xva_t, yva)], verbose=False)
            else:
                model.fit(Xtr_t, ytr)
            fit_s = time.time() - t0
            pipe = Pipeline([("prep", pre), ("model", model)])

            mlflow.set_tags({"task": "classification" if is_clf else "regression", "algorithm": name,
                             "dataset": "emi_dataset (400K)", "framework": "sklearn/xgboost"})
            params = {k: v for k, v in model.get_params().items() if isinstance(v, (int, float, str, bool, type(None)))}
            mlflow.log_params(params)
            mlflow.log_params({"n_train": len(Xtr), "n_val": len(Xva), "n_test": len(Xte), "n_features_after_prep": Xtr_t.shape[1]})

            if is_clf:
                m = {}
                for pfx, Xt, y in [("train", Xtr_t[sub], ytr[sub]), ("val", Xva_t, yva), ("test", Xte_t, yte)]:
                    m.update(clf_metrics(y, model.predict(Xt), model.predict_proba(Xt), pfx))
                pred_te = model.predict(Xte_t)
                plot_confusion(yte, pred_te, tmp / "confusion_matrix.png"); mlflow.log_artifact(str(tmp / "confusion_matrix.png"))
                (tmp / "classification_report.txt").write_text(classification_report(yte, pred_te, target_names=CLASS_LABELS, digits=4))
                mlflow.log_artifact(str(tmp / "classification_report.txt"))
            else:
                m = {}
                for pfx, Xt, y in [("train", Xtr_t[sub], ytr[sub]), ("val", Xva_t, yva), ("test", Xte_t, yte)]:
                    m.update(reg_metrics(y, model.predict(Xt), pfx))
                plot_regression(yte, model.predict(Xte_t), tmp / "regression_diagnostics.png")
                mlflow.log_artifact(str(tmp / "regression_diagnostics.png"))
            m["fit_seconds"] = fit_s
            mlflow.log_metrics(m)

            imp = importances(model, names)
            if imp is not None:
                plot_importance(imp, tmp / "feature_importance.png", f"{name} - top features")
                imp.to_csv(tmp / "feature_importance.csv", header=["importance"])
                mlflow.log_artifact(str(tmp / "feature_importance.png")); mlflow.log_artifact(str(tmp / "feature_importance.csv"))
            mlflow.log_artifact(str(REPORTS_DIR / "data_quality.json"), artifact_path="data")

            sample = Xte.head(5)
            sig = infer_signature(sample, pipe.predict(sample))
            info = mlflow.sklearn.log_model(pipe, name="model", signature=sig, input_example=sample,
                                            registered_model_name=registry, serialization_format="cloudpickle")
            version = info.registered_model_version
            client.set_model_version_tag(registry, version, "algorithm", name)
            client.set_model_version_tag(registry, version, "val_" + ("f1_weighted" if is_clf else "rmse"),
                                         f"{m['val_f1_weighted' if is_clf else 'val_rmse']:.4f}")
            results.append({"algorithm": name, "run_id": run.info.run_id, "version": int(version), "pipe": pipe, **m})
            key = "test_accuracy" if is_clf else "test_rmse"
            print(f"[{task}] {name:<20} {key}={m[key]:.4f}  fit={fit_s:.0f}s")

    # ---- model selection on the VALIDATION split (test stays untouched for honest reporting)
    sel = "val_f1_weighted" if is_clf else "val_rmse"
    best = max(results, key=lambda r: r[sel]) if is_clf else min(results, key=lambda r: r[sel])
    client.set_registered_model_alias(registry, "champion", str(best["version"]))
    client.set_model_version_tag(registry, str(best["version"]), "stage", "production")
    client.update_registered_model(registry, description=f"EMIPredict AI {'eligibility classifier' if is_clf else 'max-EMI regressor'}. "
                                                         f"Champion selected on validation {sel}.")
    with mlflow.start_run(run_id=best["run_id"]):
        mlflow.set_tag("selected_for_production", "true")

    save_champion(task, best["pipe"], best["algorithm"])

    table = pd.DataFrame([{k: v for k, v in r.items() if k not in ("pipe",)} for r in results])
    table.to_csv(MODELS_DIR / ("classification_comparison.csv" if is_clf else "regression_comparison.csv"), index=False)
    imp = importances(best["pipe"].named_steps["model"], names)
    if imp is not None:
        imp.head(25).to_csv(MODELS_DIR / ("clf_feature_importance.csv" if is_clf else "reg_feature_importance.csv"), header=["importance"])
    return best, table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--task", choices=["clf", "reg", "both"], default="both")
    args = ap.parse_args()
    import tempfile, pathlib
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="emi_artifacts_"))

    train, val, test = (pd.read_parquet(PROCESSED_DIR / f"{s}.parquet") for s in ("train", "val", "test"))
    if args.quick:
        train, val, test = train.sample(60_000, random_state=1), val.sample(15_000, random_state=1), test.sample(15_000, random_state=1)

    summary = {}
    for task in (["clf", "reg"] if args.task == "both" else [args.task]):
        best, table = run_task(task, train, val, test, args.quick, tmp)
        key = "test_accuracy" if task == "clf" else "test_rmse"
        summary[task] = {"best_algorithm": best["algorithm"], "registry_version": best["version"],
                         **{k: float(v) for k, v in best.items() if k.startswith(("val_", "test_"))}}
        print(f"\n[{task}] CHAMPION: {best['algorithm']}  {key}={best[key]:.4f}\n")
    meta_path = MODELS_DIR / "metadata.json"
    old = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    old.update(summary)
    old["class_labels"] = CLASS_LABELS
    old["targets"] = {"accuracy": TARGET_ACCURACY, "rmse": TARGET_RMSE}
    old["trained_on_rows"] = len(train)
    meta_path.write_text(json.dumps(old, indent=2))


if __name__ == "__main__":
    main()
