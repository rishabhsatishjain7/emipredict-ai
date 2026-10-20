"""Regenerate reports/model_report.md from the saved comparison tables and metadata."""
import json

import pandas as pd

from config import MODELS_DIR, REPORTS_DIR


def main():
    c = pd.read_csv(MODELS_DIR / "classification_comparison.csv")
    r = pd.read_csv(MODELS_DIR / "regression_comparison.csv")
    meta = json.loads((MODELS_DIR / "metadata.json").read_text())
    cm, rm = meta["clf"], meta["reg"]
    cc = ["algorithm", "test_accuracy", "test_precision_weighted", "test_recall_weighted", "test_f1_weighted",
          "test_roc_auc_ovr", "train_accuracy"]
    rc = ["algorithm", "test_rmse", "test_mae", "test_r2", "test_mape", "train_rmse"]
    md = [
        "# Model Performance Report", "",
        f"All metrics on the **held-out test split**; champions were chosen on the validation split. "
        f"Models trained on {meta.get('trained_on_rows', '?'):,} rows.", "",
        "## Classification: EMI eligibility (3 classes)", c[cc].round(4).to_markdown(index=False), "",
        f"**Champion: {cm['best_algorithm']}** (registry version {cm['registry_version']}) - accuracy {cm['test_accuracy']:.4f}, "
        f"weighted F1 {cm['test_f1_weighted']:.4f}, macro ROC-AUC {cm['test_roc_auc_ovr']:.4f}.", "",
        "## Regression: maximum monthly EMI (INR)", r[rc].round(3).to_markdown(index=False), "",
        f"**Champion: {rm['best_algorithm']}** (registry version {rm['registry_version']}) - RMSE ₹{rm['test_rmse']:.0f}, "
        f"MAE ₹{rm['test_mae']:.0f}, R² {rm['test_r2']:.4f}, MAPE {rm['test_mape']:.2f}%.", "",
        "## Reading these numbers honestly",
        "- If the dataset is the synthetic one produced by `src/generate_data.py`, its labels come from deterministic "
        "affordability rules plus ~1.5% label noise. Tree models recover such rules almost exactly, so the scores are high and "
        "**must not be read as expected performance on real lending data**.",
        "- Linear/logistic models trail because the decision boundary is non-linear (ratios and thresholds).",
        "- Compare `train_*` against `test_*` columns as the overfitting check (XGBoost regression has the widest gap).",
        "- Rerun `python src/train.py && python src/make_report.py` on the official dataset to get real numbers.", "",
        "## Targets from the brief",
        f"- Accuracy > 90%: {'**met**' if cm['test_accuracy'] > meta['targets']['accuracy'] else '**not met**'} ({cm['test_accuracy']:.1%})",
        f"- RMSE < ₹{meta['targets']['rmse']:,}: {'**met**' if rm['test_rmse'] < meta['targets']['rmse'] else '**not met**'} (₹{rm['test_rmse']:.0f})",
    ]
    (REPORTS_DIR / "model_report.md").write_text("\n".join(md), encoding="utf-8")
    print("wrote", REPORTS_DIR / "model_report.md")


if __name__ == "__main__":
    main()
