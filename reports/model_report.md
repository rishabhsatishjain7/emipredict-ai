# Model Performance Report

All metrics on the **held-out test split**; champions were chosen on the validation split. Models trained on 279,999 rows.

## Classification: EMI eligibility (3 classes)
| algorithm          |   test_accuracy |   test_precision_weighted |   test_recall_weighted |   test_f1_weighted |   test_roc_auc_ovr |   train_accuracy |
|:-------------------|----------------:|--------------------------:|-----------------------:|-------------------:|-------------------:|-----------------:|
| LogisticRegression |          0.8363 |                    0.8389 |                 0.8363 |             0.837  |             0.9464 |           0.8407 |
| DecisionTree       |          0.9789 |                    0.9789 |                 0.9789 |             0.9789 |             0.9899 |           0.9818 |
| RandomForest       |          0.977  |                    0.9771 |                 0.977  |             0.977  |             0.9915 |           0.9826 |
| XGBoost            |          0.9829 |                    0.9829 |                 0.9829 |             0.9829 |             0.9918 |           0.9873 |

**Champion: XGBoost** (registry version 4) - accuracy 0.9829, weighted F1 0.9829, macro ROC-AUC 0.9918.

## Regression: maximum monthly EMI (INR)
| algorithm        |   test_rmse |   test_mae |   test_r2 |   test_mape |   train_rmse |
|:-----------------|------------:|-----------:|----------:|------------:|-------------:|
| LinearRegression |    1195.33  |    802.25  |     0.963 |      65.038 |     1218.73  |
| DecisionTree     |     453.89  |    177.338 |     0.995 |       2.728 |      428.863 |
| RandomForest     |     345.451 |    123.58  |     0.997 |       2.005 |      311.735 |
| XGBoost          |     318.535 |    120.62  |     0.997 |       2.172 |      169.435 |

**Champion: XGBoost** (registry version 4) - RMSE ₹319, MAE ₹121, R² 0.9974, MAPE 2.17%.

## Reading these numbers honestly
- If the dataset is the synthetic one produced by `src/generate_data.py`, its labels come from deterministic affordability rules plus ~1.5% label noise. Tree models recover such rules almost exactly, so the scores are high and **must not be read as expected performance on real lending data**.
- Linear/logistic models trail because the decision boundary is non-linear (ratios and thresholds).
- Compare `train_*` against `test_*` columns as the overfitting check (XGBoost regression has the widest gap).
- Rerun `python src/train.py && python src/make_report.py` on the official dataset to get real numbers.

## Targets from the brief
- Accuracy > 90%: **met** (98.3%)
- RMSE < ₹2,000: **met** (₹319)