# Business Impact Assessment and Recommendations

**Scope note:** figures below that come from the brief (e.g. the 80% manual-processing reduction) are *targets*, not measurements from this build. Portfolio statistics come from the synthetic dataset and illustrate the method; recompute them on real data.

## What the platform changes

| Stakeholder | Capability | Benefit |
|---|---|---|
| Loan officers | Instant three-way rating with probabilities and a suggested affordable amount | Faster, consistent first-pass decisions; time spent on borderline cases |
| Digital lenders | Pre-qualification via the same pipeline | Instant eligibility checks without manual review |
| Risk teams | Maximum safe EMI per applicant; composite risk score; feature importances | Risk-based pricing for the `High_Risk` segment instead of binary decline |
| Compliance | Every model version, metric and data-quality report logged in MLflow | Auditable, reproducible decision process |

## Insights from the analysis (synthetic data)

- Ratios beat raw income: the strongest class separators were composite risk score, disposable income and expense-to-income - all ahead of monthly salary.
- Credit score acts as a gate (0.5% of applicants below 620 are Eligible vs 58.5% above).
- Existing obligations raise rejection sharply (50.6% vs 38.5% Not_Eligible).
- Large-ticket, long-tenure products (vehicle, personal loan) have the highest rejection rates.

## Recommendations

1. **Pilot in shadow mode first:** run the model alongside underwriters and measure agreement and override reasons before automating anything.
2. **Retrain on the institution's own data:** the synthetic labels are rule-derived; real outcomes (defaults, repayment history) will change which features matter.
3. **Price the `High_Risk` band** rather than declining it, using the maximum-EMI estimate to size offers.
4. **Fairness review before launch:** gender, marital status and age are model inputs. Test for disparate impact and consider excluding protected attributes; Indian lending guidance and local regulation apply.
5. **Monitor drift and calibration** (population stability, class mix, MAE of max EMI) and retrain on a schedule; the MLflow registry supports staged promotion via aliases.
6. Keep a **human in the loop** for final decisions; the app is decision support.
