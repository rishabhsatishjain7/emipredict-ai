# Exploratory Data Analysis Report

Records analysed: **400,000** | Columns: **49**

## 1. Eligibility patterns across lending scenarios
| emi_scenario            |   Eligible |   High_Risk |   Not_Eligible |
|:------------------------|-----------:|------------:|---------------:|
| E-commerce Shopping EMI |       25.6 |        39.7 |           34.6 |
| Education EMI           |       21.9 |        36.1 |           42   |
| Home Appliances EMI     |       25.5 |        39.5 |           35   |
| Personal Loan EMI       |       20.4 |        35.1 |           44.6 |
| Vehicle EMI             |       18.2 |        33.2 |           48.6 |

- Highest rejection rate: **Vehicle EMI** (48.6% Not_Eligible) - large tickets stretch affordability.
- Highest approval rate: **E-commerce Shopping EMI** (25.6% Eligible) - small tickets, short tenure.

## 2. Financial drivers of approval
| emi_eligibility   |   credit_score |   monthly_salary |   dti_after_loan |   risk_score |
|:------------------|---------------:|-----------------:|-----------------:|-------------:|
| Eligible          |            657 |            36300 |             0.11 |         0.27 |
| High_Risk         |            583 |            33700 |             0.15 |         0.35 |
| Not_Eligible      |            579 |            26500 |             0.27 |         0.46 |

- Eligible applicants have markedly higher credit scores and lower post-loan debt-to-income.
- The composite `risk_score` rises monotonically from Eligible to Not_Eligible, validating the engineered feature.

## 3. Correlation with maximum safe EMI
|                   |   abs_corr |
|:------------------|-----------:|
| monthly_salary    |      0.718 |
| expense_to_income |      0.696 |
| risk_score        |      0.675 |
| bank_balance      |      0.416 |
| emergency_fund    |      0.412 |

## 4. Demographic and risk-factor patterns
**% Eligible by employment type**
| employment_type   |   % eligible |
|:------------------|-------------:|
| Government        |         33.3 |
| Private           |         20.3 |
| Self-employed     |         19.5 |

**% Eligible by education**
| education     |   % eligible |
|:--------------|-------------:|
| Graduate      |         21.1 |
| High School   |         17   |
| Post Graduate |         24.4 |
| Professional  |         28.5 |

- Stable income (government, longer tenure) and higher education correlate with better outcomes, largely through higher salaries and credit scores.

## 5. Statistical summary
|       |   monthly_salary |   credit_score |   requested_amount |   requested_tenure |   max_monthly_emi |
|:------|-----------------:|---------------:|-------------------:|-------------------:|------------------:|
| count |         397629   |       396639   |     400000         |           400000   |          400000   |
| mean  |          33536.9 |          600.1 |     127236         |               29.1 |            6962.2 |
| std   |          13041.2 |           63.6 |     117659         |               18.1 |            6200.3 |
| min   |          15000   |          300   |      10000         |                3   |             500   |
| 25%   |          24100   |          557   |      48000         |               15   |            1660   |
| 50%   |          31200   |          599   |      88000         |               25   |            5800   |
| 75%   |          40300   |          643   |     166000         |               40   |           10270   |
| max   |         183100   |          850   |          1.329e+06 |               84   |           50000   |

## 6. Business insights (computed from the data)
1. **Strongest class separators (ANOVA F):** `risk_score`, `disposable_income`, `expense_to_income` - ahead of raw `monthly_salary` (rank 8 of 8). Ratios beat absolute income.
2. **Credit score gates approval:** only 0.5% of applicants under 620 are Eligible versus 58.5% at 620+.
3. **Existing obligations matter:** 50.6% of applicants with active loans are Not_Eligible versus 38.5% without - debt-to-income should be a hard underwriting rule.
4. **Ticket size and tenure:** large vehicle/personal loans have the highest rejection rates; longer tenures lower the EMI but raise total interest, so `High_Risk` cases should be priced up rather than declined.
