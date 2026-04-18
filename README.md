# SBA Loan Default Prediction

## Overview
This project focuses on predicting whether a small business loan will be repaid (Paid in Full) or default (Charged Off) using historical data from the SBA 7(a) loan program.

The goal is to understand key drivers of loan default risk and build a predictive model that can support better lending decisions.

---

## Problem Statement
Given loan, borrower, and economic characteristics, predict whether a loan will default.

This is a binary classification problem:
- 0 → Paid in Full (PIF)
- 1 → Charged Off (Default)

---

## Dataset
- Source: SBA 7(a) loan program data
- Size: ~1.9 million records
- Features include:
  - Loan amount, term, interest rate
  - Business type and industry (NAICS)
  - Borrower state
  - Lending bank
  - Jobs supported

---

## Approach

### Data Preprocessing
- Handled missing values using median imputation
- Applied log transformations for skewed variables
- Performed outlier capping for extreme values
- Encoded categorical variables

### Feature Engineering
- Created time-based features (approval year)
- Aggregated industry categories (NAICS2)
- Generated dummy variables for categorical fields

### Modeling
- Used Logistic Regression (GLM - Binomial)
- Applied forward stepwise selection using BIC
- Selected statistically significant predictors

### Evaluation
- Metrics used:
  - ROC-AUC ≈ 0.80
  - Accuracy ≈ 78%
- Performed both in-time and out-of-time validation

---

## Key Findings
- Loan size, borrower state, and lender type significantly impact default risk
- Model performs well in ranking risk (AUC ~0.8)
- Performance varies across time due to concept drift
- Fixed classification thresholds do not generalize well across economic conditions

---

## Challenges
- Large dataset required efficient processing
- Multicollinearity among financial variables
- Concept drift across different economic periods
- Sensitivity of logistic regression to outliers

---

## Future Improvements
- Use advanced models (XGBoost, Random Forest)
- Explore time-aware models for better generalization
- Implement dynamic thresholding
- Deploy as an API for real-time prediction

---

## Tech Stack
- Python
- Pandas, NumPy
- Statsmodels
- Scikit-learn
- Matplotlib / Seaborn

---

## Project Structure

- SBA_loan_default_prediction_pipeline.ipynb
- data/
- README.md
