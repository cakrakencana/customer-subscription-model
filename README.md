# Bank Marketing Campaign — Machine Learning Classification

## Overview
This project builds a machine learning model to predict customer subscription to a term deposit product. The primary objective is to improve outbound marketing efficiency by identifying high-propensity customers and reducing unnecessary call costs. The model applies a precision-focused strategy to minimize false positives and optimize operational spending.

---

## Business Problem
Traditional outbound campaigns rely on contacting broad customer lists, resulting in high operational cost and low conversion efficiency. Calling uninterested customers increases expenses and adds workload to call center teams.

The goal of this project is to use machine learning to:
- Predict which customers are likely to subscribe.
- Prioritize high-probability leads.
- Reduce outbound call volume and associated costs.
- Improve conversion efficiency and overall campaign ROI.

---

## Dataset Description
The dataset contains 7,813 customer records with 11 features, covering:
- Demographics (age, job)
- Financial attributes (balance, loan, housing)
- Contact and campaign details (contact type, campaign count, pdays, poutcome)
- Target variable: deposit (yes/no)

The dataset has no missing values and includes meaningful behavioral signals relevant for classification.

---

## Exploratory Data Analysis
Key findings:
- Customers with successful past campaign outcomes show significantly higher subscription rates.
- Cellular contact performs better than telephone or unknown contact types.
- Higher account balance is associated with higher likelihood of subscription.
- Excessive contact attempts reduce conversion, indicating customer fatigue.
- Certain months exhibit stronger subscription patterns, suggesting seasonal effects.

These insights informed the feature engineering and modeling strategy.

---

## Data Preparation and Feature Engineering
Key steps include:
- Converting the target variable into a binary numeric flag.
- Engineering a new variable, `previous_contacted`, based on `pdays`.
- One-hot encoding categorical features.
- Interpreting `pdays = -1` as “no prior contact.”
- Producing a fully numeric, model-ready dataset.

---

## Modeling and Evaluation
Four algorithms were benchmarked under consistent conditions:
- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost

Precision was selected as the primary metric due to its direct impact on call cost reduction.

Gradient Boosting achieved the highest precision, balanced performance across metrics, and strong generalizability. It was selected as the final model for tuning.

### Hyperparameter Tuning
GridSearchCV identified the optimal configuration:
- learning_rate: 0.1
- max_depth: 2
- n_estimators: 100
- subsample: 1.0

The tuned model demonstrated improved stability and precision.

---

## Threshold Optimization
The default classification threshold of 0.50 generates too many false positives for the business objective. A precision–recall analysis was conducted to identify an operating threshold aligned with cost reduction goals.

A threshold of approximately 0.82 was selected:
- Precision: ~95%
- Recall: ~19%
- False positives: 7 (very low)

This threshold intentionally prioritizes precision to ensure that outbound calls target only customers with strong intent.

---

## Business Impact Analysis
Comparison between baseline (calling everyone) and model-driven targeting:

| Metric | Before ML | After ML |
|--------|-----------|----------|
| Customers Called | 1,563 | 152 |
| Total Cost (Rp) | 3,126,000 | 304,000 |
| Conversions Captured | 747 | 145 |
| Conversion Rate | 48% | 95% |

Results are based on the test set (20% of the dataset).

Scaling to the full campaign (7,813 customers):
- Estimated cost savings per campaign: Rp 14,110,000
- Estimated annual savings (monthly campaigns): approximately Rp 168,000,000

The model significantly reduces operational expenses while improving conversion efficiency.

---

## Reproducibility and Pipeline
The project includes:
- A modular training and inference pipeline.
- Clear separation between preprocessing, modeling, and prediction logic.
