# Bank Customer Churn Risk Intelligence — Streamlit App

## Setup

```bash
cd app
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

The app will open at http://localhost:8501

## Contents

- `app.py` — main Streamlit application (5 modules: Risk Calculator, Probability
  Distribution, Feature Importance Dashboard, What-If Simulator, Model Performance)
- `models/` — trained models (Gradient Boosting, XGBoost, Logistic Regression),
  scaler, feature metadata, SHAP importances, and a reference training set used
  for portfolio-level visualizations
- `data/` — engineered dataset snapshot

## Notes

- The production model is **Gradient Boosting** (ROC-AUC 0.869 on held-out test data).
- SHAP explanations use the XGBoost model (TreeExplainer) for per-customer
  interpretability in the Risk Calculator.
- All customer-level predictions are computed live from the raw inputs using the
  same feature-engineering logic as the training pipeline (see `engineer_features()`
  in `app.py`).
