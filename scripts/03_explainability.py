"""
Step 3: Model Explainability — SHAP values & Partial Dependence Plots
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import shap
from sklearn.inspection import PartialDependenceDisplay

plt.rcParams["figure.dpi"] = 120

# Load artifacts
xgb_model = joblib.load("models/xgboost_model.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")
train_ref = pd.read_csv("models/train_reference.csv")
X_train = train_ref[feature_columns]

# Sample for SHAP speed
X_sample = X_train.sample(n=min(1500, len(X_train)), random_state=42)

# ============================================================
# SHAP summary (beeswarm)
# ============================================================
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer(X_sample)

plt.figure(figsize=(9, 7))
shap.summary_plot(shap_values, X_sample, show=False, plot_size=None)
plt.title("SHAP Summary Plot — XGBoost Churn Model")
plt.tight_layout()
plt.savefig("figures/13_shap_summary.png", bbox_inches="tight")
plt.close()

# ============================================================
# SHAP bar plot (mean |SHAP|)
# ============================================================
plt.figure(figsize=(8, 6))
shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False, plot_size=None)
plt.title("Mean Absolute SHAP Value by Feature")
plt.tight_layout()
plt.savefig("figures/14_shap_importance_bar.png", bbox_inches="tight")
plt.close()

# ============================================================
# Partial Dependence Plots for key drivers
# ============================================================
key_features = ["NumOfProducts", "Age", "IsActiveMember", "Geography_Germany"]
fig, ax = plt.subplots(figsize=(11, 8))
PartialDependenceDisplay.from_estimator(xgb_model, X_train, key_features, ax=ax, n_cols=2)
fig.suptitle("Partial Dependence Plots — Key Churn Drivers", y=1.02)
plt.tight_layout()
plt.savefig("figures/15_partial_dependence.png", bbox_inches="tight")
plt.close()

# Save mean abs shap values to json for app / report
mean_abs_shap = pd.Series(np.abs(shap_values.values).mean(axis=0), index=feature_columns).sort_values(ascending=False)
mean_abs_shap.to_csv("models/shap_importance.csv", header=["mean_abs_shap"])
print(mean_abs_shap.head(10))

print("\nSHAP + PDP analysis complete. Figures saved to figures/")
