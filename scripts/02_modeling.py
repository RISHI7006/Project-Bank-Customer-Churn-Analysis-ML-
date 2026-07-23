"""
Step 2: Feature Engineering + Model Development + Evaluation
Predictive Modeling and Risk Scoring for Bank Customer Churn
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              roc_auc_score, roc_curve, confusion_matrix, classification_report)
from xgboost import XGBClassifier

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 120
RANDOM_STATE = 42

# ============================================================
# 1. Load data
# ============================================================
df = pd.read_csv("data/European_Bank.csv")
df = df.drop(columns=["Year", "CustomerId", "Surname"])  # non-informative features

# ============================================================
# 2. Feature Engineering
# ============================================================
eps = 1e-6
df["BalanceSalaryRatio"] = df["Balance"] / (df["EstimatedSalary"] + eps)
df["ProductDensity"] = df["NumOfProducts"] / (df["Tenure"] + 1)
df["EngagementProductInteraction"] = df["IsActiveMember"] * df["NumOfProducts"]
df["AgeTenureInteraction"] = df["Age"] * df["Tenure"]
df["IsZeroBalance"] = (df["Balance"] == 0).astype(int)
df["CreditScoreBand"] = pd.cut(df["CreditScore"], bins=[0, 580, 670, 740, 800, 850],
                                labels=[1, 2, 3, 4, 5]).astype(int)

# One-hot encode categoricals
df_model = pd.get_dummies(df, columns=["Geography", "Gender"], drop_first=True)

target = "Exited"
feature_cols = [c for c in df_model.columns if c != target]
X = df_model[feature_cols]
y = df_model[target]

print("Final feature set:", list(X.columns))
print("Shape:", X.shape)

# ============================================================
# 3. Stratified Train-Test Split
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
print(f"\nTrain churn rate: {y_train.mean():.4f}  |  Test churn rate: {y_test.mean():.4f}")

# Scale numeric features (for Logistic Regression)
numeric_features = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
                     "EstimatedSalary", "BalanceSalaryRatio", "ProductDensity",
                     "EngagementProductInteraction", "AgeTenureInteraction"]
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_test_scaled[numeric_features] = scaler.transform(X_test[numeric_features])

# ============================================================
# 4. Model Development
# ============================================================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=30, class_weight="balanced", random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=400, max_depth=8, min_samples_leaf=10,
                                             class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                                                      random_state=RANDOM_STATE),
    "XGBoost": XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
                              colsample_bytree=0.8, eval_metric="logloss",
                              scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
                              random_state=RANDOM_STATE, n_jobs=-1),
}

results = []
fitted_models = {}
roc_data = {}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

for name, model in models.items():
    use_scaled = name == "Logistic Regression"
    Xtr = X_train_scaled if use_scaled else X_train
    Xte = X_test_scaled if use_scaled else X_test

    model.fit(Xtr, y_train)
    fitted_models[name] = model

    y_pred = model.predict(Xte)
    y_proba = model.predict_proba(Xte)[:, 1]

    cv_scores = cross_val_score(model, Xtr, y_train, cv=skf, scoring="roc_auc", n_jobs=-1)

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_proba),
        "CV ROC-AUC (mean)": cv_scores.mean(),
        "CV ROC-AUC (std)": cv_scores.std(),
    }
    results.append(metrics)

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_data[name] = (fpr, tpr, metrics["ROC-AUC"])

    print(f"\n{'='*60}\n{name}\n{'='*60}")
    print(classification_report(y_test, y_pred, target_names=["Retained", "Churned"]))

results_df = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False)
print("\n\n=== MODEL COMPARISON ===")
print(results_df.to_string(index=False))
results_df.to_csv("models/model_comparison.csv", index=False)

# ============================================================
# 5. ROC Curve comparison plot
# ============================================================
fig, ax = plt.subplots(figsize=(7, 6))
colors = ["#264653", "#2A9D8F", "#E9C46A", "#F4A261", "#E76F51"]
for (name, (fpr, tpr, auc)), c in zip(roc_data.items(), colors):
    ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=c, linewidth=2)
ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve Comparison — All Models")
ax.legend(loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig("figures/09_roc_comparison.png")
plt.close()

# ============================================================
# 6. Model comparison bar chart
# ============================================================
fig, ax = plt.subplots(figsize=(9, 5))
metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
results_plot = results_df.set_index("Model")[metrics_to_plot]
results_plot.plot(kind="bar", ax=ax, colormap="viridis")
ax.set_title("Model Performance Comparison")
ax.set_ylabel("Score")
ax.set_ylim(0, 1)
ax.legend(loc="lower right", ncol=2, fontsize=8)
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig("figures/10_model_comparison.png")
plt.close()

# ============================================================
# 7. Select best model (by ROC-AUC) — Confusion Matrix
# ============================================================
best_model_name = results_df.iloc[0]["Model"]
best_model = fitted_models[best_model_name]
use_scaled = best_model_name == "Logistic Regression"
Xte_best = X_test_scaled if use_scaled else X_test
y_pred_best = best_model.predict(Xte_best)

cm = confusion_matrix(y_test, y_pred_best)
fig, ax = plt.subplots(figsize=(5.5, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Retained", "Churned"], yticklabels=["Retained", "Churned"])
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title(f"Confusion Matrix — {best_model_name}")
plt.tight_layout()
plt.savefig("figures/11_confusion_matrix.png")
plt.close()

print(f"\n\nBEST MODEL: {best_model_name}")
print(f"ROC-AUC: {results_df.iloc[0]['ROC-AUC']:.4f}")

# ============================================================
# 8. Feature Importance (best tree-based model + XGBoost)
# ============================================================
importance_model_name = "XGBoost" if "XGBoost" in fitted_models else best_model_name
imp_model = fitted_models[importance_model_name]
importances = pd.Series(imp_model.feature_importances_, index=X.columns).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(8, 6))
importances.head(12).sort_values().plot(kind="barh", ax=ax, color="#2A9D8F")
ax.set_title(f"Feature Importance — {importance_model_name}")
ax.set_xlabel("Importance")
plt.tight_layout()
plt.savefig("figures/12_feature_importance.png")
plt.close()

print("\nTop 10 Feature Importances:")
print(importances.head(10))

# ============================================================
# 9. Save artifacts for Streamlit app
# ============================================================
joblib.dump(best_model, "models/best_model.pkl")
joblib.dump(fitted_models["XGBoost"], "models/xgboost_model.pkl")
joblib.dump(fitted_models["Logistic Regression"], "models/logreg_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(list(X.columns), "models/feature_columns.pkl")
joblib.dump(numeric_features, "models/numeric_features.pkl")

meta = {
    "best_model_name": best_model_name,
    "importance_model_name": importance_model_name,
    "feature_columns": list(X.columns),
    "numeric_features": numeric_features,
    "results": results_df.to_dict(orient="records"),
    "top_features": importances.head(10).to_dict(),
}
with open("models/meta.json", "w") as f:
    json.dump(meta, f, indent=2, default=str)

# Save train data reference (for what-if distributions in app)
X_train.assign(Exited=y_train).to_csv("models/train_reference.csv", index=False)
df.to_csv("data/engineered_dataset.csv", index=False)

print("\nAll models & artifacts saved to models/")
