"""
train.py
--------
Reproduces the modeling pipeline from Bank_Customer_Churn_Analysis.ipynb and
saves every artifact the Streamlit app (app.py) needs.

Run this ONCE (locally or in CI) before deploying, then commit the generated
`models/` folder to your repo — Streamlit Cloud only runs app.py, it does not
train models on every boot.

Usage:
    python train.py --data European_Bank.csv
"""

import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

RANDOM_STATE = 42


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Same feature engineering used in the notebook (section 3)."""
    drop_cols = [c for c in ["Year", "CustomerId", "Surname"] if c in df.columns]
    model_df = df.drop(columns=drop_cols)

    eps = 1e-6
    model_df["BalanceSalaryRatio"] = model_df["Balance"] / (model_df["EstimatedSalary"] + eps)
    model_df["ProductDensity"] = model_df["NumOfProducts"] / (model_df["Tenure"] + 1)
    model_df["EngagementProductInteraction"] = model_df["IsActiveMember"] * model_df["NumOfProducts"]
    model_df["AgeTenureInteraction"] = model_df["Age"] * model_df["Tenure"]
    model_df["IsZeroBalance"] = (model_df["Balance"] == 0).astype(int)
    model_df["CreditScoreBand"] = pd.cut(
        model_df["CreditScore"], bins=[0, 580, 670, 740, 800, 850], labels=[1, 2, 3, 4, 5]
    ).astype(int)

    model_df = pd.get_dummies(model_df, columns=["Geography", "Gender"], drop_first=True)
    return model_df


def main(data_path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    # 1. Load -----------------------------------------------------------
    df = pd.read_csv(data_path)
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns from {data_path}")

    # 2. Feature engineering ---------------------------------------------
    model_df = engineer_features(df)
    target = "Exited"
    feature_cols = [c for c in model_df.columns if c != target]
    X = model_df[feature_cols]
    y = model_df[target]
    print("Final feature set:", feature_cols)

    # 3. Train / test split ------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    numeric_features = [
        "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
        "EstimatedSalary", "BalanceSalaryRatio", "ProductDensity",
        "EngagementProductInteraction", "AgeTenureInteraction",
    ]
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[numeric_features] = scaler.fit_transform(X_train[numeric_features])
    X_test_scaled[numeric_features] = scaler.transform(X_test[numeric_features])

    # 4. Models -----------------------------------------------------------
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6, min_samples_leaf=30, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=400, max_depth=8, min_samples_leaf=10,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05, random_state=RANDOM_STATE
        ),
        "XGBoost": XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, eval_metric="logloss",
            scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
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
        roc_data[name] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": metrics["ROC-AUC"]}

        print(f"\n{'='*60}\n{name}\n{'='*60}")
        print(classification_report(y_test, y_pred, target_names=["Retained", "Churned"]))

    results_df = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
    best_model_name = results_df.iloc[0]["Model"]
    best_model = fitted_models[best_model_name]
    importance_model_name = "XGBoost" if "XGBoost" in fitted_models else best_model_name
    imp_model = fitted_models[importance_model_name]
    importances = pd.Series(imp_model.feature_importances_, index=X.columns).sort_values(ascending=False)

    print(f"\nBEST MODEL: {best_model_name} | ROC-AUC: {results_df.iloc[0]['ROC-AUC']:.4f}")

    # 5. Save artifacts -----------------------------------------------------
    joblib.dump(best_model, os.path.join(out_dir, "best_model.pkl"))
    joblib.dump(fitted_models["XGBoost"], os.path.join(out_dir, "xgboost_model.pkl"))
    joblib.dump(fitted_models["Logistic Regression"], os.path.join(out_dir, "logreg_model.pkl"))
    joblib.dump(scaler, os.path.join(out_dir, "scaler.pkl"))
    joblib.dump(list(X.columns), os.path.join(out_dir, "feature_columns.pkl"))
    joblib.dump(numeric_features, os.path.join(out_dir, "numeric_features.pkl"))

    meta = {
        "best_model_name": best_model_name,
        "importance_model_name": importance_model_name,
        "feature_columns": list(X.columns),
        "numeric_features": numeric_features,
        "results": results_df.to_dict(orient="records"),
        "roc_data": roc_data,
        "top_features": importances.head(10).to_dict(),
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)

    X_train.assign(Exited=y_train).to_csv(os.path.join(out_dir, "train_reference.csv"), index=False)

    print(f"\nSaved all artifacts to '{out_dir}/'. You're ready to run: streamlit run app.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="European_Bank.csv", help="Path to the raw CSV dataset")
    parser.add_argument("--out", default="models", help="Output directory for saved artifacts")
    args = parser.parse_args()
    main(args.data, args.out)
