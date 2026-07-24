"""
Bank Customer Churn Risk Intelligence - Streamlit App
------------------------------------------------------
Deploy on Streamlit Community Cloud:
  1. Push this file + requirements.txt + the `models/` folder to a GitHub repo.
  2. On https://share.streamlit.io, point to app.py in that repo.

Folder layout expected next to this file:
  app.py
  requirements.txt
  models/
    best_model.pkl
    xgboost_model.pkl
    feature_columns.pkl
    meta.json
    train_reference.csv
"""

import os
import json

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import shap
import matplotlib.pyplot as plt
import streamlit as st

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Bank Churn Risk Intelligence",
    page_icon=":bank:",
    layout="wide",
)

MODELS_DIR = "models"


# --------------------------------------------------------------------------
# Artifact loading (cached)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model artifacts...")
def load_artifacts():
    missing = [
        f for f in [
            "best_model.pkl", "xgboost_model.pkl", "feature_columns.pkl",
            "meta.json", "train_reference.csv",
        ]
        if not os.path.exists(os.path.join(MODELS_DIR, f))
    ]
    if missing:
        st.error(
            "Missing model artifact(s): " + ", ".join(missing) +
            f". Make sure the `{MODELS_DIR}/` folder is uploaded alongside app.py."
        )
        st.stop()

    best_model = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
    xgb_model = joblib.load(os.path.join(MODELS_DIR, "xgboost_model.pkl"))
    feature_columns = joblib.load(os.path.join(MODELS_DIR, "feature_columns.pkl"))
    with open(os.path.join(MODELS_DIR, "meta.json")) as f:
        meta = json.load(f)
    train_ref = pd.read_csv(os.path.join(MODELS_DIR, "train_reference.csv"))
    return best_model, xgb_model, feature_columns, meta, train_ref


# NOTE: the leading underscore on `_model` tells Streamlit's hasher to SKIP
# trying to hash the model object (it can't hash an XGBClassifier). This is
# the fix for: UnhashableParamError: Cannot hash argument 'model'.
@st.cache_resource(show_spinner="Building SHAP explainer...")
def get_explainer(_model):
    return shap.TreeExplainer(_model)


best_model, xgb_model, FEATURE_COLUMNS, META, train_ref = load_artifacts()
explainer = get_explainer(xgb_model)

BEST_MODEL_NAME = META.get("best_model_name", "Best Model")


# --------------------------------------------------------------------------
# Feature engineering (must mirror the notebook exactly)
# --------------------------------------------------------------------------
def engineer_features(raw: dict) -> pd.DataFrame:
    eps = 1e-6
    balance, salary = raw["Balance"], raw["EstimatedSalary"]
    tenure, age = raw["Tenure"], raw["Age"]
    num_products, is_active, cs = raw["NumOfProducts"], raw["IsActiveMember"], raw["CreditScore"]

    row = {
        "CreditScore": cs,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": raw["HasCrCard"],
        "IsActiveMember": is_active,
        "EstimatedSalary": salary,
        "BalanceSalaryRatio": balance / (salary + eps),
        "ProductDensity": num_products / (tenure + 1),
        "EngagementProductInteraction": is_active * num_products,
        "AgeTenureInteraction": age * tenure,
        "IsZeroBalance": int(balance == 0),
        "CreditScoreBand": int(
            pd.cut([cs], bins=[0, 580, 670, 740, 800, 850], labels=[1, 2, 3, 4, 5])[0]
        ),
        "Geography_Germany": int(raw["Geography"] == "Germany"),
        "Geography_Spain": int(raw["Geography"] == "Spain"),
        "Gender_Male": int(raw["Gender"] == "Male"),
    }
    return pd.DataFrame([row])[FEATURE_COLUMNS]


def risk_tier(proba: float):
    if proba >= 0.5:
        return "High Risk", "#C1121F"
    elif proba >= 0.25:
        return "Medium Risk", "#E09F3E"
    return "Low Risk", "#2A9D8F"


# --------------------------------------------------------------------------
# Sidebar - customer inputs (shared across tabs)
# --------------------------------------------------------------------------
st.sidebar.header("Customer Profile")

credit_score = st.sidebar.slider("Credit Score", 350, 850, 650)
age = st.sidebar.slider("Age", 18, 92, 38)
tenure = st.sidebar.slider("Tenure (years)", 0, 10, 5)
balance = st.sidebar.number_input("Balance (EUR )", 0.0, 300000.0, 75000.0, step=1000.0)
salary = st.sidebar.number_input("Estimated Salary (EUR )", 0.0, 250000.0, 100000.0, step=1000.0)
num_products = st.sidebar.selectbox("Number of Products", [1, 2, 3, 4])
geography = st.sidebar.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
has_card = st.sidebar.radio("Has Credit Card?", ["Yes", "No"], horizontal=True)
is_active = st.sidebar.radio("Active Member?", ["Yes", "No"], horizontal=True)

raw = {
    "CreditScore": credit_score,
    "Age": age,
    "Tenure": tenure,
    "Balance": balance,
    "NumOfProducts": num_products,
    "HasCrCard": 1 if has_card == "Yes" else 0,
    "IsActiveMember": 1 if is_active == "Yes" else 0,
    "EstimatedSalary": salary,
    "Geography": geography,
    "Gender": gender,
}

X_input = engineer_features(raw)
proba = float(best_model.predict_proba(X_input)[0, 1])
tier, color = risk_tier(proba)


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("Bank Customer Churn Risk Intelligence")
st.caption(f"Scoring model: **{BEST_MODEL_NAME}**  |  Explanation model: **XGBoost (SHAP)**")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Risk Calculator", "Why This Score (SHAP)", "Feature Importance",
     "Model Leaderboard", "Business Insights"]
)

# --------------------------------------------------------------------------
# Tab 1 - Risk Calculator
# --------------------------------------------------------------------------
with tab1:
    col1, col2 = st.columns([1, 1.4])

    with col1:
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=proba * 100,
                number={"suffix": "%"},
                title={"text": "Churn Probability"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": color},
                    "steps": [
                        {"range": [0, 25], "color": "#e8f6f3"},
                        {"range": [25, 50], "color": "#fdf2e3"},
                        {"range": [50, 100], "color": "#fbe4e4"},
                    ],
                },
            )
        )
        fig.update_layout(height=320, margin=dict(t=50, b=10, l=20, r=20))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            f"### Risk Tier: <span style='color:{color}'>{tier}</span>",
            unsafe_allow_html=True,
        )
        st.metric("Churn Probability", f"{proba*100:.1f}%")

    with col2:
        st.subheader("Recommended Action")
        if tier == "High Risk":
            st.error(
                "[HIGH] Priority retention outreach recommended. Consider a personal "
                "call from a relationship manager, a product-fit review, and a "
                "loyalty offer within the next 7 days."
            )
        elif tier == "Medium Risk":
            st.warning(
                "[MED] Monitor closely. Consider an engagement nudge (app/email "
                "campaign) and check whether the customer is over-holding products."
            )
        else:
            st.success(
                "[LOW] Low churn risk. Standard engagement cadence is sufficient."
            )

        st.subheader("Customer Snapshot")
        snap = pd.DataFrame(
            {
                "Field": ["Credit Score", "Age", "Tenure", "Balance", "Salary",
                          "Products", "Geography", "Gender", "Has Card", "Active"],
                "Value": [credit_score, age, f"{tenure} yrs", f"EUR {balance:,.0f}",
                          f"EUR {salary:,.0f}", num_products, geography, gender,
                          has_card, is_active],
            }
        )
        st.dataframe(snap, hide_index=True, use_container_width=True)

# --------------------------------------------------------------------------
# Tab 2 - SHAP explanation for this specific customer
# --------------------------------------------------------------------------
with tab2:
    st.subheader("What's driving this customer's score?")
    st.caption(
        "SHAP values show how much each feature pushed this customer's predicted "
        "risk up (red) or down (blue), relative to the average customer."
    )

    shap_values = explainer(X_input)

    fig_waterfall, ax = plt.subplots(figsize=(9, 6))
    shap.plots.waterfall(shap_values[0], show=False, max_display=12)
    plt.tight_layout()
    st.pyplot(fig_waterfall, use_container_width=True)
    plt.close(fig_waterfall)

# --------------------------------------------------------------------------
# Tab 3 - Global feature importance
# --------------------------------------------------------------------------
with tab3:
    st.subheader("Global Feature Importance")
    st.caption("Averaged across the training population (XGBoost model).")

    top_features = META.get("top_features", {})
    if top_features:
        imp_df = (
            pd.DataFrame(list(top_features.items()), columns=["Feature", "Importance"])
            .sort_values("Importance", ascending=True)
        )
        fig_imp = px.bar(
            imp_df, x="Importance", y="Feature", orientation="h",
            color="Importance", color_continuous_scale="Teal",
        )
        fig_imp.update_layout(height=450, coloraxis_showscale=False)
        st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.info("No `top_features` found in meta.json.")

# --------------------------------------------------------------------------
# Tab 4 - Model leaderboard
# --------------------------------------------------------------------------
with tab4:
    st.subheader("Model Comparison")
    results = META.get("results", [])
    if results:
        results_df = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False)
        st.dataframe(
            results_df.style.format(
                {c: "{:.4f}" for c in results_df.columns if c != "Model"}
            ),
            hide_index=True,
            use_container_width=True,
        )
        fig_bar = px.bar(
            results_df, x="Model", y="ROC-AUC", color="ROC-AUC",
            color_continuous_scale="Blues", text_auto=".3f",
        )
        fig_bar.update_layout(height=400, coloraxis_showscale=False)
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("No `results` found in meta.json.")

# --------------------------------------------------------------------------
# Tab 5 - Business insights (static summary from the notebook)
# --------------------------------------------------------------------------
with tab5:
    st.subheader("Key Insights")
    st.markdown(
        """
- **Product over-holding** is the single largest churn driver: customers with 3+ products churn at 83-100%, vs. 7.6% at 2 products.
- **Engagement** matters more than demographics: inactive members churn at roughly 2x the rate of active members.
- **Germany** is a structurally higher-risk market (~32% churn vs. ~16% in France/Spain).
- **Age** shows a steady, positive association with churn risk.
- **Tenure** alone is a weak predictor - relationship length doesn't guarantee loyalty without engagement.
        """
    )
    st.subheader("Business Recommendations")
    st.markdown(
        """
1. Trigger priority retention outreach for customers scoring above the high-risk threshold.
2. Investigate the 3-4 product segment as a product-fit / cross-sell process issue.
3. Launch a Germany-specific retention and competitive review.
4. Prioritize engagement-activation campaigns for inactive members.
5. Use scenario simulation to quantify expected risk reduction of interventions before committing budget.
        """
    )
