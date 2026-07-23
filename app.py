"""
app.py — Bank Churn Risk Intelligence
Streamlit Community Cloud entry point.

Expects a `models/` folder (produced by train.py) sitting next to this file,
containing: best_model.pkl, xgboost_model.pkl, logreg_model.pkl, scaler.pkl,
feature_columns.pkl, numeric_features.pkl, meta.json, train_reference.csv
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st

st.set_page_config(page_title="Bank Churn Risk Intelligence", page_icon="🏦", layout="wide")

MODELS_DIR = "models"


# ----------------------------------------------------------------------
# Cached loaders
# ----------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    required = [
        "best_model.pkl", "xgboost_model.pkl", "feature_columns.pkl",
        "numeric_features.pkl", "meta.json", "train_reference.csv",
    ]
    missing = [f for f in required if not os.path.exists(os.path.join(MODELS_DIR, f))]
    if missing:
        st.error(
            "Missing model artifacts: " + ", ".join(missing) +
            ".\n\nRun `python train.py --data European_Bank.csv` locally first, "
            "then commit the `models/` folder to your repo."
        )
        st.stop()

    best_model = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
    xgb_model = joblib.load(os.path.join(MODELS_DIR, "xgboost_model.pkl"))
    feature_columns = joblib.load(os.path.join(MODELS_DIR, "feature_columns.pkl"))
    with open(os.path.join(MODELS_DIR, "meta.json")) as f:
        meta = json.load(f)
    train_ref = pd.read_csv(os.path.join(MODELS_DIR, "train_reference.csv"))
    return best_model, xgb_model, feature_columns, meta, train_ref


@st.cache_resource
def get_explainer(_model):
    return shap.TreeExplainer(_model)


best_model, xgb_model, FEATURE_COLUMNS, META, train_ref = load_artifacts()
explainer = get_explainer(xgb_model)


# ----------------------------------------------------------------------
# Feature engineering — must mirror train.py exactly
# ----------------------------------------------------------------------
def engineer_features(raw: dict) -> pd.DataFrame:
    eps = 1e-6
    balance, salary, tenure, age = raw["Balance"], raw["EstimatedSalary"], raw["Tenure"], raw["Age"]
    num_products, is_active, cs = raw["NumOfProducts"], raw["IsActiveMember"], raw["CreditScore"]

    row = {
        "CreditScore": cs, "Age": age, "Tenure": tenure, "Balance": balance,
        "NumOfProducts": num_products, "HasCrCard": raw["HasCrCard"], "IsActiveMember": is_active,
        "EstimatedSalary": salary, "BalanceSalaryRatio": balance / (salary + eps),
        "ProductDensity": num_products / (tenure + 1),
        "EngagementProductInteraction": is_active * num_products,
        "AgeTenureInteraction": age * tenure, "IsZeroBalance": int(balance == 0),
        "CreditScoreBand": int(pd.cut([cs], bins=[0, 580, 670, 740, 800, 850], labels=[1, 2, 3, 4, 5])[0]),
        "Geography_Germany": int(raw["Geography"] == "Germany"),
        "Geography_Spain": int(raw["Geography"] == "Spain"),
        "Gender_Male": int(raw["Gender"] == "Male"),
    }
    return pd.DataFrame([row]).reindex(columns=FEATURE_COLUMNS, fill_value=0)


def engineer_features_batch(df: pd.DataFrame) -> pd.DataFrame:
    eps = 1e-6
    out = df.copy()
    out["BalanceSalaryRatio"] = out["Balance"] / (out["EstimatedSalary"] + eps)
    out["ProductDensity"] = out["NumOfProducts"] / (out["Tenure"] + 1)
    out["EngagementProductInteraction"] = out["IsActiveMember"] * out["NumOfProducts"]
    out["AgeTenureInteraction"] = out["Age"] * out["Tenure"]
    out["IsZeroBalance"] = (out["Balance"] == 0).astype(int)
    out["CreditScoreBand"] = pd.cut(
        out["CreditScore"], bins=[0, 580, 670, 740, 800, 850], labels=[1, 2, 3, 4, 5]
    ).astype(int)
    out = pd.get_dummies(out, columns=["Geography", "Gender"], drop_first=True)
    return out.reindex(columns=FEATURE_COLUMNS, fill_value=0)


def risk_tier(proba: float):
    if proba >= 0.5:
        return "High Risk", "#C1121F"
    elif proba >= 0.25:
        return "Medium Risk", "#E09F3E"
    return "Low Risk", "#2A9D8F"


def gauge(proba: float, color: str):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=proba * 100,
        number={"suffix": "%"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 25], "color": "#e8f5f0"},
                {"range": [25, 50], "color": "#fbe9d6"},
                {"range": [50, 100], "color": "#fbe0e0"},
            ],
        },
    ))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=30, b=10))
    return fig


# ----------------------------------------------------------------------
# Sidebar navigation
# ----------------------------------------------------------------------
st.sidebar.title("🏦 Churn Intelligence")
page = st.sidebar.radio(
    "Navigate",
    ["Single Customer", "Batch Scoring", "Model Performance", "Key Insights"],
)
st.sidebar.markdown("---")
st.sidebar.caption(f"Best model: **{META['best_model_name']}**")
st.sidebar.caption(f"Test ROC-AUC: **{META['results'][0]['ROC-AUC']:.3f}**")


# ----------------------------------------------------------------------
# Page: Single Customer
# ----------------------------------------------------------------------
if page == "Single Customer":
    st.title("Customer Churn Risk Calculator")
    st.caption("Adjust the profile to score an individual customer in real time.")

    c1, c2, c3 = st.columns(3)
    with c1:
        credit_score = st.slider("Credit Score", 350, 850, 650)
        age = st.slider("Age", 18, 92, 38)
        tenure = st.slider("Tenure (years)", 0, 10, 5)
    with c2:
        balance = st.number_input("Balance (€)", 0.0, 300000.0, 75000.0, step=1000.0)
        salary = st.number_input("Estimated Salary (€)", 0.0, 250000.0, 100000.0, step=1000.0)
        num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
    with c3:
        geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        has_card = st.radio("Has Credit Card?", ["Yes", "No"], horizontal=True)
        is_active = st.radio("Active Member?", ["Yes", "No"], horizontal=True)

    raw = {
        "CreditScore": credit_score, "Age": age, "Tenure": tenure, "Balance": balance,
        "NumOfProducts": num_products, "HasCrCard": 1 if has_card == "Yes" else 0,
        "IsActiveMember": 1 if is_active == "Yes" else 0, "EstimatedSalary": salary,
        "Geography": geography, "Gender": gender,
    }
    X_input = engineer_features(raw)
    proba = float(best_model.predict_proba(X_input)[0, 1])
    tier, color = risk_tier(proba)

    st.markdown("---")
    g1, g2 = st.columns([1, 1.3])
    with g1:
        st.plotly_chart(gauge(proba, color), use_container_width=True)
        st.markdown(
            f"<h3 style='text-align:center;color:{color};'>{tier}</h3>",
            unsafe_allow_html=True,
        )

    with g2:
        st.subheader("What's driving this score?")
        xgb_input = X_input.reindex(columns=FEATURE_COLUMNS, fill_value=0)
        shap_vals = explainer(xgb_input)
        contrib = pd.Series(shap_vals.values[0], index=FEATURE_COLUMNS).sort_values(key=np.abs, ascending=True).tail(8)
        fig = px.bar(
            contrib, orientation="h",
            color=contrib.values > 0,
            color_discrete_map={True: "#C1121F", False: "#2A9D8F"},
            labels={"value": "Impact on churn probability", "index": ""},
        )
        fig.update_layout(showlegend=False, height=320, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Red bars push risk up, green bars push risk down.")

# ----------------------------------------------------------------------
# Page: Batch Scoring
# ----------------------------------------------------------------------
elif page == "Batch Scoring":
    st.title("Batch Churn Scoring")
    st.caption(
        "Upload a CSV with columns: CreditScore, Age, Tenure, Balance, NumOfProducts, "
        "HasCrCard, IsActiveMember, EstimatedSalary, Geography, Gender"
    )

    uploaded = st.file_uploader("Upload customer CSV", type="csv")
    if uploaded is not None:
        batch_df = pd.read_csv(uploaded)
        required_cols = {
            "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
            "HasCrCard", "IsActiveMember", "EstimatedSalary", "Geography", "Gender",
        }
        missing = required_cols - set(batch_df.columns)
        if missing:
            st.error(f"Missing required columns: {sorted(missing)}")
        else:
            X_batch = engineer_features_batch(batch_df)
            probs = best_model.predict_proba(X_batch)[:, 1]
            batch_df["ChurnProbability"] = probs
            batch_df["RiskTier"] = pd.cut(
                probs, bins=[-0.01, 0.25, 0.5, 1.0], labels=["Low Risk", "Medium Risk", "High Risk"]
            )

            st.success(f"Scored {len(batch_df)} customers.")
            m1, m2, m3 = st.columns(3)
            m1.metric("High Risk", int((batch_df["RiskTier"] == "High Risk").sum()))
            m2.metric("Medium Risk", int((batch_df["RiskTier"] == "Medium Risk").sum()))
            m3.metric("Low Risk", int((batch_df["RiskTier"] == "Low Risk").sum()))

            st.dataframe(
                batch_df.sort_values("ChurnProbability", ascending=False),
                use_container_width=True,
            )
            st.download_button(
                "Download scored CSV",
                batch_df.to_csv(index=False).encode("utf-8"),
                file_name="scored_customers.csv",
                mime="text/csv",
            )

# ----------------------------------------------------------------------
# Page: Model Performance
# ----------------------------------------------------------------------
elif page == "Model Performance":
    st.title("Model Performance Comparison")

    results_df = pd.DataFrame(META["results"])
    st.dataframe(results_df.set_index("Model"), use_container_width=True)

    metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    long_df = results_df.melt(id_vars="Model", value_vars=metrics_to_plot, var_name="Metric", value_name="Score")
    fig = px.bar(long_df, x="Model", y="Score", color="Metric", barmode="group")
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)

    if "roc_data" in META:
        st.subheader("ROC Curves")
        fig2 = go.Figure()
        for name, d in META["roc_data"].items():
            fig2.add_trace(go.Scatter(x=d["fpr"], y=d["tpr"], mode="lines", name=f"{name} (AUC={d['auc']:.3f})"))
        fig2.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"), name="Random"))
        fig2.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", height=450)
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader(f"Feature Importance — {META['importance_model_name']}")
    top_feats = pd.Series(META["top_features"]).sort_values()
    fig3 = px.bar(top_feats, orientation="h", labels={"value": "Importance", "index": ""})
    fig3.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig3, use_container_width=True)

# ----------------------------------------------------------------------
# Page: Key Insights
# ----------------------------------------------------------------------
else:
    st.title("Key Insights & Business Recommendations")

    st.markdown("""
    - **Product over-holding** is the single largest churn driver: customers with 3+ products
      churn at 83–100%, versus a healthy 7.6% for the 2-product segment.
    - **Engagement** matters more than demographics: inactive members churn at roughly
      twice the rate of active members.
    - **Germany** is a structurally higher-risk market (~32% churn vs. ~16% in France/Spain).
    - **Age** shows a steady, positive association with churn risk.
    - **Tenure alone** is a weak predictor — relationship length doesn't guarantee loyalty
      without engagement.
    """)

    st.subheader("Recommendations")
    st.markdown("""
    1. Trigger priority retention outreach for customers scoring above the high-risk threshold.
    2. Investigate the 3–4 product segment for product-fit / over-selling issues.
    3. Launch a Germany-specific retention and competitive review.
    4. Prioritize engagement-activation campaigns for inactive members.
    5. Use the batch scoring page to simulate portfolio-wide risk before committing budget.
    """)

    st.subheader("Training Data Snapshot")
    st.dataframe(train_ref.head(20), use_container_width=True)
