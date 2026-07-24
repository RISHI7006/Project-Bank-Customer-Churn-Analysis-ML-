import streamlit as st
import pandas as pd
import numpy as np
import joblib, json
import plotly.graph_objects as go
import plotly.express as px
import shap

st.set_page_config(page_title="Bank Churn Risk Intelligence", page_icon="🏦", layout="wide")

@st.cache_resource
def load_artifacts():
    best_model = joblib.load("models/best_model.pkl")
    xgb_model = joblib.load("models/xgboost_model.pkl")
    feature_columns = joblib.load("models/feature_columns.pkl")
    with open("models/meta.json") as f:
        meta = json.load(f)
    train_ref = pd.read_csv("models/train_reference.csv")
    return best_model, xgb_model, feature_columns, meta, train_ref

@st.cache_resource
def get_explainer(_model):
    return shap.TreeExplainer(_model)

best_model, xgb_model, FEATURE_COLUMNS, META, train_ref = load_artifacts()
explainer = get_explainer(xgb_model)

def engineer_features(raw):
    eps = 1e-6
    balance, salary, tenure, age = raw["Balance"], raw["EstimatedSalary"], raw["Tenure"], raw["Age"]
    num_products, is_active, cs = raw["NumOfProducts"], raw["IsActiveMember"], raw["CreditScore"]
    row = {
        "CreditScore": cs, "Age": age, "Tenure": tenure, "Balance": balance,
        "NumOfProducts": num_products, "HasCrCard": raw["HasCrCard"], "IsActiveMember": is_active,
        "EstimatedSalary": salary, "BalanceSalaryRatio": balance/(salary+eps),
        "ProductDensity": num_products/(tenure+1), "EngagementProductInteraction": is_active*num_products,
        "AgeTenureInteraction": age*tenure, "IsZeroBalance": int(balance==0),
        "CreditScoreBand": int(pd.cut([cs], bins=[0,580,670,740,800,850], labels=[1,2,3,4,5])[0]),
        "Geography_Germany": int(raw["Geography"]=="Germany"), "Geography_Spain": int(raw["Geography"]=="Spain"),
        "Gender_Male": int(raw["Gender"]=="Male"),
    }
    return pd.DataFrame([row])[FEATURE_COLUMNS]

st.title("🏦 Customer Churn Risk Calculator")
c1, c2, c3 = st.columns(3)
with c1:
    credit_score = st.slider("Credit Score", 350, 850, 650)
    age = st.slider("Age", 18, 92, 38)
    tenure = st.slider("Tenure", 0, 10, 5)
with c2:
    balance = st.number_input("Balance (€)", 0.0, 300000.0, 75000.0)
    salary = st.number_input("Salary (€)", 0.0, 250000.0, 100000.0)
    num_products = st.selectbox("Number of Products", [1,2,3,4])
with c3:
    geography = st.selectbox("Geography", ["France","Germany","Spain"])
    gender = st.selectbox("Gender", ["Male","Female"])
    has_card = st.radio("Has Credit Card?", ["Yes","No"], horizontal=True)
    is_active = st.radio("Active Member?", ["Yes","No"], horizontal=True)

raw = {"CreditScore": credit_score, "Age": age, "Tenure": tenure, "Balance": balance,
       "NumOfProducts": num_products, "HasCrCard": 1 if has_card=="Yes" else 0,
       "IsActiveMember": 1 if is_active=="Yes" else 0, "EstimatedSalary": salary,
       "Geography": geography, "Gender": gender}

X_input = engineer_features(raw)
proba = float(best_model.predict_proba(X_input)[0,1])
tier, color = ("High Risk","#C1121F") if proba>=0.5 else ("Medium Risk","#E09F3E") if proba>=0.25 else ("Low Risk","#2A9D8F")

fig = go.Figure(go.Indicator(mode="gauge+number", value=proba*100, number={"suffix":"%"},
    gauge={"axis":{"range":[0,100]}, "bar":{"color":color}}))
st.plotly_chart(fig, use_container_width=True)
st.markdown(f"### Risk Tier: <span style='color:{color}'>{tier}</span>", unsafe_allow_html=True)