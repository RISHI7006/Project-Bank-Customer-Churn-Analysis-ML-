# Bank Customer Churn — Predictive Risk Intelligence

Predicting customer churn for a European retail bank using engineered features,
gradient boosting / XGBoost models, and SHAP-based explainability, deployed as
an interactive Streamlit dashboard.

## 🚀 Live App
[Add your deployed Streamlit URL here after deployment]

## 📄 Research Paper
See [`docs/Research_Paper_Bank_Churn.docx`](docs/Research_Paper_Bank_Churn.docx)

## Project Structure

```
├── app/
│   ├── app.py              # Streamlit dashboard (entry point)
│   └── models/              # Trained models & metadata used by the app
├── scripts/
│   ├── 01_eda.py            # Exploratory data analysis
│   ├── 02_modeling.py       # Model training & comparison
│   └── 03_explainability.py # SHAP explainability analysis
├── figures/                 # Generated charts from the analysis
├── docs/                    # Research paper & executive summary
├── data/                    # Raw & engineered datasets
├── requirements.txt
└── README.md
```

## Dashboard Features

- **🎯 Risk Calculator** — real-time churn probability for a single customer, with SHAP explanation
- **📊 Probability Distribution** — portfolio-wide churn risk visualization
- **🔍 Feature Importance** — global model drivers + SHAP impact analysis
- **🧪 What-If Simulator** — test retention interventions (e.g. engagement, product count)
- **📈 Model Performance** — comparison across candidate models

## Run Locally

```bash
git clone <this-repo-url>
cd bank-churn-prediction
pip install -r requirements.txt
streamlit run app/app.py
```

The app will open at http://localhost:8501

## Model

- **Production model:** Gradient Boosting (best ROC-AUC on held-out test data)
- **Explainability model:** XGBoost + SHAP TreeExplainer
- Full training/evaluation pipeline in `scripts/02_modeling.py`
