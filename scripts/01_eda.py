"""
Step 1: Exploratory Data Analysis
Predictive Modeling and Risk Scoring for Bank Customer Churn
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 120

df = pd.read_csv("data/European_Bank.csv")
print(df.shape)
print(df.describe(include="all").T)

# ---------- Target distribution ----------
fig, ax = plt.subplots(figsize=(5, 4))
counts = df["Exited"].value_counts().sort_index()
ax.bar(["Retained (0)", "Churned (1)"], counts.values, color=["#2E86AB", "#C1121F"])
for i, v in enumerate(counts.values):
    ax.text(i, v + 50, f"{v}\n({v/len(df)*100:.1f}%)", ha="center")
ax.set_title("Customer Churn Distribution")
ax.set_ylabel("Number of Customers")
plt.tight_layout()
plt.savefig("figures/01_churn_distribution.png")
plt.close()

# ---------- Churn by Geography / Gender ----------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
sns.barplot(data=df, x="Geography", y="Exited", hue="Geography", ax=axes[0], palette="Blues_d", errorbar=None, legend=False)
axes[0].set_title("Churn Rate by Geography")
axes[0].set_ylabel("Churn Rate")
sns.barplot(data=df, x="Gender", y="Exited", hue="Gender", ax=axes[1], palette="Reds_d", errorbar=None, legend=False)
axes[1].set_title("Churn Rate by Gender")
axes[1].set_ylabel("Churn Rate")
plt.tight_layout()
plt.savefig("figures/02_churn_by_geo_gender.png")
plt.close()

# ---------- Age distribution ----------
fig, ax = plt.subplots(figsize=(6, 4.5))
sns.kdeplot(data=df, x="Age", hue="Exited", fill=True, common_norm=False, alpha=0.4, ax=ax,
            palette={0: "#2E86AB", 1: "#C1121F"})
ax.set_title("Age Distribution by Churn Status")
plt.tight_layout()
plt.savefig("figures/03_age_distribution.png")
plt.close()

# ---------- NumOfProducts vs churn ----------
fig, ax = plt.subplots(figsize=(6, 4.5))
sns.barplot(data=df, x="NumOfProducts", y="Exited", ax=ax, color="#4C72B0", errorbar=None)
ax.set_title("Churn Rate by Number of Products")
ax.set_ylabel("Churn Rate")
plt.tight_layout()
plt.savefig("figures/04_churn_by_numproducts.png")
plt.close()

# ---------- IsActiveMember vs churn ----------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
sns.barplot(data=df, x="IsActiveMember", y="Exited", ax=axes[0], color="#55A630", errorbar=None)
axes[0].set_title("Churn Rate: Active vs Inactive Members")
axes[0].set_xticks([0, 1])
axes[0].set_xticklabels(["Inactive", "Active"])
sns.barplot(data=df, x="HasCrCard", y="Exited", ax=axes[1], color="#E09F3E", errorbar=None)
axes[1].set_title("Churn Rate: Credit Card Ownership")
axes[1].set_xticks([0, 1])
axes[1].set_xticklabels(["No Card", "Has Card"])
plt.tight_layout()
plt.savefig("figures/05_churn_activity_card.png")
plt.close()

# ---------- Balance distribution ----------
fig, ax = plt.subplots(figsize=(6, 4.5))
sns.boxplot(data=df, x="Exited", y="Balance", hue="Exited", ax=ax, palette={0: "#2E86AB", 1: "#C1121F"}, legend=False)
ax.set_xticks([0, 1])
ax.set_xticklabels(["Retained", "Churned"])
ax.set_title("Account Balance by Churn Status")
plt.tight_layout()
plt.savefig("figures/06_balance_boxplot.png")
plt.close()

# ---------- Correlation heatmap ----------
num_cols = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
            "HasCrCard", "IsActiveMember", "EstimatedSalary", "Exited"]
fig, ax = plt.subplots(figsize=(8, 6.5))
corr = df[num_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
ax.set_title("Correlation Matrix (Numeric Features)")
plt.tight_layout()
plt.savefig("figures/07_correlation_heatmap.png")
plt.close()

# ---------- Tenure vs churn ----------
fig, ax = plt.subplots(figsize=(6, 4.5))
sns.barplot(data=df, x="Tenure", y="Exited", ax=ax, color="#7209B7", errorbar=None)
ax.set_title("Churn Rate by Tenure (Years)")
ax.set_ylabel("Churn Rate")
plt.tight_layout()
plt.savefig("figures/08_churn_by_tenure.png")
plt.close()

print("\nGeography churn rates:\n", df.groupby("Geography")["Exited"].mean())
print("\nGender churn rates:\n", df.groupby("Gender")["Exited"].mean())
print("\nActive member churn rates:\n", df.groupby("IsActiveMember")["Exited"].mean())
print("\nNumProducts churn rates:\n", df.groupby("NumOfProducts")["Exited"].mean())
print("\nZero-balance customers:", (df["Balance"] == 0).sum(), f"({(df['Balance']==0).mean()*100:.1f}%)")
print("\nZero-balance churn rate:", df.loc[df["Balance"] == 0, "Exited"].mean())
print("Non-zero-balance churn rate:", df.loc[df["Balance"] > 0, "Exited"].mean())

print("\nDone. Figures saved to figures/")
