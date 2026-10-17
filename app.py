import numpy as np
import pandas as pd
import plotly.express as px
from src.data_processing import get_preprocessed_features, load_and_clean_data
from src.model import train_attrition_model
import streamlit as st

st.set_page_config(page_title="HR Attrition Analytics", layout="wide")


@st.cache_data
def get_data():
  try:
    return load_and_clean_data("data/raw/employee_data.csv")
  except FileNotFoundError:
    # Generates a synthetic dataset for quick testing if data file is absent
    np.random.seed(42)
    n = 1000
    departments = ["Sales", "R&D", "Human Resources"]
    roles = [
        "Sales Executive",
        "Research Scientist",
        "Laboratory Technician",
        "Manager",
    ]

    sample_df = pd.DataFrame({
        "Department": np.random.choice(departments, n, p=[0.3, 0.6, 0.1]),
        "JobRole": np.random.choice(roles, n),
        "YearsAtCompany": np.random.exponential(scale=5, size=n).astype(int),
        "MonthlyIncome": np.random.randint(2500, 15000, size=n),
        "OverTime": np.random.choice(["Yes", "No"], n, p=[0.28, 0.72]),
        "WorkLifeBalance": np.random.choice([1, 2, 3, 4], n, p=[0.1, 0.3, 0.4, 0.2]),
        "Attrition": np.random.choice(["Yes", "No"], n, p=[0.16, 0.84]),
    })
    sample_df.to_csv("data/raw/employee_data.csv", index=False)
    return load_and_clean_data("data/raw/employee_data.csv")


df = get_data()

st.title("Workforce Retention & Attrition Insights")
st.markdown("Diagnosing turnover patterns to guide evidence-based retention.")

# --- SIDEBAR FILTERS ---
st.sidebar.header("Filter Segment")
selected_dept = st.sidebar.multiselect(
    "Department",
    options=df["Department"].unique(),
    default=df["Department"].unique(),
)
filtered_df = df[df["Department"].isin(selected_dept)]

# --- KPIS ---
total_headcount = len(filtered_df)
attrition_count = filtered_df["Attrition_Flag"].sum()
attrition_rate = (
    (attrition_count / total_headcount * 100) if total_headcount > 0 else 0
)

col1, col2, col3 = st.columns(3)
col1.metric("Active Headcount", f"{total_headcount:,}")
col2.metric("Total Exits", f"{attrition_count:,}")
col3.metric("Attrition Rate", f"{attrition_rate:.1f}%")

st.divider()

# --- CHARTS ---
r1_col1, r1_col2 = st.columns(2)

with r1_col1:
  st.subheader("Attrition Rate by Tenure Band")
  tenure_grp = (
      filtered_df.groupby("TenureBand", observed=False)["Attrition_Flag"]
      .mean()
      .reset_index()
  )
  tenure_grp["Attrition_Rate_%"] = tenure_grp["Attrition_Flag"] * 100
  fig_tenure = px.bar(
      tenure_grp,
      x="TenureBand",
      y="Attrition_Rate_%",
      color="Attrition_Rate_%",
      color_continuous_scale="Reds",
  )
  st.plotly_chart(fig_tenure, use_container_width=True)

with r1_col2:
  st.subheader("Attrition Rate by Overtime Status")
  if "OverTime" in filtered_df.columns:
    ot_grp = (
        filtered_df.groupby("OverTime")["Attrition_Flag"].mean().reset_index()
    )
    ot_grp["Attrition_Rate_%"] = ot_grp["Attrition_Flag"] * 100
    fig_ot = px.bar(
        ot_grp,
        x="OverTime",
        y="Attrition_Rate_%",
        color="OverTime",
        color_discrete_map={"Yes": "#EF553B", "No": "#636EFA"},
    )
    st.plotly_chart(fig_ot, use_container_width=True)

# --- MODEL INSIGHTS ---
st.divider()
st.subheader("Key Drivers (Random Forest Feature Importance)")

X, y = get_preprocessed_features(filtered_df)
if len(y.unique()) > 1:
  _, feat_imp, _ = train_attrition_model(X, y)
  fig_imp = px.bar(
      feat_imp.head(8),
      x="Importance",
      y="Feature",
      orientation="h",
      title="Top 8 Predictors of Turnover",
  )
  fig_imp.update_layout(yaxis={"autorange": "reversed"})
  st.plotly_chart(fig_imp, use_container_width=True)

# --- RECOMMENDATIONS ---
st.subheader("Strategic Recommendations")
st.markdown("""
- **Early-Tenure Intervention:** Review onboarding and initial manager touchpoints within the **0-1 year** mark where churn typically spikes.
- **Overtime Rebalancing:** Conduct workload audits for teams logging continuous overtime to mitigate burnout risk.
- **Compensation Auditing:** Benchmark base salaries across departments with elevated voluntary exits.
""")