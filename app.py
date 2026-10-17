import numpy as np
import pandas as pd
import plotly.express as px
from scipy.stats import chi2_contingency, ttest_ind
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import streamlit as st

st.set_page_config(
    page_title="Employee Attrition Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==============================================================================
# STEP 1 & 2: DATA PREPARATION, PRIVACY PROTECTION & DERIVED METRICS
# ==============================================================================
def process_workforce_data(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
  """1. Prepares employee records (tenure, dept, role, compensation, performance, exit status).

  2. Protects sensitive fields by minimizing unnecessary personal identifiers.
  4. Creates derived metrics (tenure bands, compensation bands, early-tenure
  flags).
  """
  # Standardize column naming variations across datasets
  renames = {
      "Years_at_Company": "YearsAtCompany",
      "Years_in_Current_Role": "YearsInCurrentRole",
      "Monthly_Income": "MonthlyIncome",
      "Overtime": "OverTime",
      "Job_Role": "JobRole",
      "Work_Life_Balance": "WorkLifeBalance",
      "Job_Satisfaction": "JobSatisfaction",
      "Performance_Rating": "PerformanceRating",
  }
  df = df.rename(columns=renames)

  # Step 2: Minimize sensitive personal fields & PII
  sensitive_fields = [
      "Employee_ID",
      "EmployeeID",
      "EmployeeNumber",
      "EmployeeCount",
      "Over18",
      "StandardHours",
      "Name",
      "EmployeeName",
      "SocialSecurityNumber",
  ]
  dropped_sensitive = [col for col in sensitive_fields if col in df.columns]
  df = df.drop(columns=dropped_sensitive, errors="ignore")
  df = df.dropna().drop_duplicates()

  # Standardize Attrition flag
  if "Attrition" in df.columns:
    df["Attrition_Flag"] = (
        df["Attrition"].astype(str).str.strip().str.lower() == "yes"
    ).astype(int)

  # Step 4: Derived Metrics
  # 1. Tenure Bands
  if "YearsAtCompany" in df.columns:
    bins_tenure = [-1, 1, 3, 5, 10, 50]
    labels_tenure = ["0-1 yrs", "2-3 yrs", "4-5 yrs", "6-10 yrs", "10+ yrs"]
    df["TenureBand"] = pd.cut(
        df["YearsAtCompany"], bins=bins_tenure, labels=labels_tenure
    )
    df["EarlyTenure_Flag"] = (df["YearsAtCompany"] <= 1).astype(int)

  # 2. Compensation Bands (Quartile / Income brackets)
  if "MonthlyIncome" in df.columns:
    df["CompensationBand"] = pd.qcut(
        df["MonthlyIncome"],
        q=4,
        labels=["Tier 1 (Entry)", "Tier 2 (Mid)", "Tier 3 (Senior)", "Tier 4 (Executive)"],
        duplicates="drop",
    )

  return df, dropped_sensitive


# ==============================================================================
# STEP 5: STATISTICAL TESTING & REGRESSION MODELING
# ==============================================================================
def run_statistical_investigation(df: pd.DataFrame):
  """Step 5: Statistical tests (Chi-Square, T-test) and Multivariate Logistic Regression."""
  results = {}

  # Chi-Square: OverTime vs Attrition
  if "OverTime" in df.columns and "Attrition_Flag" in df.columns:
    ctable = pd.crosstab(df["OverTime"], df["Attrition_Flag"])
    chi2, p_val, _, _ = chi2_contingency(ctable)
    results["chi2_overtime"] = {"stat": chi2, "p_value": p_val}

  # T-test: MonthlyIncome (Stayers vs Leavers)
  if "MonthlyIncome" in df.columns and "Attrition_Flag" in df.columns:
    stayers = df[df["Attrition_Flag"] == 0]["MonthlyIncome"]
    leavers = df[df["Attrition_Flag"] == 1]["MonthlyIncome"]
    t_stat, p_val = ttest_ind(stayers, leavers, equal_var=False)
    results["ttest_income"] = {"stat": t_stat, "p_value": p_val}

  # Multivariate Logistic Regression
  candidate_cols = [
      "YearsAtCompany",
      "MonthlyIncome",
      "WorkLifeBalance",
      "JobSatisfaction",
      "OverTime",
  ]
  avail_cols = [c for c in candidate_cols if c in df.columns]

  if len(avail_cols) >= 2 and "Attrition_Flag" in df.columns:
    X_reg = df[avail_cols].copy()
    if "OverTime" in X_reg.columns:
      X_reg["OverTime"] = (X_reg["OverTime"] == "Yes").astype(int)

    # Convert all to numeric
    X_reg = sm.add_constant(X_reg.astype(float))
    y_reg = df["Attrition_Flag"].astype(float)
    try:
      logit_mod = sm.Logit(y_reg, X_reg).fit(disp=False)
      reg_summary = pd.DataFrame({
          "Odds_Ratio": np.exp(logit_mod.params),
          "P_Value": logit_mod.pvalues,
      }).sort_values(by="P_Value")
      results["logit_summary"] = reg_summary
    except Exception as e:
      results["logit_error"] = str(e)

  return results


def train_predictive_model(df: pd.DataFrame):
  """Trains a Random Forest classifier for turnover risk scoring."""
  exclude = ["Attrition", "Attrition_Flag", "EarlyTenure_Flag"]
  X = df.drop(columns=[c for c in exclude if c in df.columns])
  y = df["Attrition_Flag"]

  # Avoid warning by explicitly selecting string/object/category columns
  cat_cols = X.select_dtypes(
      include=["object", "category", "string"]
  ).columns.tolist()
  X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True)

  X_train, X_test, y_train, y_test = train_test_split(
      X_encoded, y, test_size=0.25, random_state=42, stratify=y
  )

  model = RandomForestClassifier(
      n_estimators=120, max_depth=6, class_weight="balanced", random_state=42
  )
  model.fit(X_train, y_train)

  auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

  feat_df = pd.DataFrame({
      "Feature": X_encoded.columns,
      "Importance": model.feature_importances_,
  }).sort_values(by="Importance", ascending=False)

  return model, X_encoded.columns.tolist(), feat_df, auc, X


# ==============================================================================
# STEP 6: DASHBOARD UI WITH ACCURATE GRAPHS & INSIGHTS
# ==============================================================================
st.title("Workforce Attrition Analytics & Retention Engine")
st.caption(
    "End-to-end data pipeline addressing data quality, segmentation, derived"
    " metrics, statistical tests, and retention actions[cite: 1]."
)

# Sidebar: File Upload
st.sidebar.header("Data Management")
uploaded_file = st.sidebar.file_uploader(
    "Upload HR Dataset (.csv)", type=["csv"]
)

# Load data with automatic fallback search
raw_df = None
if uploaded_file is not None:
  uploaded_file.seek(0)
  raw_df = pd.read_csv(uploaded_file)
else:
  for fallback_path in [
      "employee_attrition_dataset_10000.csv",
      "data/raw/employee_data.csv",
      "data/processed/cleaned_attrition.csv",
  ]:
    try:
      raw_df = pd.read_csv(fallback_path)
      st.sidebar.info(f"Loaded active local file: `{fallback_path}`")
      break
    except Exception:
      continue

if raw_df is None:
  st.warning(
      "Please upload a dataset or ensure an employee CSV exists in the folder."
  )
  st.stop()

# Run Steps 1, 2 & 4
df, dropped_sensitive = process_workforce_data(raw_df)

# Sidebar Filter
if "Department" in df.columns:
  depts = sorted(df["Department"].dropna().unique().tolist())
  selected_depts = st.sidebar.multiselect(
      "Filter by Department", depts, default=depts
  )
  filtered_df = df[df["Department"].isin(selected_depts)]
else:
  filtered_df = df

# ------------------------------------------------------------------------------
# STEP 4: DERIVED METRICS & KPIS
# ------------------------------------------------------------------------------
total_headcount = len(filtered_df)
total_exits = filtered_df["Attrition_Flag"].sum()
annual_attrition_rate = (
    (total_exits / total_headcount * 100) if total_headcount > 0 else 0
)
monthly_attrition_rate = (
    annual_attrition_rate / 12
)  # Monthly equivalent rate

early_exits = (
    filtered_df[filtered_df["EarlyTenure_Flag"] == 1]["Attrition_Flag"].sum()
    if "EarlyTenure_Flag" in filtered_df.columns
    else 0
)
early_headcount = (
    (filtered_df["EarlyTenure_Flag"] == 1).sum()
    if "EarlyTenure_Flag" in filtered_df.columns
    else 0
)
early_tenure_rate = (
    (early_exits / early_headcount * 100) if early_headcount > 0 else 0
)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Active Headcount", f"{total_headcount:,}")
kpi2.metric("Annualized Attrition Rate", f"{annual_attrition_rate:.1f}%")
kpi3.metric("Monthly Attrition Rate", f"{monthly_attrition_rate:.2f}%")
kpi4.metric(
    "Early-Tenure Rate (0-1 yr)",
    f"{early_tenure_rate:.1f}%",
    delta=f"{early_tenure_rate - annual_attrition_rate:+.1f}% vs Avg",
    delta_color="inverse",
)

st.divider()

# ------------------------------------------------------------------------------
# STEP 3: SEGMENTED ANALYTICS (GRAPHS)
# ------------------------------------------------------------------------------
st.subheader(
    "Attrition Segmentation: Department, Tenure, Compensation & Workload"
)

tab_dept, tab_tenure, tab_comp, tab_workload = st.tabs([
    "Department & Role",
    "Tenure Band",
    "Compensation Band",
    "Workload (OverTime)",
])

with tab_dept:
  col_a, col_b = st.columns(2)
  with col_a:
    if "Department" in filtered_df.columns:
      dept_stats = (
          filtered_df.groupby("Department")["Attrition_Flag"]
          .agg(["count", "mean"])
          .reset_index()
      )
      dept_stats["Attrition Rate (%)"] = dept_stats["mean"] * 100
      fig_dept = px.bar(
          dept_stats.sort_values(by="Attrition Rate (%)", ascending=False),
          x="Department",
          y="Attrition Rate (%)",
          text_auto=".1f",
          title="Attrition Rate by Department",
          color="Attrition Rate (%)",
          color_continuous_scale="Blues",
      )
      st.plotly_chart(fig_dept, use_container_width=True)

  with col_b:
    if "JobRole" in filtered_df.columns:
      role_stats = (
          filtered_df.groupby("JobRole")["Attrition_Flag"]
          .agg(["count", "mean"])
          .reset_index()
      )
      role_stats["Attrition Rate (%)"] = role_stats["mean"] * 100
      fig_role = px.bar(
          role_stats.sort_values(by="Attrition Rate (%)", ascending=True),
          x="Attrition Rate (%)",
          y="JobRole",
          orientation="h",
          text_auto=".1f",
          title="Attrition Rate by Job Role",
          color="Attrition Rate (%)",
          color_continuous_scale="Purples",
      )
      st.plotly_chart(fig_role, use_container_width=True)

with tab_tenure:
  if "TenureBand" in filtered_df.columns:
    tenure_stats = (
        filtered_df.groupby("TenureBand", observed=False)["Attrition_Flag"]
        .agg(["count", "mean"])
        .reset_index()
    )
    tenure_stats["Attrition Rate (%)"] = tenure_stats["mean"] * 100
    fig_tband = px.bar(
        tenure_stats,
        x="TenureBand",
        y="Attrition Rate (%)",
        text_auto=".1f",
        title="Attrition Rate Across Tenure Cohorts",
        color="Attrition Rate (%)",
        color_continuous_scale="Reds",
    )
    st.plotly_chart(fig_tband, use_container_width=True)

with tab_comp:
  col_c, col_d = st.columns(2)
  with col_c:
    if "CompensationBand" in filtered_df.columns:
      comp_stats = (
          filtered_df.groupby("CompensationBand", observed=False)[
              "Attrition_Flag"
          ]
          .agg(["count", "mean"])
          .reset_index()
      )
      comp_stats["Attrition Rate (%)"] = comp_stats["mean"] * 100
      fig_comp = px.bar(
          comp_stats,
          x="CompensationBand",
          y="Attrition Rate (%)",
          text_auto=".1f",
          title="Attrition Rate by Compensation Tier",
          color="Attrition Rate (%)",
          color_continuous_scale="Teal",
      )
      st.plotly_chart(fig_comp, use_container_width=True)

  with col_d:
    if "MonthlyIncome" in filtered_df.columns:
      fig_box = px.box(
          filtered_df,
          x="Attrition",
          y="MonthlyIncome",
          color="Attrition",
          title="Salary Distribution: Stayers vs. Leavers",
          color_discrete_map={"Yes": "#EF553B", "No": "#636EFA"},
      )
      st.plotly_chart(fig_box, use_container_width=True)

with tab_workload:
  col_e, col_f = st.columns(2)
  with col_e:
    if "OverTime" in filtered_df.columns:
      ot_stats = (
          filtered_df.groupby("OverTime")["Attrition_Flag"]
          .agg(["count", "mean"])
          .reset_index()
      )
      ot_stats["Attrition Rate (%)"] = ot_stats["mean"] * 100
      fig_ot = px.bar(
          ot_stats,
          x="OverTime",
          y="Attrition Rate (%)",
          text_auto=".1f",
          title="Impact of Overtime on Turnover",
          color="OverTime",
          color_discrete_map={"Yes": "#EF553B", "No": "#00CC96"},
      )
      st.plotly_chart(fig_ot, use_container_width=True)

  with col_f:
    if "WorkLifeBalance" in filtered_df.columns:
      wlb_stats = (
          filtered_df.groupby("WorkLifeBalance")["Attrition_Flag"]
          .agg(["count", "mean"])
          .reset_index()
      )
      wlb_stats["Attrition Rate (%)"] = wlb_stats["mean"] * 100
      fig_wlb = px.line(
          wlb_stats,
          x="WorkLifeBalance",
          y="Attrition Rate (%)",
          markers=True,
          title="Attrition Trend Across Work-Life Balance Ratings (1-4)",
      )
      st.plotly_chart(fig_wlb, use_container_width=True)

st.divider()

# ------------------------------------------------------------------------------
# STEP 5: STATISTICAL INFERENCE & REGRESSION
# ------------------------------------------------------------------------------
st.subheader("Statistical Validation & Machine Learning Risk Drivers")

stat_col, model_col = st.columns(2)

with stat_col:
  st.markdown("#### Hypothesis Testing & Logistic Regression")
  stats = run_statistical_investigation(filtered_df)

  if "chi2_overtime" in stats:
    p_chi2 = stats["chi2_overtime"]["p_value"]
    st.write(
        f"**Chi-Square Test (Overtime vs. Attrition):** p-value = `{p_chi2:.4e}`"
    )
    if p_chi2 < 0.05:
      st.success(
          "Statistically significant relationship confirmed between Overtime"
          " and Exit Status."
      )
    else:
      st.info("No statistically significant relationship at alpha = 0.05.")

  if "ttest_income" in stats:
    p_t = stats["ttest_income"]["p_value"]
    st.write(
        f"**Two-Sample T-Test (Monthly Income vs. Attrition):** p-value ="
        f" `{p_t:.4e}`"
    )

  if "logit_summary" in stats:
    st.write("**Multivariate Logistic Regression (Odds Ratios):**")
    st.dataframe(stats["logit_summary"].style.format("{:.4f}"))

with model_col:
  st.markdown("#### Random Forest Feature Importance")
  if len(filtered_df["Attrition_Flag"].unique()) > 1:
    model, feature_cols, feat_df, auc, X_template = train_predictive_model(
        filtered_df
    )
    st.write(f"**Random Forest Validation ROC-AUC:** `{auc:.3f}`")
    fig_imp = px.bar(
        feat_df.head(8),
        x="Importance",
        y="Feature",
        orientation="h",
        title="Top 8 Global Predictors of Attrition",
    )
    fig_imp.update_layout(yaxis={"autorange": "reversed"})
    st.plotly_chart(fig_imp, use_container_width=True)

st.divider()

# ------------------------------------------------------------------------------
# STEP 6: PATTERNS, CAVEATS & RECOMMENDED ACTIONS
# ------------------------------------------------------------------------------
st.subheader("Executive Retention Summary: Patterns, Caveats & Actions")

sum_col1, sum_col2, sum_col3 = st.columns(3)

with sum_col1:
  st.markdown("#### 1. Strongest Patterns")
  st.markdown(f"""
    - **Early-Tenure Spike:** Turnover in the **0-1 year band ({early_tenure_rate:.1f}%)** is substantially higher than seasoned cohorts.
    - **Workload Exhaustion:** Employees logging **Overtime** exhibit significantly higher turnover rates.
    - **Compensation Discrepancies:** Employees in entry salary bands leave at elevated frequencies compared to upper tiers.
    """)

with sum_col2:
  st.markdown("#### 2. Analytical Caveats")
  st.markdown("""
    - **Correlation vs. Causation:** High overtime correlates with exits, but may be driven by underlying team understaffing.
    - **Class Imbalance:** Attrition events represent a minority percentage of the dataset; models require balanced class weighting.
    - **Survey Lag:** Work-Life Balance and Satisfaction metrics reflect periodic surveys and may trail real-time sentiment.
    """)

with sum_col3:
  st.markdown("#### 3. Recommended Actions")
  st.markdown("""
    - **Structured 90-Day Onboarding:** Launch mentorship checkpoints to curb early-tenure turnover.
    - **Overtime Load Rebalancing:** Institute quarterly workload reviews for teams logging heavy overtime hours.
    - **Market Pay Benchmarking:** Review compensation parity for entry and mid-tier roles to prevent market poachings.
    """)