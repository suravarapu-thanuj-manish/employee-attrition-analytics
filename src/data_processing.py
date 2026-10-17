import numpy as np
import pandas as pd


def load_and_clean_data(file_path: str) -> pd.DataFrame:
  """Loads raw HR data, normalizes column names, and drops sensitive fields."""
  df = pd.read_csv(file_path)

  # Standardize naming differences
  column_renames = {
      "Years_at_Company": "YearsAtCompany",
      "Years_in_Current_Role": "YearsInCurrentRole",
      "Monthly_Income": "MonthlyIncome",
      "Overtime": "OverTime",
      "Job_Role": "JobRole",
      "Work_Life_Balance": "WorkLifeBalance",
      "Job_Satisfaction": "JobSatisfaction",
      "Performance_Rating": "PerformanceRating",
  }
  df = df.rename(columns=column_renames)

  # Drop direct personal identifiers
  sensitive_cols = [
      "Employee_ID",
      "EmployeeID",
      "EmployeeNumber",
      "EmployeeCount",
      "Over18",
      "StandardHours",
      "Name",
      "EmployeeName",
  ]
  df = df.drop(
      columns=[col for col in sensitive_cols if col in df.columns],
      errors="ignore",
  )
  df = df.dropna().drop_duplicates()

  # Standardize target variable
  if "Attrition" in df.columns:
    df["Attrition_Flag"] = (
        df["Attrition"].astype(str).str.strip().str.lower() == "yes"
    ).astype(int)

  # Derived Metrics: Tenure Bands
  bins = [-1, 1, 3, 5, 10, 50]
  labels = ["0-1 yrs", "2-3 yrs", "4-5 yrs", "6-10 yrs", "10+ yrs"]
  if "YearsAtCompany" in df.columns:
    df["TenureBand"] = pd.cut(df["YearsAtCompany"], bins=bins, labels=labels)

  return df


def get_preprocessed_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
  """Prepares numerical and one-hot encoded categorical features for modeling."""
  exclude_cols = ["Attrition", "Attrition_Flag"]
  X = df.drop(columns=[col for col in exclude_cols if col in df.columns])
  y = df["Attrition_Flag"]

  categorical_cols = X.select_dtypes(
      include=["object", "category"]
  ).columns.tolist()
  X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

  return X, y