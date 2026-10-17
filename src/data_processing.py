import numpy as np
import pandas as pd


def load_and_clean_data(file_path: str) -> pd.DataFrame:
  """Loads raw HR data, removes sensitive PII, and applies quality checks."""
  df = pd.read_csv(file_path)

  # Drop non-predictive/sensitive fields to protect employee privacy
  sensitive_cols = [
      "EmployeeNumber",
      "EmployeeCount",
      "Over18",
      "StandardHours",
      "Name",
  ]
  df = df.drop(
      columns=[col for col in sensitive_cols if col in df.columns],
      errors="ignore",
  )

  # Handle missing values
  df = df.dropna().drop_duplicates()

  # Standardize target variable
  if "Attrition" in df.columns and df["Attrition"].dtype == object:
    df["Attrition_Flag"] = (df["Attrition"].str.strip() == "Yes").astype(int)
  elif "Attrition" in df.columns:
    df["Attrition_Flag"] = df["Attrition"].astype(int)

  # Derived Metrics: Tenure Bands
  bins = [-1, 1, 3, 5, 10, 40]
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

  # Identify categorical columns for one-hot encoding
  categorical_cols = X.select_dtypes(
      include=["object", "category"]
  ).columns.tolist()
  X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

  return X, y