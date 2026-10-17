import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
import statsmodels.api as sm


def run_logistic_regression_stats(X: pd.DataFrame, y: pd.Series):
  """Fits a statsmodels Logit to identify meaningful predictors and p-values."""
  X_const = sm.add_constant(X.astype(float))
  logit_model = sm.Logit(y, X_const)
  try:
    result = logit_model.fit(disp=False)
    summary_df = pd.DataFrame({
        "Odds_Ratio": np.exp(result.params),
        "P_Value": result.pvalues,
    }).sort_values(by="P_Value")
    return summary_df
  except Exception as e:
    return f"Logit convergence note: {e}"


def train_attrition_model(X: pd.DataFrame, y: pd.Series):
  """Trains a Random Forest model with class balancing."""
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.25, random_state=42, stratify=y
  )

  model = RandomForestClassifier(
      n_estimators=150, max_depth=6, class_weight="balanced", random_state=42
  )
  model.fit(X_train, y_train)

  y_pred = model.predict(X_test)
  y_prob = model.predict_proba(X_test)[:, 1]

  metrics = {
      "roc_auc": roc_auc_score(y_test, y_prob),
      "report": classification_report(y_test, y_pred, output_dict=True),
  }

  feature_importance = pd.DataFrame({
      "Feature": X.columns,
      "Importance": model.feature_importances_,
  }).sort_values(by="Importance", ascending=False)

  return model, feature_importance, metrics