# Employee Attrition Analytics & Retention Engine

## Project Overview
This project applies workforce analytics and machine learning to HR records to understand patterns associated with employee turnover and identify strategic areas to improve retention.

---

## Step-by-Step Approach
1. **Data Ingestion & Schema Alignment:** Combined and structured employee records covering tenure, department, role, compensation, performance, and exit status.
2. **Data Hygiene & Privacy Protection:** Removed direct identifiers (e.g., Employee IDs, names) to protect sensitive employee records and validated data types.
3. **Segmentation Analysis:** Analyzed attrition trends across departments, job roles, tenure cohorts, compensation brackets, and workload indicators (overtime).
4. **Derived Metrics:** Computed annual attrition rates, monthly equivalent turnover, and early-tenure turnover rates (0–1 year tenure).
5. **Statistical & Predictive Modeling:** Performed Chi-Square tests and independent Two-Sample T-tests to test significance, evaluated multivariate odds ratios via Logistic Regression, and trained a Random Forest classifier to identify top turnover drivers.
6. **Executive Dashboard:** Built an interactive Streamlit dashboard summarizing turnover KPIs, feature importances, analytical caveats, and evidence-based retention interventions.

---

## Key Findings & Strategic Insights

### 1. Strongest Patterns
- **Early-Tenure Spike:** Turnover peaks substantially in the **0–1 year** window, highlighting onboarding vulnerabilities.
- **Workload Exhaustion:** Employees logging regular **Overtime** display significantly higher attrition rates compared to standard schedule peers.
- **Compensation Disparities:** Employees in entry salary quartiles depart at elevated frequencies compared to upper compensation tiers.

### 2. Caveats
- **Imbalance Considerations:** Turnover events constitute a minority fraction of workforce headcount, necessitating balanced class weighting during model calibration.
- **Correlation vs. Causation:** Overtime strongly correlates with attrition, but may reflect systemic team understaffing rather than individual burnout alone.
- **Survey Lag:** Work-life balance and job satisfaction metrics depend on periodic survey cadence and may trail real-time sentiment.

### 3. Recommended Retention Actions
- **Structured 90-Day Onboarding:** Implement structured manager check-ins at 30, 60, and 90 days to curb early-tenure churn.
- **Workload & Overtime Audits:** Establish quarterly capacity reviews for teams consistently logging overtime hours.
- **Targeted Compensation Reviews:** Benchmark entry and mid-tier salaries against industry averages to reduce market poaching.

---

## Project Structure
```text
employee-attrition-analytics/
├── data/
│   ├── raw/
│   │   └── employee_data.csv
│   └── processed/
│       └── cleaned_attrition.csv
├── notebooks/
│   └── 01_eda_and_statistical_tests.ipynb
├── src/
│   ├── __init__.py
│   ├── data_processing.py
│   └── model.py
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
