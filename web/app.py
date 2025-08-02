import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
from src.metrics import load_data, compute_metrics
from src.risk_score import compute_risk_score, normalize, load_bounds

st.title("📊 Risk Analysis Dashboard")

# 1. Загрузка даних
df = load_data('data/tasks.csv')

# 2. Показ базових метрик
metrics = compute_metrics(df)
st.subheader("Core Metrics")
metrics_labels = {
    "total_tasks": "Total number of tasks",
    "critical_bugs": "Number of critical bugs",
    "blocked": "Number of blocked tasks",
    "avg_dev_days": "Average deviation between estimate and actual (days)",
    "added_mid_sprint": "Number of tasks added after sprint started",
    "avg_duration_days": "Average task duration (days)",
    "pct_on_estimate": "% of tasks completed within estimate",
    "total_bugs": "Total number of bugs",
    "total_reopened": "Number of reopened tasks"
}

# Виведення метрик із новими назвами
metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])
metrics_df.index = metrics_df.index.map(lambda x: metrics_labels.get(x, x))  # мапінг на англійські назви
st.write(metrics_df)

# 3. Ризиковий бал
bounds = load_bounds('data/history.csv')
risk = compute_risk_score(metrics, bounds)
st.subheader(f"Overall Risk Score: {risk}")

# 4. Деталізація за метриками
st.subheader("Metric Breakdown")
for k, v in metrics.items():
    label = metrics_labels.get(k, k)
    st.write(f"**{label}:** {v} (normalized: {round(normalize(v, *bounds[k]), 2)})")