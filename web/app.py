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
    "total_tasks": "Загальна кількість задач",
    "critical_bugs": "Кількість критичних багів",
    "blocked": "Кількість заблокованих задач",
    "avg_dev_days": "Середнє відхилення між оцінкою і фактом (дні)",
    "added_mid_sprint": "Кількість задач, доданих після початку спринту",
    "avg_duration_days": "Середня тривалість задач (дні)",
    "pct_on_estimate": "% задач, що завершені в межах оцінки",
    "total_bugs": "Загальна кількість багів",
    "total_reopened": "Кількість перевідкритих задач"
}

# Виведення метрик із новими назвами
metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])
metrics_df.index = metrics_df.index.map(lambda x: metrics_labels.get(x, x))  # мапінг на людські назви
st.write(metrics_df)

# 3. Ризиковий бал
bounds = load_bounds('data/history.csv')
risk = compute_risk_score(metrics, bounds)
st.subheader(f"Загальний Risk Score: {risk}")

# 4. Деталізація за метриками
st.subheader("Деталізація по метриках")
for k, v in metrics.items():
    label = metrics_labels.get(k, k)
    st.write(f"**{label}:** {v} (нормалізовано: {round(normalize(v, *bounds[k]), 2)})")