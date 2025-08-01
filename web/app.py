# web/app.py
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
from src.metrics import load_data, compute_metrics
from src.risk_score import compute_risk_score, normalize
from src.risk_score import compute_risk_score, normalize, load_bounds

st.title("📊 Risk Analysis Dashboard")

# 1. Загрузка даних
df = load_data('data/tasks.csv')
st.sidebar.markdown("### Filters")
# (тут пізніше можна додати фільтри за датами, пріоритетами тощо)

# 2. Показ базових метрик
metrics = compute_metrics(df)
st.subheader("Core Metrics")
st.write(pd.DataFrame.from_dict(metrics, orient='index', columns=['Value']))

# 3. Risk Score
# Для demo: межі за історією можна захардкодити або динамічно вирахувати
bounds = load_bounds('data/history.csv')
risk = compute_risk_score(metrics, bounds)
st.subheader(f"Overall Risk Score: {risk}")
# Порогові позначки
if risk < 0.3:
    st.success("Low Risk")
elif risk < 0.6:
    st.warning("Medium Risk")
else:
    st.error("High Risk")

# 4. Деталізація за метриками
st.subheader("Metric Breakdown")
for k, v in metrics.items():
    st.write(f"**{k}**: {v} (norm: {round(normalize(v, *bounds[k]), 2)})")