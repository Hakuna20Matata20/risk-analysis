# web/app.py

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

from joblib import load
from src.metrics import load_data, compute_metrics
from src.risk_score import compute_risk_score, normalize, load_bounds

# Load pre-trained ML model
model = load('models/xgb_risk_model.joblib')

# Title
st.title("📊 Risk Analysis Dashboard")

# Sidebar: interactive filters
st.sidebar.header("Filters")
df_raw = load_data('data/tasks.csv')

# 1. Sprint selector
sprint_choice = st.sidebar.selectbox(
    "Select sprint",
    sorted(df_raw['sprint_id'].unique()),
    index=len(df_raw['sprint_id'].unique()) - 1
)
df = df_raw[df_raw['sprint_id'] == sprint_choice]

# 2. Priority multiselect
priorities = st.sidebar.multiselect(
    "Priority levels",
    options=df['priority'].unique(),
    default=list(df['priority'].unique())
)
df = df[df['priority'].isin(priorities)]

# 3. Bug-only checkbox
show_bugs = st.sidebar.checkbox("Show only bugs", value=False)
if show_bugs:
    df = df[df['is_bug'] == True]

# Compute metrics on filtered data
metrics = compute_metrics(df)

# ML Prediction
X_current = np.array([[
    metrics['total_tasks'],
    metrics['critical_bugs'],
    metrics['blocked'],
    metrics['avg_dev_days'],
    metrics['added_mid_sprint'],
    metrics['avg_duration_days'],
    metrics['pct_on_estimate'],
    metrics['total_bugs'],
    metrics['total_reopened']
]])
prob = model.predict_proba(X_current)[0, 1]
st.subheader(f"💡 ML Prediction: {round(prob * 100, 1)}% chance of High-Risk Sprint")

# Core Metrics table
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
metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])
metrics_df.index = metrics_df.index.map(lambda x: metrics_labels.get(x, x))
st.write(metrics_df)

# Overall Risk Score
bounds = load_bounds('data/history.csv')
risk_score = compute_risk_score(metrics, bounds)
st.subheader(f"Overall Risk Score: {risk_score}")

# Load history for visualizations
history_df = pd.read_csv('data/history.csv')

# Якщо в history.csv немає готового стовпця risk_score — обчислимо його тут
if 'risk_score' not in history_df.columns:
    # Спочатку зберемо границі нормалізації (вже має бути імпортовано load_bounds)
    bounds = load_bounds('data/history.csv')
    # Визначимо список фіч, якими ми розраховуємо risk_score
    feature_cols = [
        'total_tasks', 'critical_bugs', 'blocked',
        'avg_dev_days', 'added_mid_sprint', 'avg_duration_days',
        'pct_on_estimate', 'total_bugs', 'total_reopened'
    ]
    # Обчислимо risk_score для кожного спринту
    history_df['risk_score'] = history_df.apply(
        lambda row: compute_risk_score(
            {col: row[col] for col in feature_cols},
            bounds
        ),
        axis=1
    )

# Trend Charts
st.markdown("---")
st.subheader("Trend Charts")

# 1. Risk Score trend
st.markdown("**Risk Score by Sprint**")
st.line_chart(history_df.set_index('sprint_id')['risk_score'])

# 2. Key metrics over time
st.markdown("**Key Metrics Over Time**")
fig_metrics = px.line(
    history_df,
    x='sprint_id',
    y=['critical_bugs', 'blocked', 'pct_on_estimate'],
    labels={'value': 'Count / %', 'variable': 'Metric', 'sprint_id': 'Sprint'}
)
st.plotly_chart(fig_metrics, use_container_width=True)

# Distributions
st.markdown("---")
st.subheader("Distributions")

# 3. Histogram of Risk Scores
st.markdown("**Distribution of Risk Scores**")
fig1, ax1 = plt.subplots()
ax1.hist(history_df['risk_score'], bins=10)
st.pyplot(fig1)

# 4. Box plot of key metrics
st.markdown("**Box Plot of Key Metrics**")
df_melt = history_df[['critical_bugs', 'blocked', 'pct_on_estimate']]\
    .melt(var_name='metric', value_name='value')
fig2 = px.box(df_melt, x='metric', y='value')
st.plotly_chart(fig2, use_container_width=True)

# Correlation Heatmap
st.markdown("---")
st.subheader("Correlation Heatmap")
fig3, ax3 = plt.subplots()
sns.heatmap(history_df.corr(), annot=True, ax=ax3)
st.pyplot(fig3)

# Metric Breakdown with Risk Level and Overage
st.markdown("---")
st.subheader("Metric Breakdown")

def get_risk_category(norm_value: float) -> str:
    if norm_value > 1.0:
        return "Critical"
    elif norm_value > 0.66:
        return "High"
    elif norm_value > 0.33:
        return "Medium"
    else:
        return "Low"

for k, v in metrics.items():
    label = metrics_labels.get(k, k)
    min_h, max_h = bounds[k]
    norm = normalize(v, min_h, max_h)
    risk_cat = get_risk_category(norm)
    overage_pct = ((v - max_h) / max_h * 100) if v > max_h else 0

    if overage_pct > 0:
        overage_label = f"{round(overage_pct, 1)}% above historical max"
    else:
        overage_label = "Within historical range"

    st.write(
        f"**{label}:** {v}  \n"
        f"• Risk Level: **{risk_cat}**  \n"
        f"• Overage: **{overage_label}**"
    )