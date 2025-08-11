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
from src.anomaly import detect_anomalies

# ======================
# Setup & Model loading
# ======================
st.set_page_config(page_title="Risk Analysis Dashboard", layout="wide")

try:
    model = load('models/xgb_risk_model.joblib')
    model_loaded = True
except Exception:
    model = None
    model_loaded = False

# ======================
# Title
# ======================
st.title("📊 Risk Analysis Dashboard")

# ======================
# Sidebar: interactive filters
# ======================
st.sidebar.header("Filters")
df_raw = load_data('data/tasks.csv')

# 1) Sprint selector
sprints = sorted(df_raw['sprint_id'].unique())
sprint_choice = st.sidebar.selectbox("Select sprint", sprints, index=len(sprints) - 1)
df = df_raw[df_raw['sprint_id'] == sprint_choice]

# 2) Priority multiselect
prio_options = list(df['priority'].unique())
priorities = st.sidebar.multiselect("Priority levels", options=prio_options, default=prio_options)
df = df[df['priority'].isin(priorities)]

# 3) Bug-only checkbox
show_bugs = st.sidebar.checkbox("Show only bugs", value=False)
if show_bugs:
    df = df[df['is_bug'] == True]

if not model_loaded:
    st.sidebar.info("⚠️ ML-модель не знайдена. Запустіть тренування: `python3 -m src.train_model`")

# ======================
# Compute metrics for filtered data
# ======================
metrics = compute_metrics(df)

# Bounds & Risk Score
bounds = load_bounds('data/history.csv')
risk_score = compute_risk_score(metrics, bounds)

# ======================
# ML Prediction (if model available)
# ======================
st.subheader("AI Insights")
if model_loaded:
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
    st.metric("ML Prediction — High-Risk Probability", f"{prob*100:.1f}%")
else:
    prob = 0.0
    st.metric("ML Prediction — High-Risk Probability", "N/A")

# ======================
# What-if simulator
# ======================
with st.expander("🧪 What-if simulator"):
    st.caption("Змініть ключові метрики і подивіться, як змінюються Risk Score та ML-ймовірність.")
    base_risk = risk_score
    base_prob = prob

    blocked_sim = st.slider("Blocked tasks", 0, int(max(metrics['blocked'] * 2, 50)), int(metrics['blocked']))
    critical_sim = st.slider("Critical bugs", 0, int(max(metrics['critical_bugs'] * 2, 50)), int(metrics['critical_bugs']))
    added_mid_sim = st.slider("Tasks added after sprint started", 0, int(max(metrics['added_mid_sprint'] * 2, 200)), int(metrics['added_mid_sprint']))
    pct_est_sim = st.slider("% of tasks completed within estimate", 0.0, 100.0, float(metrics['pct_on_estimate']), 0.5)

    sim_metrics = dict(metrics)
    sim_metrics.update({
        'blocked': blocked_sim,
        'critical_bugs': critical_sim,
        'added_mid_sprint': added_mid_sim,
        'pct_on_estimate': pct_est_sim
    })

    sim_risk = compute_risk_score(sim_metrics, bounds)

    if model_loaded:
        X_sim = np.array([[
            sim_metrics['total_tasks'],
            sim_metrics['critical_bugs'],
            sim_metrics['blocked'],
            sim_metrics['avg_dev_days'],
            sim_metrics['added_mid_sprint'],
            sim_metrics['avg_duration_days'],
            sim_metrics['pct_on_estimate'],
            sim_metrics['total_bugs'],
            sim_metrics['total_reopened']
        ]])
        sim_prob = model.predict_proba(X_sim)[0, 1]
    else:
        sim_prob = 0.0

    col1, col2 = st.columns(2)
    col1.metric("Simulated Risk Score", f"{sim_risk:.3f}", delta=f"{sim_risk - base_risk:+.3f}")
    if model_loaded:
        col2.metric("Simulated High-Risk Probability", f"{sim_prob*100:.1f}%", delta=f"{(sim_prob - base_prob)*100:+.1f}%")
    else:
        col2.metric("Simulated High-Risk Probability", "N/A")

# ======================
# Core Metrics table
# ======================
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
st.dataframe(metrics_df)

# ======================
# Overall Risk Score
# ======================
st.subheader(f"Overall Risk Score: {risk_score:.3f}")

# ======================
# Load history & enrich (risk_score + anomalies)
# ======================
history_df = pd.read_csv('data/history.csv')

# Ensure risk_score exists for history
if 'risk_score' not in history_df.columns:
    feature_cols = [
        'total_tasks', 'critical_bugs', 'blocked',
        'avg_dev_days', 'added_mid_sprint', 'avg_duration_days',
        'pct_on_estimate', 'total_bugs', 'total_reopened'
    ]
    history_df['risk_score'] = history_df.apply(
        lambda row: compute_risk_score({c: row[c] for c in feature_cols}, bounds),
        axis=1
    )

# Anomalies
anom_df = detect_anomalies('data/history.csv')
history_df = history_df.merge(anom_df, on='sprint_id', how='left')

# ======================
# Trend Charts
# ======================
st.markdown("---")
st.subheader("Trend Charts")

st.markdown("**Risk Score by Sprint**")
st.line_chart(history_df.set_index('sprint_id')['risk_score'])

st.markdown("**Key Metrics Over Time**")
fig_metrics = px.line(
    history_df,
    x='sprint_id',
    y=['critical_bugs', 'blocked', 'pct_on_estimate'],
    labels={'value': 'Count / %', 'variable': 'Metric', 'sprint_id': 'Sprint'}
)
st.plotly_chart(fig_metrics, use_container_width=True)

st.markdown("**Anomaly view (Risk Score vs Sprint)**")
fig_anom = px.scatter(
    history_df,
    x='sprint_id',
    y='risk_score',
    color='anomaly',
    color_discrete_map={True: 'red', False: 'blue'},
    labels={'sprint_id': 'Sprint', 'risk_score': 'Risk Score', 'anomaly': 'Anomaly'}
)
st.plotly_chart(fig_anom, use_container_width=True)

st.markdown("**Top anomalous sprints**")
top_anom = history_df.sort_values('anomaly_score', ascending=False).head(5)[
    ['sprint_id', 'risk_score', 'anomaly_score', 'critical_bugs', 'blocked', 'total_bugs']
]
st.dataframe(top_anom.reset_index(drop=True))

# ======================
# Distributions
# ======================
st.markdown("---")
st.subheader("Distributions")

st.markdown("**Distribution of Risk Scores**")
fig1, ax1 = plt.subplots()
ax1.hist(history_df['risk_score'], bins=10)
st.pyplot(fig1)

st.markdown("**Box Plot of Key Metrics**")
df_melt = history_df[['critical_bugs', 'blocked', 'pct_on_estimate']].melt(
    var_name='metric', value_name='value'
)
fig2 = px.box(df_melt, x='metric', y='value')
st.plotly_chart(fig2, use_container_width=True)

# ======================
# Correlation Heatmap
# ======================
st.markdown("---")
st.subheader("Correlation Heatmap")
fig3, ax3 = plt.subplots()
corr_df = history_df.select_dtypes(include=[np.number]).corr()
sns.heatmap(corr_df, annot=True, ax=ax3)
st.pyplot(fig3)

# ======================
# Metric Breakdown with Risk Level & Overage
# ======================
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

    overage_label = f"{overage_pct:.1f}% above historical max" if overage_pct > 0 else "Within historical range"

    st.write(
        f"**{label}:** {v}  \n"
        f"• Risk Level: **{risk_cat}**  \n"
        f"• Overage: **{overage_label}**"
    )