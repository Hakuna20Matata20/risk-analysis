# src/risk_score.py

import pandas as pd
from src.metrics import compute_metrics

# Визначте ваги для кожної метрики (сума = 1)
WEIGHTS = {
    'critical_bugs':    0.18,
    'blocked':          0.20,
    'avg_dev_days':     0.15,
    'total_reopened':   0.12,
    'added_mid_sprint': 0.10,
    'avg_duration_days':0.08,
    'pct_on_estimate':  0.07,
    'total_bugs':       0.06,
    'total_tasks':      0.04
}

def normalize(value, min_v, max_v):
    """
    Повертає нормалізоване значення value у діапазоні [0,1]
    за формулою (value - min) / (max - min).
    Якщо max == min, повертає 0.
    """
    if max_v > min_v:
        return (value - min_v) / (max_v - min_v)
    return 0.0

def compute_risk_score(metrics: dict, bounds: dict) -> float:
    """
    Обчислює композитний Risk Score як зважену суму нормалізованих метрик.
    - metrics: словник {назва_метрики: її_значення}
    - bounds:  словник {назва_метрики: (min, max)} по історії
    Повертає float із трьома знаками після коми.
    """
    score = 0.0
    for key, weight in WEIGHTS.items():
        min_v, max_v = bounds[key]
        norm = normalize(metrics[key], min_v, max_v)
        score += weight * norm
    return round(score, 3)

def load_bounds(history_csv: str = 'data/history.csv') -> dict:
    """
    Зчитує історичні метрики з CSV і повертає словник
    {назва_метрики: (min, max)} для кожної колонки (окрім sprint_id).
    """
    hist = pd.read_csv(history_csv)
    bounds = {}
    for col in hist.columns:
        if col == 'sprint_id':
            continue
        bounds[col] = (hist[col].min(), hist[col].max())
    return bounds

if __name__ == '__main__':
    # Імпортуємо завантажувач даних тільки в режимі прямого запуску
    from src.metrics import load_data

    # 1) Завантажуємо останній набір задач
    df = load_data('data/tasks.csv')

    # 2) Обчислюємо метрики для поточного спринту
    mets = compute_metrics(df)
    print("Metrics:", mets)

    # 3) Підвантажуємо історичні межі для нормалізації
    bounds = load_bounds('data/history.csv')
    print("Bounds:", bounds)

    # 4) Обчислюємо та виводимо Risk Score
    risk = compute_risk_score(mets, bounds)
    print("Risk Score:", risk)