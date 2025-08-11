# src/anomaly.py
import pandas as pd
from sklearn.ensemble import IsolationForest

FEATURE_COLS = [
    'total_tasks', 'critical_bugs', 'blocked',
    'avg_dev_days', 'added_mid_sprint', 'avg_duration_days',
    'pct_on_estimate', 'total_bugs', 'total_reopened'
]

def detect_anomalies(history_csv: str = 'data/history.csv',
                     contamination: float = 0.1,
                     random_state: int = 42) -> pd.DataFrame:
    """
    Повертає history з колонками:
      - anomaly (True/False)
      - anomaly_score (чим більше, тим "більш аномально")
    """
    df = pd.read_csv(history_csv)
    X = df[FEATURE_COLS]

    iso = IsolationForest(
        contamination=contamination,
        random_state=random_state
    )
    iso.fit(X)
    # -1 = аномалія, 1 = норм
    labels = iso.predict(X)
    scores = -iso.decision_function(X)  # вищий бал = гірше

    df['anomaly'] = (labels == -1)
    df['anomaly_score'] = scores
    return df[['sprint_id', 'anomaly', 'anomaly_score']]