# src/metrics.py
import pandas as pd

def load_data(path: str = 'data/tasks.csv') -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=['created','start','end'])

def compute_metrics(df: pd.DataFrame) -> dict:
    total = len(df)
    critical_bugs = df[(df['is_bug']) & (df['priority'].isin(['High','Critical']))].shape[0]
    blocked = df[(df['status']=='Blocked') | (df['status']=='blocked')].shape[0]
    avg_dev = abs(df['estimation_days'] - df['actual_days']).mean()
    added_mid = df[df['created'] > df['created'].min()].shape[0]
    avg_duration = df['actual_days'].mean()
    pct_in_est = (df[df['actual_days'] <= df['estimation_days']].shape[0] / total) * 100
    bugs = df[df['is_bug']==True].shape[0]
    reopened = df['reopened_count'].sum()

    return {
        'total_tasks': total,
        'critical_bugs': critical_bugs,
        'blocked': blocked,
        'avg_dev_days': avg_dev,
        'added_mid_sprint': added_mid,
        'avg_duration_days': avg_duration,
        'pct_on_estimate': pct_in_est,
        'total_bugs': bugs,
        'total_reopened': reopened
    }