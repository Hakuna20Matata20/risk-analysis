# src/historical_metrics.py
import pandas as pd
from src.metrics import compute_metrics, load_data

def build_history(input_csv: str = 'data/tasks.csv',
                  output_csv: str = 'data/history.csv'):
    df = load_data(input_csv)
    history = []

    # проходим по каждому спринту
    for sprint, group in df.groupby('sprint_id'):
        mets = compute_metrics(group)
        mets['sprint_id'] = sprint
        history.append(mets)

    hist_df = pd.DataFrame(history).sort_values('sprint_id')
    hist_df.to_csv(output_csv, index=False)
    print(f"Saved history to {output_csv}")

if __name__ == '__main__':
    build_history()