# src/train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
from joblib import dump
from src.risk_score import compute_risk_score

def load_history(path: str = 'data/history.csv') -> pd.DataFrame:
    """Завантажує історичні метрики спринтів."""
    return pd.read_csv(path)

def prepare_data(df: pd.DataFrame):
    """
    Готує X та y для тренування:
    - обчислює risk_score по кожному рядку
    - ставить label=1, якщо risk_score >= threshold, інакше 0
    - повертає X (фічі) та y (мітки)
    """
    # Фічі, які використовуються для розрахунку risk_score і як X
    feature_cols = [
        'total_tasks', 'critical_bugs', 'blocked',
        'avg_dev_days', 'added_mid_sprint', 'avg_duration_days',
        'pct_on_estimate', 'total_bugs', 'total_reopened'
    ]

    # 1) Розрахуємо межі для нормалізації прямо по цій історії
    bounds = {col: (df[col].min(), df[col].max()) for col in feature_cols}

    # 2) Для кожного спринту обчислюємо risk_score
    df['risk_score'] = df.apply(
        lambda row: compute_risk_score(
            {col: row[col] for col in feature_cols},
            bounds
        ),
        axis=1
    )

    # 3) Генеруємо label: 1 якщо ризик високий (threshold=0.6), інакше 0
    threshold = 0.6
    df['label'] = (df['risk_score'] >= threshold).astype(int)

    # 4) Підготовка X та y
    X = df[feature_cols]
    y = df['label']
    return X, y

def train():
    # 1) Завантажуємо історію
    df = load_history('data/history.csv')

    # 2) Готуємо дані
    X, y = prepare_data(df)

    # 3) Розбиваємо на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 4) Шукаємо гіперпараметри для XGBoost
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
    }
    grid = GridSearchCV(
        XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        param_grid,
        cv=3,
        scoring='roc_auc',
        n_jobs=-1
    )
    grid.fit(X_train, y_train)

    # 5) Виводимо результати
    print("Best params:", grid.best_params_)
    print("Train AUC:", grid.best_score_)
    print("Test AUC:", grid.score(X_test, y_test))

    # 6) Зберігаємо модель
    dump(grid.best_estimator_, 'models/xgb_risk_model.joblib')
    print("Model saved to models/xgb_risk_model.joblib")

if __name__ == '__main__':
    train()