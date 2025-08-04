# data/generate_data.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_tasks(n_sprints: int = 20) -> pd.DataFrame:
    """
    Генерує синтетичні дані задач для кількох спринтів з випадковою кількістю задач.

    Параметри:
    - n_sprints: кількість спринтів

    Повертає:
    - DataFrame з полями:
        sprint_id, task_id, estimation_days, actual_days,
        created, start, end, priority, status, reopened_count, is_bug
    """
    np.random.seed(42)
    all_data = []

    for sprint in range(1, n_sprints + 1):
        # Випадкова кількість задач у цьому спринті від 150 до 300
        tasks_count = np.random.randint(150, 301)
        # Початок спринту: відступ у днях від поточної дати
        sprint_start = datetime.now() - timedelta(days=14 * (n_sprints - sprint + 1))

        for i in range(tasks_count):
            estimation = np.random.randint(1, 8)  # очікувані дні
            actual = max(1, int(np.random.normal(loc=estimation, scale=2)))
            created_offset = np.random.randint(0, 7)  # дні після старту спринту
            created = sprint_start + timedelta(days=created_offset)
            start = created + timedelta(days=np.random.randint(0, 2))
            end = start + timedelta(days=actual)

            priority = np.random.choice(
                ['Low', 'Medium', 'High', 'Critical'],
                p=[0.4, 0.3, 0.2, 0.1]
            )
            status = np.random.choice(
                ['Done', 'In Progress', 'Blocked', 'Reopened'],
                p=[0.6, 0.2, 0.1, 0.1]
            )
            reopened = np.random.poisson(0.2)
            is_bug = np.random.rand() < 0.3

            all_data.append({
                'sprint_id': sprint,
                'task_id': f'S{str(sprint).zfill(2)}T{i+1}',
                'estimation_days': estimation,
                'actual_days': actual,
                'created': created,
                'start': start,
                'end': end,
                'priority': priority,
                'status': status,
                'reopened_count': reopened,
                'is_bug': is_bug
            })

    return pd.DataFrame(all_data)

if __name__ == '__main__':
    # Генеруємо дані для 20 спринтів
    df = generate_tasks(n_sprints=20)
    df.to_csv('data/tasks.csv', index=False)
    print(f"Generated data/tasks.csv with {len(df)} records across {df['sprint_id'].nunique()} sprints")