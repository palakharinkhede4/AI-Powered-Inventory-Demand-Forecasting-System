import pandas as pd
import numpy as np
import os

def load_data():
    return pd.read_csv(os.path.join("data", "raw", "train.csv"))

def preprocess(df):
    df['date'] = pd.to_datetime(df['date'])

    # Use one store (cleaner signal)
    df = df[df['store_nbr'] == 1]

    # Aggregate
    df = df.groupby('date').agg({
        'sales': 'sum',
        'onpromotion': 'sum'
    }).reset_index()

    df = df.rename(columns={
        'date': 'ds',
        'sales': 'y'
    })

    # 🔥 LOG TRANSFORM (IMPORTANT)
    df['y'] = np.log1p(df['y'])

    return df

def save(df):
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/processed.csv", index=False)

if __name__ == "__main__":
    df = load_data()
    df = preprocess(df)
    save(df)
    print("✅ Preprocessing done")