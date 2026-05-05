import pickle
import pandas as pd
import numpy as np

def predict(days=30):
    with open("models/prophet.pkl", "rb") as f:
        model = pickle.load(f)

    # Load processed data
    df = pd.read_csv("data/processed/processed.csv")

    future = model.make_future_dataframe(periods=days)

    # Use last known promotion pattern
    last_promos = df['onpromotion'].tail(days).values
    future['onpromotion'] = list(df['onpromotion']) + list(last_promos)

    forecast = model.predict(future)

    result = forecast[['ds', 'yhat']].tail(days)

    # 🔥 Reverse log transform
    result['yhat'] = np.expm1(result['yhat'])

    return result

if __name__ == "__main__":
    result = predict(30)
    print(result.head())