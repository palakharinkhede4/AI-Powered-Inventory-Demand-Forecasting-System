import pandas as pd
from prophet import Prophet
import pickle
import os

def train():
    df = pd.read_csv("data/processed/processed.csv")

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.1,
        seasonality_mode='multiplicative'
    )

    model.add_regressor('onpromotion')

    model.fit(df)

    os.makedirs("models", exist_ok=True)

    with open("models/prophet.pkl", "wb") as f:
        pickle.dump(model, f)

    print("✅ Model trained and saved")

if __name__ == "__main__":
    train()