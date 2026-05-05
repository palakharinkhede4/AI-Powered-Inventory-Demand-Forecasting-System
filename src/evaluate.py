import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error

def evaluate():
    df = pd.read_csv("data/processed/processed.csv")

    df['ds'] = pd.to_datetime(df['ds'])

    train = df[:-30]
    test = df[-30:]

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.1,
        seasonality_mode='multiplicative'
    )

    model.add_regressor('onpromotion')

    model.fit(train)

    future = model.make_future_dataframe(periods=30)

    # 🔥 Use real promo values
    future['onpromotion'] = list(train['onpromotion']) + list(test['onpromotion'])

    forecast = model.predict(future)

    pred = forecast[['ds', 'yhat']].tail(30)

    pred['ds'] = pd.to_datetime(pred['ds'])
    test['ds'] = pd.to_datetime(test['ds'])

    merged = test.merge(pred, on='ds')

    # 🔥 Reverse log transform
    y_true = np.expm1(merged['y'])
    y_pred = np.expm1(merged['yhat'])

    mape = mean_absolute_percentage_error(y_true, y_pred)
    accuracy = (1 - mape) * 100

    print(f"MAPE: {mape:.4f}")
    print(f"Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    evaluate()