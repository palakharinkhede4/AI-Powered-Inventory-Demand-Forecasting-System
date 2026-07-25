import pickle
import pandas as pd
import numpy as np
import os

MODEL_PATH = os.path.join("models", "prophet.pkl")
DATA_PATH = os.path.join("data", "processed", "processed.csv")

def load_model_and_data():
    if not os.path.exists(MODEL_PATH):
        print(f"⚠️ Model file not found at {MODEL_PATH}. Auto-training model...")
        from src.train_model import train
        train()
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Processed data file not found at {DATA_PATH}. Run 'python src/preprocess.py' first.")

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    
    df = pd.read_csv(DATA_PATH)
    df['ds'] = pd.to_datetime(df['ds'])
def get_historical_dataframe():
    """Returns the complete un-logged historical sales dataset."""
    _, df = load_model_and_data()
    hist = df.copy()
    hist['sales'] = np.expm1(hist['y']).round(2)
    hist['date'] = hist['ds'].dt.strftime('%Y-%m-%d')
    cols = ['date', 'sales']
    if 'onpromotion' in hist.columns:
        cols.append('onpromotion')
    return hist[cols]

def generate_forecast(days: int = 30, promo_boost_pct: float = 0.0):
    """Generates demand forecast with uncertainty intervals and seasonal components.
    
    Args:
        days: Horizon of days to forecast.
        promo_boost_pct: Optional percentage modifier for future promotions (e.g. +10% promo boost).
    """
    model, df = load_model_and_data()

    # Create future dataframe
    future = model.make_future_dataframe(periods=days)

    # Handle regressor 'onpromotion'
    if 'onpromotion' in df.columns:
        last_promos = df['onpromotion'].tail(days).values
        # Apply promo boost if specified
        future_promos = last_promos * (1.0 + promo_boost_pct / 100.0)
        future['onpromotion'] = list(df['onpromotion']) + list(future_promos)

    # Model prediction
    forecast = model.predict(future)

    # Exponentiate predictions (reversing log1p transform)
    forecast_cols = ['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend']
    if 'weekly' in forecast.columns:
        forecast_cols.append('weekly')
        
    res = forecast[forecast_cols].copy()
    
    # Clip negative values & reverse log1p for demand fields
    for col in ['yhat', 'yhat_lower', 'yhat_upper']:
        res[col] = np.clip(np.expm1(res[col]), 0, None)
        
    res['trend'] = np.clip(np.expm1(res['trend']), 0, None)

    # Format dates as string YYYY-MM-DD
    res['ds'] = pd.to_datetime(res['ds']).dt.strftime('%Y-%m-%d')
    
    # Historical actuals dataset (last 90 days for clean chart display)
    hist = df.copy()
    hist['y'] = np.expm1(hist['y'])
    hist['ds'] = hist['ds'].dt.strftime('%Y-%m-%d')
    
    historical_tail = hist[['ds', 'y']].tail(90)
    forecast_tail = res.tail(days)

    return {
        "historical": historical_tail.to_dict(orient="records"),
        "forecast": forecast_tail.to_dict(orient="records"),
        "full_forecast": res.to_dict(orient="records")
    }

if __name__ == "__main__":
    out = generate_forecast(14)
    print("Forecast samples:", out["forecast"][:3])