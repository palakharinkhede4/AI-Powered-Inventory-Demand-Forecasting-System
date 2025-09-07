# %% [markdown]
# # AI Inventory & Demand Forecasting — LightGBM/XGBoost (Tuned)
#
# **What’s new vs. Prophet version**
# - Gradient boosting models (LightGBM/XGBoost) with **hyperparameter tuning**
# - **TimeSeriesSplit** CV and **WAPE** scorer (robust to zeros)
# - Lags/rolling stats + calendar + exogenous (price/discount/competitor/weather/promo)
# - 365-day **recursive simulation** to keep lags consistent
# - Inventory policy: EOQ, Safety Stock (from residual risk), ROP
#
# **Edit these first:** DATA_PATH, STORE_ID, PRODUCT_ID, cost parameters.

# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV

plt.rcParams["figure.figsize"] = (11, 5)
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 100)

# Try to import LightGBM and XGBoost
try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

print(f"LightGBM available: {HAS_LGBM} | XGBoost available: {HAS_XGB}")

# %%
# --------------------
# CONFIG — EDIT ME
# --------------------
DATA_PATH = r"C:\Users\palak\OneDrive\Desktop\Virtual Env\retail_store_inventory.csv"

STORE_ID = "S001"
PRODUCT_ID = "P0001"

# Horizon/validation
FORECAST_DAYS = 365
TEST_DAYS = 90          # size of final holdout for validation

# Inventory policy params (example values — adapt to your business)
ORDERING_COST = 50.0    # per order
PRODUCT_COST = 15.0     # per unit
HOLDING_COST_PERCENT = 0.20
ANNUAL_HOLDING_COST = PRODUCT_COST * HOLDING_COST_PERCENT
LEAD_TIME_DAYS = 14
LEAD_TIME_STD_DEV = 2
Z_SCORE = 1.65          # ~95% service level

# Future exogenous assumptions during simulation
ASSUME_FUTURE_PROMO = 0               # 0/1 or craft a schedule
CARRY_FORWARD_PRICE = True
CARRY_FORWARD_COMP_PRICE = True
CARRY_FORWARD_DISCOUNT = True

# Random seed
RNG = 42

# %%
# --------------------
# METRICS
# --------------------
def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.sum(np.abs(y_true))
    if denom <= 1e-8:
        return np.nan
    return np.sum(np.abs(y_true - y_pred)) / denom

def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred))
    denom[denom == 0] = 1.0
    return np.mean(2.0 * np.abs(y_pred - y_true) / denom)

wape_scorer = make_scorer(lambda yt, yp: -wape(yt, yp), greater_is_better=True)  # maximize negative WAPE

# %%
# --------------------
# LOAD + FILTER
# --------------------
def load_and_filter(path: str, store_id: str, product_id: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found at: {path}")
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[(df["Store ID"] == store_id) & (df["Product ID"] == product_id)].copy()
    df = df.sort_values("Date").reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No rows for Store {store_id} and Product {product_id}.")
    return df

df = load_and_filter(DATA_PATH, STORE_ID, PRODUCT_ID)
print(df.head())
print(df[["Date","Units Sold","Price","Discount","Holiday/Promotion","Competitor Pricing","Weather Condition"]].describe(include="all"))

# %%
# --------------------
# FEATURE ENGINEERING
# --------------------
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["dow"] = out["Date"].dt.dayofweek
    out["week"] = out["Date"].dt.isocalendar().week.astype(int)
    out["month"] = out["Date"].dt.month
    out["year"] = out["Date"].dt.year
    out["is_weekend"] = (out["dow"] >= 5).astype(int)
    return out

def add_lags_and_rolls(df: pd.DataFrame,
                       target_col: str = "Units Sold",
                       lags: List[int] = [1,7,14,28],
                       rolls: List[int] = [7,14,28]) -> pd.DataFrame:
    out = df.copy()
    for L in lags:
        out[f"lag_{L}"] = out[target_col].shift(L)
    for R in rolls:
        out[f"rmean_{R}"] = out[target_col].shift(1).rolling(R).mean()
        out[f"rstd_{R}"]  = out[target_col].shift(1).rolling(R).std(ddof=0)
    return out

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    # keep relevant columns
    use_cols = ["Date", "Units Sold", "Price", "Discount", "Competitor Pricing",
                "Holiday/Promotion", "Weather Condition", "Seasonality"]
    x = df[use_cols].copy()
    x = add_time_features(x)
    x = add_lags_and_rolls(x, target_col="Units Sold")
    x = x.dropna().reset_index(drop=True)   # drop initial NaNs from lags/rolls
    return x

fe_df = build_features(df)
fe_df.head()

# %%
# --------------------
# TRAIN/TEST SPLIT
# --------------------
assert len(fe_df) > TEST_DAYS + 30, "Not enough history for chosen TEST_DAYS. Reduce TEST_DAYS or use more data."

train = fe_df.iloc[:-TEST_DAYS].copy()
test  = fe_df.iloc[-TEST_DAYS:].copy()

TARGET = "Units Sold"
CATEGORICAL = ["Weather Condition", "Seasonality", "dow", "month", "is_weekend"]
FEATURES = [c for c in fe_df.columns if c not in ["Date", TARGET]]

X_train = train[FEATURES].copy()
y_train = train[TARGET].values
X_test  = test[FEATURES].copy()
y_test  = test[TARGET].values

# Preprocess (OHE for categorical)
pre = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ("num", "passthrough", [c for c in FEATURES if c not in CATEGORICAL]),
    ],
    remainder="drop"
)

# %%
# --------------------
# HYPERPARAMETER TUNING (TimeSeriesSplit)
# --------------------
def tune_lgbm(X, y, preproc, random_state=RNG):
    if not HAS_LGBM:
        return None, None
    model = LGBMRegressor(
        objective="regression",
        random_state=random_state,
        n_estimators=800,
        verbose=-1
    )
    param_grid = {
        "hgb__num_leaves": [31, 63, 127, 255],
        "hgb__learning_rate": [0.03, 0.05, 0.08, 0.12],
        "hgb__feature_fraction": [0.7, 0.8, 0.9, 1.0],
        "hgb__bagging_fraction": [0.7, 0.8, 0.9, 1.0],
        "hgb__bagging_freq": [0, 1, 5],
        "hgb__min_child_samples": [10, 20, 30, 50],
        "hgb__lambda_l1": [0.0, 0.1, 0.5],
        "hgb__lambda_l2": [0.0, 0.1, 0.5]
    }
    pipe = Pipeline([("prep", preproc), ("hgb", model)])
    tscv = TimeSeriesSplit(n_splits=5)
    search = RandomizedSearchCV(
        pipe, param_distributions=param_grid,
        n_iter=30, scoring=wape_scorer, cv=tscv, n_jobs=-1, random_state=random_state, verbose=1
    )
    search.fit(X, y)
    return search.best_estimator_, search.best_score_

def tune_xgb(X, y, preproc, random_state=RNG):
    if not HAS_XGB:
        return None, None
    model = XGBRegressor(
        objective="reg:squarederror",
        random_state=random_state,
        n_estimators=1200,
        tree_method="hist",
        verbosity=0
    )
    param_grid = {
        "hgb__max_depth": [3, 4, 5, 6, 8],
        "hgb__learning_rate": [0.03, 0.05, 0.08, 0.12],
        "hgb__subsample": [0.6, 0.8, 1.0],
        "hgb__colsample_bytree": [0.6, 0.8, 1.0],
        "hgb__min_child_weight": [1, 3, 5, 7],
        "hgb__gamma": [0.0, 0.05, 0.1, 0.2],
        "hgb__reg_alpha": [0.0, 0.1, 0.5],
        "hgb__reg_lambda": [0.0, 0.1, 0.5]
    }
    pipe = Pipeline([("prep", preproc), ("hgb", model)])
    tscv = TimeSeriesSplit(n_splits=5)
    search = RandomizedSearchCV(
        pipe, param_distributions=param_grid,
        n_iter=30, scoring=wape_scorer, cv=tscv, n_jobs=-1, random_state=random_state, verbose=1
    )
    search.fit(X, y)
    return search.best_estimator_, search.best_score_

# %%
# Run tuning
best_model = None
best_cv = -np.inf
best_name = None

if HAS_LGBM:
    lgbm_est, lgbm_cv = tune_lgbm(X_train, y_train, pre)
    print(f"Best LightGBM CV (neg WAPE): {lgbm_cv:.4f}")
    if lgbm_est is not None and lgbm_cv > best_cv:
        best_model = lgbm_est
        best_cv = lgbm_cv
        best_name = "LightGBM"

if HAS_XGB:
    xgb_est, xgb_cv = tune_xgb(X_train, y_train, pre)
    print(f"Best XGBoost CV (neg WAPE): {xgb_cv:.4f}")
    if xgb_est is not None and xgb_cv > best_cv:
        best_model = xgb_est
        best_cv = xgb_cv
        best_name = "XGBoost"

if best_model is None:
    # Fallback: small LightGBM-like settings using RandomForest-ish values (no extra deps)
    from sklearn.ensemble import HistGradientBoostingRegressor
    print("Neither LightGBM nor XGBoost found — falling back to HistGradientBoostingRegressor.")
    best_model = Pipeline([
        ("prep", pre),
        ("hgb", HistGradientBoostingRegressor(
            learning_rate=0.08, max_iter=600, min_samples_leaf=25, random_state=RNG))
    ])
    best_name = "HistGBR"
    best_model.fit(X_train, y_train)

print(f"\nSelected model: {best_name}")

# %%
# --------------------
# VALIDATE ON HOLDOUT
# --------------------
best_model.fit(X_train, y_train)
pred_test = best_model.predict(X_test)

val_wape  = wape(y_test, pred_test)
val_smape = smape(y_test, pred_test)
val_mae   = mean_absolute_error(y_test, pred_test)

print(f"Validation — WAPE: {val_wape:.3f}  |  sMAPE: {val_smape:.3f}  |  MAE: {val_mae:.2f}  |  Model: {best_name}")

plt.figure()
plt.plot(test["Date"], y_test, label="Actual")
plt.plot(test["Date"], pred_test, label="Predicted")
plt.title(f"Validation — Actual vs Predicted ({best_name})")
plt.xlabel("Date"); plt.ylabel("Units Sold"); plt.legend(); plt.tight_layout(); plt.show()

# capture residual std for safety stock
residuals = y_test - pred_test
resid_std = float(np.std(residuals, ddof=0))
print(f"Residual std (validation): {resid_std:.2f}")

# %%
# --------------------
# FUTURE SIMULATION (365d)
# --------------------
def simulate_future(df_full: pd.DataFrame,
                    fe_full: pd.DataFrame,
                    pipeline: Pipeline,
                    horizon_days: int = 365) -> pd.DataFrame:
    """
    Recursive daily simulation to keep lag/rolling features coherent.
    We carry forward exogenous values unless user provides a promo/price plan.
    """
    hist = df_full.copy().sort_values("Date").reset_index(drop=True)
    last_date = hist["Date"].iloc[-1]

    demand_series = list(hist["Units Sold"].values)

    # baselines for exogenous
    last_price = float(hist["Price"].iloc[-1])
    last_comp  = float(hist["Competitor Pricing"].iloc[-1])
    last_disc  = float(hist["Discount"].iloc[-1])

    preds = []

    for d in range(1, horizon_days + 1):
        cur_date = last_date + pd.Timedelta(days=d)

        row = {
            "Date": cur_date,
            "Price": last_price if CARRY_FORWARD_PRICE else float(hist["Price"].iloc[-(d % 30 or 1)]),
            "Discount": last_disc if CARRY_FORWARD_DISCOUNT else float(hist["Discount"].iloc[-(d % 30 or 1)]),
            "Competitor Pricing": last_comp if CARRY_FORWARD_COMP_PRICE else float(hist["Competitor Pricing"].iloc[-(d % 30 or 1)]),
            "Holiday/Promotion": ASSUME_FUTURE_PROMO,
            "Weather Condition": hist["Weather Condition"].iloc[-1],   # baseline
            "Seasonality": hist["Seasonality"].iloc[-1],               # baseline
        }
        tmp = pd.DataFrame([row])
        # add same time/lag/roll features
        tmp = add_time_features(tmp)

        for L in [1,7,14,28]:
            tmp[f"lag_{L}"] = demand_series[-L] if len(demand_series) >= L else (demand_series[-1] if demand_series else 0)
        for R in [7,14,28]:
            history = pd.Series(demand_series)
            tmp[f"rmean_{R}"] = history.rolling(R).mean().iloc[-1] if len(history) >= R else history.mean()
            tmp[f"rstd_{R}"]  = history.rolling(R).std(ddof=0).iloc[-1] if len(history) >= R else history.std(ddof=0)

        # order columns to match training FEATURES
        tmp = tmp[FEATURES]

        yhat = float(pipeline.predict(tmp)[0])
        yhat = max(yhat, 0.0)
        preds.append({"ds": cur_date, "yhat": yhat})

        demand_series.append(yhat)

    return pd.DataFrame(preds)

future_forecast = simulate_future(df, fe_df, best_model, FORECAST_DAYS)
print(future_forecast.head(), future_forecast.tail(), sep="\n\n")

plt.figure()
plt.plot(future_forecast["ds"], future_forecast["yhat"])
plt.title(f"{FORECAST_DAYS}-Day Forecast — {best_name}")
plt.xlabel("Date"); plt.ylabel("Units Sold")
plt.tight_layout(); plt.show()

# %%
# --------------------
# INVENTORY POLICY
# --------------------
def calculate_eoq(annual_demand: float, ordering_cost: float, holding_cost: float) -> float:
    if annual_demand <= 0 or holding_cost <= 0:
        return 0.0
    return float(np.sqrt((2 * annual_demand * ordering_cost) / holding_cost))

def calculate_rop_from_residuals(future_df: pd.DataFrame,
                                 lead_time_days: int,
                                 z_score: float,
                                 residual_std: float) -> Tuple[int, int]:
    """Mean demand from forecast; std from validation residuals as proxy for uncertainty."""
    mean_d = float(future_df["yhat"].mean())
    demand_during_lead = mean_d * lead_time_days
    safety_stock = z_score * residual_std * np.sqrt(lead_time_days)
    rop = demand_during_lead + safety_stock
    return int(np.ceil(rop)), int(np.ceil(safety_stock))

annual_demand = float(future_forecast["yhat"].sum())
eoq = calculate_eoq(annual_demand, ORDERING_COST, ANNUAL_HOLDING_COST)
rop, safety_stock = calculate_rop_from_residuals(future_forecast, LEAD_TIME_DAYS, Z_SCORE, resid_std)

os.makedirs("outputs", exist_ok=True)
future_forecast.to_csv("outputs/boosted_forecast.csv", index=False)

print("\n--- RESULTS ---")
print(f"Model selected: {best_name}  |  CV (neg WAPE): {best_cv:.4f}")
print(f"Validation — WAPE: {val_wape:.3f} | sMAPE: {val_smape:.3f} | MAE: {val_mae:.2f}")
print(f"Forecast horizon: {FORECAST_DAYS} days")
print(f"Annual Demand (sum forecast): {annual_demand:.0f}")
print(f"Safety Stock: {safety_stock} units")
print(f"Reorder Point (ROP): {rop} units")
print(f"Economic Order Quantity (EOQ): {int(np.ceil(eoq))} units")
print(f"Action: When inventory for Product {PRODUCT_ID} at Store {STORE_ID} drops to {rop}, order {int(np.ceil(eoq))} units.")
