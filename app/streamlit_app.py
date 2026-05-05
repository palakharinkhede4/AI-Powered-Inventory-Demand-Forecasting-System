import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Inventory functions
def calculate_eoq(D, S, H):
    return np.sqrt((2 * D * S) / H)

def reorder_point(daily_demand, lead_time, safety_stock):
    return (daily_demand * lead_time) + safety_stock

# UI config
st.set_page_config(page_title="Inventory Optimization System", layout="wide")

st.title("📦 AI Inventory & Demand Forecasting System")

st.markdown("""
### 🎯 Problem
Businesses often face:
- Overstocking → high storage cost  
- Understocking → lost sales  

### 💡 Solution
This system predicts future demand and recommends optimal inventory decisions.
""")

# Load model
with open("models/prophet.pkl", "rb") as f:
    model = pickle.load(f)

df = pd.read_csv("data/processed/processed.csv")

# ---------------- INPUT SECTION ----------------
st.sidebar.header("⚙️ Input Parameters")

days = st.sidebar.slider("Forecast Days", 7, 60, 30)

current_stock = st.sidebar.number_input("Current Inventory", value=5000)

lead_time = st.sidebar.number_input("Lead Time (days)", value=5)

ordering_cost = st.sidebar.number_input("Ordering Cost (₹)", value=50)

holding_cost = st.sidebar.number_input("Holding Cost per Unit (₹)", value=2)

safety_stock = st.sidebar.number_input("Safety Stock", value=500)

# ---------------- FORECAST ----------------
if st.button("🚀 Run Inventory Optimization"):

    future = model.make_future_dataframe(periods=days)

    last_promos = df['onpromotion'].tail(days).values
    future['onpromotion'] = list(df['onpromotion']) + list(last_promos)

    forecast = model.predict(future)

    result = forecast[['ds', 'yhat']].tail(days)
    result['yhat'] = np.expm1(result['yhat'])

    # ---------------- DEMAND ----------------
    total_demand = result['yhat'].sum()
    avg_daily_demand = result['yhat'].mean()

    # ---------------- INVENTORY ----------------
    eoq = calculate_eoq(total_demand, ordering_cost, holding_cost)
    rop = reorder_point(avg_daily_demand, lead_time, safety_stock)

    # ---------------- OUTPUT ----------------
    st.subheader("📊 Demand Forecast")
    st.line_chart(result.set_index('ds'))

    st.write("### 📈 Forecast Summary")
    st.write(f"**Total Expected Demand:** {int(total_demand)} units")
    st.write(f"**Average Daily Demand:** {int(avg_daily_demand)} units")

    st.write("### 📦 Inventory Recommendations")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Optimal Order Quantity (EOQ)", int(eoq))

    with col2:
        st.metric("Reorder Point (ROP)", int(rop))

    # ---------------- STATUS ----------------
    st.write("### ⚠️ Inventory Status")

    if current_stock <= rop:
        st.error("🔴 Reorder Now! Stock is below reorder level.")
    else:
        st.success("🟢 Stock is sufficient.")

    st.write(f"Current Stock: {current_stock}")
    st.write(f"Reorder Level: {int(rop)}")

    # ---------------- DATA TABLE ----------------
    st.write("### 📋 Forecast Data")
    st.dataframe(result)