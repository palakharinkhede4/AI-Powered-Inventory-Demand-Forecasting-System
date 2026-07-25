import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import io
import os
import sys
from datetime import datetime, timedelta

# Ensure project root is in sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.forecast import generate_forecast
try:
    from src.forecast import get_historical_dataframe
except ImportError:
    get_historical_dataframe = None

from src.inventory import (
    calculate_eoq,
    reorder_point,
    classify_inventory_status,
    calculate_recommended_reorder
)
from src.report import create_purchase_order_excel

API_URL = "http://127.0.0.1:8000"

def load_historical_data():
    """Safely loads historical sales data with fallback."""
    if get_historical_dataframe is not None:
        try:
            return get_historical_dataframe()
        except Exception:
            pass
    # Fallback directly to CSV
    data_path = os.path.join(ROOT_DIR, "data", "processed", "processed.csv")
    df = pd.read_csv(data_path)
    df['sales'] = np.expm1(df['y']).round(2)
    df['date'] = pd.to_datetime(df['ds']).dt.strftime('%Y-%m-%d')
    cols = ['date', 'sales']
    if 'onpromotion' in df.columns:
        cols.append('onpromotion')
    return df[cols]

st.set_page_config(
    page_title="AI Demand Forecasting & Inventory System",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Plus Jakarta Sans font + Clean modern UI cards)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    
    .sub-header {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-bottom: 1.5rem;
    }

    .kpi-card {
        background-color: #1e293b;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #334155;
        text-align: center;
    }
    
    .kpi-title {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 8px;
    }
    
    .kpi-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f8fafc;
    }
    
    .kpi-sub {
        font-size: 0.8rem;
        color: #38bdf8;
        margin-top: 4px;
    }

    .status-banner-red {
        background-color: #451a1a;
        border-left: 6px solid #ef4444;
        padding: 18px;
        border-radius: 10px;
        color: #fca5a5;
        margin-bottom: 20px;
    }
    .status-banner-yellow {
        background-color: #422006;
        border-left: 6px solid #f59e0b;
        padding: 18px;
        border-radius: 10px;
        color: #fde68a;
        margin-bottom: 20px;
    }
    .status-banner-green {
        background-color: #064e3b;
        border-left: 6px solid #10b981;
        padding: 18px;
        border-radius: 10px;
        color: #a7f3d0;
        margin-bottom: 20px;
    }
    .status-banner-blue {
        background-color: #172554;
        border-left: 6px solid #3b82f6;
        padding: 18px;
        border-radius: 10px;
        color: #bfdbfe;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="main-header">📦 AI Inventory & Demand Forecasting Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Predict future sales demand, prevent costly stockouts, and automate supplier order replenishment.</div>', unsafe_allow_html=True)

# Real-World Value Expander
with st.expander("💡 Why Use This System? (Real-World Business Impact)"):
    st.markdown("""
    * **The Problem:** Global retailers lose over **$1.1 Trillion annually** due to overstocking (capital tied up in excess inventory and storage fees) and stockouts (lost sales revenue and customer churn).
    * **The Solution:** This system uses machine learning time-series forecasting to answer three vital daily operational questions:
      1. 📈 **How much will customers buy in the next 7 to 60 days?** (Demand Forecasting)
      2. ⚠️ **Do we have enough inventory, or when will we run out?** (Stockout Risk Alert)
      3. 📦 **How many units should we order right now to minimize holding & shipping costs?** (EOQ Reorder Calculation)
    """)

# ---------------- SIDEBAR (SIMPLIFIED TO 4 INPUTS) ----------------
st.sidebar.header("⚙️ Operational Controls")

days = st.sidebar.slider("🗓️ Forecast Horizon (Days)", min_value=7, max_value=60, value=30, step=1)
current_stock = st.sidebar.number_input("📦 Current Stock Level (Units)", min_value=0, value=5000, step=100)
lead_time = st.sidebar.number_input("🚚 Supplier Lead Time (Days)", min_value=1, max_value=30, value=5, step=1)

promo_choice = st.sidebar.selectbox(
    "📣 Promotional Event Scenario",
    options=["Normal Demand (No Promo)", "Moderate Promo (+15% Demand)", "Major Flash Sale (+35% Demand)"]
)

promo_boost_map = {
    "Normal Demand (No Promo)": 0.0,
    "Moderate Promo (+15% Demand)": 15.0,
    "Major Flash Sale (+35% Demand)": 35.0
}
promo_boost = promo_boost_map[promo_choice]

# Hidden Advanced Parameters in Expander (for power users)
with st.sidebar.expander("🛠️ Advanced Cost Parameters"):
    ordering_cost = st.number_input("Ordering Cost S (₹/order)", value=50.0, min_value=1.0)
    holding_cost = st.number_input("Holding Cost H (₹/unit/year)", value=2.0, min_value=0.1)
    safety_stock = st.number_input("Safety Stock Buffer (Units)", value=500, min_value=0)

# Check API health status
api_online = False
try:
    health_resp = requests.get(f"{API_URL}/health", timeout=1.0)
    if health_resp.status_code == 200:
        api_online = True
except Exception:
    api_online = False

# Fetch forecast data
if api_online:
    try:
        payload = {
            "days": days,
            "current_stock": current_stock,
            "lead_time": lead_time,
            "ordering_cost": ordering_cost,
            "holding_cost": holding_cost,
            "safety_stock": safety_stock,
            "promo_boost_pct": float(promo_boost)
        }
        res = requests.post(f"{API_URL}/api/v1/forecast", json=payload).json()
        metrics = res["metrics"]
        status_info = res["status_info"]
        hist_df = pd.DataFrame(res["historical"])
        fc_df = pd.DataFrame(res["forecast"])
    except Exception:
        api_online = False

if not api_online:
    raw_data = generate_forecast(days=days, promo_boost_pct=float(promo_boost))
    hist_df = pd.DataFrame(raw_data["historical"])
    fc_df = pd.DataFrame(raw_data["forecast"])
    
    yhat_vals = fc_df['yhat'].values
    tot_demand = float(np.sum(yhat_vals))
    avg_demand = float(np.mean(yhat_vals)) if len(yhat_vals) > 0 else 0.0
    
    eoq_val = calculate_eoq(tot_demand, ordering_cost, holding_cost)
    rop_val = reorder_point(avg_demand, lead_time, safety_stock)
    
    status_info = classify_inventory_status(current_stock, rop_val, eoq_val, safety_stock)
    reorder_qty = calculate_recommended_reorder(current_stock, rop_val, eoq_val)
    
    metrics = {
        "total_expected_demand": round(tot_demand, 2),
        "avg_daily_demand": round(avg_demand, 2),
        "eoq": round(eoq_val, 2),
        "rop": round(rop_val, 2),
        "recommended_reorder_qty": int(reorder_qty)
    }

# Calculate Days of Stock Remaining
avg_daily = metrics["avg_daily_demand"]
days_remaining = round(current_stock / avg_daily, 1) if avg_daily > 0 else 999
stockout_date = (datetime.now() + timedelta(days=days_remaining)).strftime("%b %d, %Y") if days_remaining < 365 else "N/A"

# ---------------- TOP KPI CARDS ----------------
k1, k2, k3, k4 = st.columns(4)

with k1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Projected {days}-Day Demand</div>
        <div class="kpi-value">{int(metrics['total_expected_demand']):,}</div>
        <div class="kpi-sub">Units needed</div>
    </div>
    """, unsafe_allow_html=True)

with k2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Avg Daily Sales Rate</div>
        <div class="kpi-value">{int(metrics['avg_daily_demand']):,}</div>
        <div class="kpi-sub">Units / day</div>
    </div>
    """, unsafe_allow_html=True)

with k3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Stock Runway</div>
        <div class="kpi-value">{days_remaining} Days</div>
        <div class="kpi-sub">Runs out ~ {stockout_date}</div>
    </div>
    """, unsafe_allow_html=True)

with k4:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Optimal Order Size (EOQ)</div>
        <div class="kpi-value">{int(metrics['eoq']):,}</div>
        <div class="kpi-sub">Units per batch</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# ---------------- INVENTORY HEALTH STATUS BANNER ----------------
level = status_info.get("level", "Green").lower()
icon = status_info.get("icon", "🟢")
title = status_info.get("title", "")
msg = status_info.get("message", "")
reorder_qty = metrics["recommended_reorder_qty"]

banner_class = f"status-banner-{level}"
st.markdown(f"""
    <div class="{banner_class}">
        <h3 style="margin:0; padding-bottom:6px; font-weight:700;">{icon} Inventory Status: {title.upper()}</h3>
        <p style="margin:0; padding-bottom:8px; font-size:0.95rem;">{msg}</p>
        <span style="font-weight:600;">Current Inventory: {current_stock:,} units | Reorder Threshold (ROP): {int(metrics['rop']):,} units | Recommended Reorder Order: {reorder_qty:,} units</span>
    </div>
""", unsafe_allow_html=True)

# ---------------- MAIN APP TABS ----------------
tab_forecast, tab_history, tab_po, tab_tech = st.tabs([
    "📈 Demand Forecast & Visuals",
    "📊 Historical Data Explorer",
    "📝 Purchase Order Generator",
    "🛠️ Technical Architecture"
])

# ----- TAB 1: DEMAND FORECAST -----
with tab_forecast:
    fig = go.Figure()

    # Historical Actual Sales
    if not hist_df.empty:
        fig.add_trace(go.Scatter(
            x=hist_df['ds'],
            y=hist_df['y'],
            mode='lines',
            name='Historical Sales',
            line=dict(color='#818cf8', width=2)
        ))

    # Uncertainty Upper Bound
    fig.add_trace(go.Scatter(
        x=fc_df['ds'],
        y=fc_df['yhat_upper'],
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))

    # Uncertainty Lower Bound & Shading
    fig.add_trace(go.Scatter(
        x=fc_df['ds'],
        y=fc_df['yhat_lower'],
        mode='lines',
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(56, 189, 248, 0.18)',
        name='95% Demand Uncertainty Range',
        hoverinfo='skip'
    ))

    # Forecast Centerline
    fig.add_trace(go.Scatter(
        x=fc_df['ds'],
        y=fc_df['yhat'],
        mode='lines+markers',
        name='Predicted Demand Centerline',
        line=dict(color='#34d399', width=3)
    ))

    # Daily Reorder Threshold Line (units/day)
    daily_rop = metrics['rop'] / lead_time if lead_time > 0 else metrics['avg_daily_demand']
    fig.add_trace(go.Scatter(
        x=list(hist_df['ds']) + list(fc_df['ds']),
        y=[daily_rop] * (len(hist_df) + len(fc_df)),
        mode='lines',
        name=f"Daily Reorder Baseline ({int(daily_rop):,} units/day)",
        line=dict(color='#f87171', width=2, dash='dash')
    ))

    fig.update_layout(
        title=f"{days}-Day Demand Forecast with 95% Confidence Bounds",
        xaxis_title="Date",
        yaxis_title="Units Sold / Demanded per Day",
        template="plotly_dark",
        hovermode="x unified",
        height=480,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"💡 **Chart Note:** The red dashed baseline represents the **Daily Reorder Rate** (`{int(daily_rop):,} units/day`). The cumulative **{int(lead_time)}-Day Total ROP Threshold** is `{int(metrics['rop']):,} units`.")

    with st.expander("📋 View Daily Forecast Breakdown Table"):
        display_df = fc_df.copy()
        for c in ['yhat', 'yhat_lower', 'yhat_upper']:
            if c in display_df.columns:
                display_df[c] = display_df[c].round(2)
        st.dataframe(display_df, use_container_width=True)

# ----- TAB 2: HISTORICAL DATA EXPLORER -----
with tab_history:
    st.subheader("📊 Training Dataset Explorer")
    st.write("Examine the historical daily retail store sales dataset used to train the demand forecasting model.")

    hist_full = load_historical_data()
    
    # Dataset Summary Metrics
    c_h1, c_h2, c_h3, c_h4 = st.columns(4)
    with c_h1:
        st.metric("Total Days Recorded", f"{len(hist_full):,} days")
    with c_h2:
        st.metric("Historical Date Range", f"{hist_full['date'].min()} to {hist_full['date'].max()}")
    with c_h3:
        st.metric("Average Daily Sales", f"{int(hist_full['sales'].mean()):,} units")
    with c_h4:
        st.metric("Peak Single-Day Sales", f"{int(hist_full['sales'].max()):,} units")

    st.write("### Historical Sales Records Table")
    st.dataframe(hist_full, use_container_width=True, height=350)

    # Download CSV button
    csv_data = hist_full.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Historical Dataset (CSV)",
        data=csv_data,
        file_name="historical_store_sales.csv",
        mime="text/csv",
        use_container_width=False
    )

# ----- TAB 3: PURCHASE ORDER GENERATOR -----
with tab_po:
    st.subheader("📝 Automated Supplier Purchase Order Generator")
    st.write("Generate a formal, supplier-ready Excel Purchase Order (.xlsx) based on current AI inventory recommendations.")

    po_c1, po_c2 = st.columns(2)
    with po_c1:
        vendor_name = st.text_input("Supplier / Vendor Name", value="Global Logistics Supply Ltd")
        item_name = st.text_input("Item / Product Name", value="Aggregated Retail SKU #1")
    with po_c2:
        unit_cost = st.number_input("Unit Purchase Price (₹)", value=25.0, min_value=0.1)
        est_total_cost = reorder_qty * unit_cost
        st.write(f"**Total Estimated Purchase Order Value:** `₹{est_total_cost:,.2f}`")

    st.divider()

    po_bytes = create_purchase_order_excel(
        vendor_name=vendor_name,
        item_name=item_name,
        current_stock=current_stock,
        rop=metrics['rop'],
        eoq=metrics['eoq'],
        recommended_qty=reorder_qty,
        unit_cost=unit_cost,
        avg_daily_demand=metrics['avg_daily_demand'],
        lead_time_days=int(lead_time)
    )

    st.download_button(
        label="📥 Download Official Purchase Order (.xlsx)",
        data=po_bytes,
        file_name=f"Purchase_Order_{vendor_name.replace(' ', '_')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

# ----- TAB 4: TECHNICAL ARCHITECTURE -----
with tab_tech:
    st.subheader("🛠️ Technical Architecture & System Flow")
    st.markdown("""
    ### 🏗️ Decoupled MLOps System Flow
    
    ```
    ┌─────────────────────────┐         HTTP REST          ┌──────────────────────────┐
    │   Streamlit Frontend    │ ────────────────────────► │   FastAPI Backend Server │
    │ (UI & Plotly Charts)    │ ◄──────────────────────── │   (Port 8000)            │
    └─────────────────────────┘      JSON Payload          └────────────┬─────────────┘
                                                                        │
                                                    ┌───────────────────┼───────────────────┐
                                                    ▼                   ▼                   ▼
                                            ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
                                            │ Prophet ML   │    │ Inventory    │    │ Purchase     │
                                            │ Forecast     │    │ Optimization │    │ Order Export │
                                            └──────────────┘    └──────────────┘    └──────────────┘
    ```
    
    ### 🔬 Machine Learning & Optimization Logic:
    1. **Demand Forecasting Engine**:
       - Model: **Log-Transformed Meta Prophet** (`yearly_seasonality=True`, `weekly_seasonality=True`, `multiplicative`).
       - Features: Promotional regressor (`onpromotion`) + multiplicative seasonal decomposition.
       - Uncertainty Estimation: Generates 95% confidence bounds (`yhat_lower` and `yhat_upper`).
    
    2. **Inventory Optimization Formulas**:
       - **Economic Order Quantity (EOQ)**: Calculates batch size minimizing holding and ordering costs:
         $$\\text{EOQ} = \\sqrt{\\frac{2 \\cdot D \\cdot S}{H}}$$
       - **Reorder Point (ROP)**: Determines stock level triggering a new purchase order:
         $$\\text{ROP} = (d \\cdot L) + SS$$
       - Where: $D$ = Total Demand, $S$ = Ordering Cost, $H$ = Holding Cost, $d$ = Daily Demand, $L$ = Lead Time, $SS$ = Safety Stock.
    
    3. **REST API Endpoints**:
       - `GET /health` — Service health check.
       - `POST /api/v1/forecast` — Executes model inference & inventory status classification.
       - `POST /api/v1/generate-po` — Streams formatted Excel Purchase Orders.
    """)