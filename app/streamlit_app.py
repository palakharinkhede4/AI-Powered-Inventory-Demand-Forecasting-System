import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import io
import os
import sys

# Ensure project root is in sys.path for Streamlit Cloud execution
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.forecast import generate_forecast
from src.inventory import (
    calculate_eoq,
    reorder_point,
    classify_inventory_status,
    calculate_recommended_reorder
)
from src.report import create_purchase_order_excel

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="AI Inventory & Demand Forecasting System",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for polished modern look
st.markdown("""
    <style>
    .metric-card {
        background-color: #1e222d;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #2e3440;
    }
    .status-card-red {
        background-color: #3b181c;
        border-left: 6px solid #ff4b4b;
        padding: 15px;
        border-radius: 6px;
        color: #ffa1a1;
    }
    .status-card-yellow {
        background-color: #3b3018;
        border-left: 6px solid #ffa100;
        padding: 15px;
        border-radius: 6px;
        color: #ffdfa1;
    }
    .status-card-green {
        background-color: #183b24;
        border-left: 6px solid #00c853;
        padding: 15px;
        border-radius: 6px;
        color: #a1ffc4;
    }
    .status-card-blue {
        background-color: #182a3b;
        border-left: 6px solid #29b6f6;
        padding: 15px;
        border-radius: 6px;
        color: #a1e5ff;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📦 AI Inventory & Demand Forecasting System")
st.caption("Decoupled MLOps Architecture: FastAPI Backend | Prophet Uncertainty Forecasting | Inventory Optimization")

# Check FastAPI health status
api_online = False
try:
    health_resp = requests.get(f"{API_URL}/health", timeout=1.5)
    if health_resp.status_code == 200:
        api_online = True
except Exception:
    api_online = False

if api_online:
    st.sidebar.success("⚡ REST API: Online (Connected to FastAPI)")
else:
    st.sidebar.info("💡 Mode: Embedded Engine (FastAPI offline at port 8000)")

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ Forecasting & Inventory Inputs")

days = st.sidebar.slider("Forecast Horizon (Days)", 7, 60, 30)
promo_boost = st.sidebar.slider("Promotional Scenario Boost (%)", -50, 100, 0)

st.sidebar.subheader("📦 Inventory Parameters")
current_stock = st.sidebar.number_input("Current Stock (Units)", value=5000, min_value=0)
lead_time = st.sidebar.number_input("Supplier Lead Time (Days)", value=5, min_value=1)
ordering_cost = st.sidebar.number_input("Ordering Cost S (₹/order)", value=50.0, min_value=0.0)
holding_cost = st.sidebar.number_input("Holding Cost H (₹/unit/year)", value=2.0, min_value=0.1)
safety_stock = st.sidebar.number_input("Safety Stock Buffer (Units)", value=500, min_value=0)

st.sidebar.subheader("🏭 Supplier & PO Details")
vendor_name = st.sidebar.text_input("Supplier Name", value="Global Logistics Supply")
item_name = st.sidebar.text_input("Item Description", value="Store #1 Aggregated SKU")
unit_cost = st.sidebar.number_input("Unit Cost (₹)", value=25.0, min_value=0.1)

# Fetch forecast data (from REST API if online, else local engine)
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
    except Exception as e:
        st.error(f"API Error: {e}. Falling back to embedded engine.")
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

# ---------------- METRICS ROW ----------------
m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric("Total Projected Demand", f"{int(metrics['total_expected_demand']):,} units")
with m2:
    st.metric("Avg Daily Demand", f"{int(metrics['avg_daily_demand']):,} units/day")
with m3:
    st.metric("Optimal EOQ", f"{int(metrics['eoq']):,} units")
with m4:
    st.metric("Reorder Point (ROP)", f"{int(metrics['rop']):,} units")

st.divider()

# ---------------- INVENTORY HEALTH GRID ----------------
st.subheader("🎯 Inventory Health Status Grid")

level = status_info.get("level", "Green")
icon = status_info.get("icon", "🟢")
title = status_info.get("title", "")
msg = status_info.get("message", "")
reorder_qty = metrics["recommended_reorder_qty"]

card_css = f"status-card-{level.lower()}"
st.markdown(f"""
    <div class="{card_css}">
        <h3>{icon} {title} (Status: {level.upper()})</h3>
        <p>{msg}</p>
        <p><b>Current Stock:</b> {current_stock:,} units | <b>Reorder Threshold:</b> {int(metrics['rop']):,} units | <b>Recommended Reorder:</b> {reorder_qty:,} units</p>
    </div>
""", unsafe_allow_html=True)

st.write("")

# ---------------- VISUALIZATIONS (PLOTLY) ----------------
tab1, tab2, tab3 = st.tabs(["📈 Demand Forecast & Uncertainty Bands", "📋 Forecast Data Table", "📊 Trend & Seasonal Components"])

with tab1:
    fig = go.Figure()

    # Historical Actual Sales
    if not hist_df.empty:
        fig.add_trace(go.Scatter(
            x=hist_df['ds'],
            y=hist_df['y'],
            mode='lines',
            name='Historical Sales',
            line=dict(color='#8884d8', width=2)
        ))

    # Upper Bound (for Shaded Uncertainty Area)
    fig.add_trace(go.Scatter(
        x=fc_df['ds'],
        y=fc_df['yhat_upper'],
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))

    # Lower Bound & Shading
    fig.add_trace(go.Scatter(
        x=fc_df['ds'],
        y=fc_df['yhat_lower'],
        mode='lines',
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(0, 184, 212, 0.2)',
        name='95% Confidence Interval',
        hoverinfo='skip'
    ))

    # Forecast Centerline (yhat)
    fig.add_trace(go.Scatter(
        x=fc_df['ds'],
        y=fc_df['yhat'],
        mode='lines+markers',
        name='Predicted Demand (yhat)',
        line=dict(color='#00e676', width=3)
    ))

    # Reorder Point Baseline
    fig.add_trace(go.Scatter(
        x=list(hist_df['ds']) + list(fc_df['ds']),
        y=[metrics['rop']] * (len(hist_df) + len(fc_df)),
        mode='lines',
        name=f"Reorder Point ({int(metrics['rop'])} units)",
        line=dict(color='#ff1744', width=2, dash='dash')
    ))

    fig.update_layout(
        title=f"{days}-Day AI Demand Forecast with 95% Uncertainty Bands",
        xaxis_title="Date",
        yaxis_title="Units Sold / Demanded",
        template="plotly_dark",
        hovermode="x unified",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.write("### Detailed Daily Predictions")
    display_df = fc_df.copy()
    for c in ['yhat', 'yhat_lower', 'yhat_upper']:
        if c in display_df.columns:
            display_df[c] = display_df[c].round(2)
    st.dataframe(display_df, use_container_width=True)

with tab3:
    st.write("### Underlying Trend Component")
    if 'trend' in fc_df.columns:
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=fc_df['ds'],
            y=fc_df['trend'],
            mode='lines',
            name='Trend',
            line=dict(color='#ffab40', width=2)
        ))
        fig_trend.update_layout(
            title="Baseline Trend Curve",
            xaxis_title="Date",
            yaxis_title="Baseline Sales Trend",
            template="plotly_dark",
            height=400
        )
        st.plotly_chart(fig_trend, use_container_width=True)

st.divider()

# ---------------- ONE-CLICK PURCHASE ORDER GENERATOR ----------------
st.subheader("📝 One-Click Automated Purchase Order Generator")
st.write("Generate and download a formal, supplier-ready Purchase Order (PO) Excel spreadsheet based on current inventory metrics.")

c_po1, c_po2 = st.columns([2, 1])

with c_po1:
    st.markdown(f"""
    * **Vendor:** {vendor_name}
    * **Item:** {item_name}
    * **Recommended Order Qty:** `{reorder_qty:,} units`
    * **Unit Price:** `₹{unit_cost:,.2f}`
    * **Estimated Total Order Cost:** `₹{reorder_qty * unit_cost:,.2f}`
    """)

with c_po2:
    if api_online:
        try:
            po_payload = {
                "vendor_name": vendor_name,
                "item_name": item_name,
                "current_stock": current_stock,
                "rop": metrics['rop'],
                "eoq": metrics['eoq'],
                "recommended_qty": reorder_qty,
                "unit_cost": unit_cost,
                "avg_daily_demand": metrics['avg_daily_demand'],
                "lead_time_days": int(lead_time)
            }
            po_bytes = requests.post(f"{API_URL}/api/v1/generate-po", json=po_payload).content
        except Exception:
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
    else:
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
        label="📥 Download Purchase Order (.xlsx)",
        data=po_bytes,
        file_name=f"PO_{vendor_name.replace(' ', '_')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )