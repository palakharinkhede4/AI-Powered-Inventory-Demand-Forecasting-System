# 📦 Smart AI Inventory & Demand Forecasting System

A full-stack, MLOps-powered Inventory Optimization & Demand Forecasting platform built with **Prophet**, **FastAPI**, **Streamlit**, **Plotly**, and **OpenPyXL**.

---

## 💡 Real-World Business Use Case & Value Proposition

### 🎯 The Problem
Global retail and e-commerce businesses lose over **$1.1 Trillion every year** due to inventory mismanagement:
- **Overstocking**: Ties up working capital in unsold inventory, increases warehouse holding costs, and leads to product obsolescence/spoilage.
- **Stockouts / Understocking**: Results in lost sales revenue, unfulfilled customer orders, damaged brand reputation, and customer churn.

### 💼 How This System Solves It
This system acts as an intelligent automated assistant for store managers and supply chain planners by answering three critical daily operational questions:
1. **Demand Forecasting**: *"How many units will customers buy over the next 7 to 60 days?"* -> Log-transformed Meta Prophet model predicts daily sales with 95% confidence bounds.
2. **Stockout Risk Warning**: *"When will current warehouse inventory run out?"* -> Dynamic Stock Runway counter & color-coded Reorder Point (ROP) alerts.
3. **Automated Order Replenishment**: *"How many units should we order right now?"* -> Economic Order Quantity (EOQ) optimization + 1-Click Purchase Order (.xlsx) generation.

---

## 🌟 Key Features

- 📈 **Time-Series Forecasting with Uncertainty**: Log-transformed Meta Prophet model predicting demand with **95% uncertainty confidence bands** (`yhat_lower`, `yhat_upper`).
- ⏱️ **Stock Runway & Stockout Date Counter**: Real-time calculation of remaining days of inventory stock based on sales velocity.
- 🎯 **Color-Coded Health Status Grid**: Dynamic Red/Yellow/Green/Blue classification based on Reorder Point (ROP), Economic Order Quantity (EOQ), and Safety Stock buffer.
- 📊 **Historical Dataset Explorer**: Full access to raw historical daily store sales data with interactive filtering and 1-click CSV download.
- 📝 **Automated Purchase Order Generator**: One-click generation and instant download of formal supplier-ready Excel Purchase Orders (`.xlsx`).
- 🛠️ **Decoupled REST API (FastAPI)**: Production REST backend serving model predictions and inventory optimization endpoints.

---

## 🏗️ System Architecture

```
                                 ┌─────────────────────────┐
                                 │   Streamlit Frontend    │
                                 │ (UI & Interactive Charts)│
                                 └────────────┬────────────┘
                                              │  HTTP REST
                                              ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                             FastAPI REST Backend                                │
│ ┌──────────────────────────┐ ┌──────────────────────────┐ ┌────────────────────┐ │
│ │  Prophet Forecasting     │ │  Inventory Optimization  │ │  Purchase Order    │ │
│ │  (Uncertainty Bounds)    │ │  (EOQ / ROP Algorithms)  │ │  Excel Report      │ │
│ └──────────────────────────┘ └──────────────────────────┘ └────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch FastAPI Backend
```bash
uvicorn src.api:app --reload --port 8000
```
> Interactive API Documentation available at `http://127.0.0.1:8000/docs`

### 3. Launch Streamlit Frontend Dashboard
```bash
streamlit run app/streamlit_app.py
```
> Access app in browser at `http://localhost:8501`

---

## ⚙️ Tech Stack & Methods

- **Machine Learning:** Facebook Prophet, Pandas, NumPy, Scikit-learn
- **Backend API:** FastAPI, Uvicorn, Pydantic
- **Frontend UI & Data Viz:** Streamlit, Plotly, HTML/CSS
- **Reporting:** OpenPyXL (Excel generation)