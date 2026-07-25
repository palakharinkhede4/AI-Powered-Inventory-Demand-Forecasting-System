# 📦 AI Inventory & Demand Forecasting System

A full-stack, MLOps-powered Inventory Optimization & Demand Forecasting system. Built with **Prophet**, **FastAPI**, **Streamlit**, and **Plotly**.

---

## 🌟 Key Features

- 📈 **Time-Series Forecasting**: Log-transformed Meta Prophet model predicting demand with **95% uncertainty confidence bands** (`yhat_lower`, `yhat_upper`).
- ⚡ **Decoupled REST API (FastAPI)**: Production-grade REST backend for model inference and inventory optimization.
- 🎯 **Inventory Health Status Grid**: Dynamic Red/Yellow/Green/Blue classification based on Reorder Point (ROP), Economic Order Quantity (EOQ), and Safety Stock buffer.
- 📊 **Interactive Analytics Dashboard**: Modern Plotly chart with historical sales, predicted demand centerline, uncertainty intervals, baseline trend, and seasonal decomposition.
- 📝 **Automated Purchase Order Generator**: One-click generation and instant download of formal supplier-ready Excel Purchase Orders (`.xlsx`).

---

## 🏗️ System Architecture

```
                       ┌─────────────────────────┐
                       │   Streamlit Frontend    │
                       │ (Interactive Dashboard) │
                       └────────────┬────────────┘
                                    │  HTTP / REST
                                    ▼
┌───────────────────────────────────────────────────────────────────────┐
│                          FastAPI Backend                              │
│ ┌──────────────────────┐  ┌────────────────────┐  ┌───────────────────┐│
│ │   Demand Forecast    │  │     Inventory      │  │     Purchase      ││
│ │  (Prophet Engine)    │  │  Optimization EOQ  │  │   Order Export    ││
│ └──────────────────────┘  └────────────────────┘  └───────────────────┘│
└───────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Preprocess Data & Train Model
```bash
python src/preprocess.py
python src/train_model.py
python src/evaluate.py
```

### 3. Launch FastAPI Backend
```bash
uvicorn src.api:app --reload --port 8000
```
> Interactive API Docs will be available at: `http://127.0.0.1:8000/docs`

### 4. Launch Streamlit Frontend Dashboard
```bash
streamlit run app/streamlit_app.py
```
> Dashboard will open at `http://localhost:8501`

---

## 🛠️ Tech Stack

- **ML Engine:** Facebook Prophet, Scikit-learn, Pandas, NumPy
- **REST API Backend:** FastAPI, Uvicorn, Pydantic
- **Frontend & UI:** Streamlit, Plotly, HTML/CSS
- **Exporting:** OpenPyXL, Pandas (Excel Generation)