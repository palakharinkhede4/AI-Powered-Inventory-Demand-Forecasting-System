from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import os
import sys

# Ensure project root is in sys.path
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

app = FastAPI(
    title="AI Inventory & Demand Forecasting API",
    description="Production REST API for time-series demand forecasting and automated inventory replenishment optimization.",
    version="1.0.0"
)

# ----------------- SCHEMAS -----------------
class ForecastRequest(BaseModel):
    days: int = Field(default=30, ge=1, le=180, description="Number of future forecast days")
    current_stock: float = Field(default=5000, ge=0, description="Current warehouse inventory count")
    lead_time: float = Field(default=5, ge=1, description="Supplier lead time in days")
    ordering_cost: float = Field(default=50, ge=0, description="Fixed cost per purchase order (₹)")
    holding_cost: float = Field(default=2, gt=0, description="Holding cost per unit per period (₹)")
    safety_stock: float = Field(default=500, ge=0, description="Safety stock buffer")
    promo_boost_pct: float = Field(default=0.0, ge=-50.0, le=200.0, description="Percentage change in promotional activity")

class PORequest(BaseModel):
    vendor_name: str = Field(default="Acme Supplies Ltd")
    item_name: str = Field(default="Store Item #1")
    current_stock: float
    rop: float
    eoq: float
    recommended_qty: int
    unit_cost: float = Field(default=15.0)
    avg_daily_demand: float
    lead_time_days: int = Field(default=5)

# ----------------- ENDPOINTS -----------------
@app.get("/health", tags=["Health Check"])
def health_check():
    return {
        "status": "online",
        "service": "AI Inventory & Demand Forecasting API",
        "version": "1.0.0"
    }

@app.post("/api/v1/forecast", tags=["Forecasting & Inventory Optimization"])
def forecast_and_optimize(req: ForecastRequest):
    try:
        data = generate_forecast(days=req.days, promo_boost_pct=req.promo_boost_pct)
        forecast_list = data["forecast"]
        
        # Demand calculations
        yhat_values = [item["yhat"] for item in forecast_list]
        total_demand = float(np.sum(yhat_values))
        avg_daily_demand = float(np.mean(yhat_values)) if yhat_values else 0.0
        
        # Inventory optimization
        eoq = calculate_eoq(total_demand, req.ordering_cost, req.holding_cost)
        rop = reorder_point(avg_daily_demand, req.lead_time, req.safety_stock)
        
        status_info = classify_inventory_status(
            current_stock=req.current_stock,
            rop=rop,
            eoq=eoq,
            safety_stock=req.safety_stock
        )
        
        reorder_qty = calculate_recommended_reorder(
            current_stock=req.current_stock,
            rop=rop,
            eoq=eoq
        )
        
        return {
            "success": True,
            "metrics": {
                "total_expected_demand": round(total_demand, 2),
                "avg_daily_demand": round(avg_daily_demand, 2),
                "eoq": round(eoq, 2),
                "rop": round(rop, 2),
                "recommended_reorder_qty": int(reorder_qty)
            },
            "status_info": status_info,
            "historical": data["historical"],
            "forecast": forecast_list
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/generate-po", tags=["Automation & Reports"])
def generate_po(req: PORequest):
    try:
        excel_bytes = create_purchase_order_excel(
            vendor_name=req.vendor_name,
            item_name=req.item_name,
            current_stock=req.current_stock,
            rop=req.rop,
            eoq=req.eoq,
            recommended_qty=req.recommended_qty,
            unit_cost=req.unit_cost,
            avg_daily_demand=req.avg_daily_demand,
            lead_time_days=req.lead_time_days
        )
        
        filename = f"Purchase_Order_{req.vendor_name.replace(' ', '_')}.xlsx"
        return Response(
            content=excel_bytes,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
