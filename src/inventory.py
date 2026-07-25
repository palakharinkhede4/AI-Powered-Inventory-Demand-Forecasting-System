import numpy as np

def calculate_eoq(total_demand: float, ordering_cost: float, holding_cost: float) -> float:
    """Calculates Economic Order Quantity (EOQ)."""
    if holding_cost <= 0 or total_demand <= 0:
        return 0.0
    return float(np.sqrt((2 * total_demand * ordering_cost) / holding_cost))

def reorder_point(daily_demand: float, lead_time: float, safety_stock: float) -> float:
    """Calculates Reorder Point (ROP)."""
    return float((daily_demand * lead_time) + safety_stock)

def classify_inventory_status(current_stock: float, rop: float, eoq: float, safety_stock: float = 0):
    """Classifies inventory into Red/Yellow/Green/Blue health status levels."""
    safety_buffer = rop - safety_stock if rop >= safety_stock else rop * 0.5
    
    if current_stock <= safety_buffer:
        return {
            "status": "CRITICAL_STOCKOUT",
            "level": "Red",
            "icon": "🔴",
            "title": "Critical Stockout Risk",
            "message": "Immediate reorder required! Current stock is depleted near or below safety buffer.",
            "alert_type": "error"
        }
    elif current_stock <= rop:
        return {
            "status": "REORDER_WARNING",
            "level": "Yellow",
            "icon": "🟡",
            "title": "Reorder Point Reached",
            "message": "Stock has fallen below Reorder Point (ROP). Place a purchase order soon.",
            "alert_type": "warning"
        }
    elif current_stock <= rop + (1.5 * eoq if eoq > 0 else rop):
        return {
            "status": "OPTIMAL_HEALTHY",
            "level": "Green",
            "icon": "🟢",
            "title": "Optimal Inventory Level",
            "message": "Stock levels are healthy and within optimal operational parameters.",
            "alert_type": "success"
        }
    else:
        return {
            "status": "OVERSTOCKED",
            "level": "Blue",
            "icon": "🔵",
            "title": "Overstock Warning",
            "message": "Inventory level significantly exceeds current demand requirements, increasing holding costs.",
            "alert_type": "info"
        }

def calculate_recommended_reorder(current_stock: float, rop: float, eoq: float) -> int:
    """Calculates recommended purchase order quantity."""
    if current_stock <= rop:
        needed = (rop + eoq) - current_stock
        return max(int(np.ceil(eoq)), int(np.ceil(needed)))
    return 0

if __name__ == "__main__":
    eoq = calculate_eoq(10000, 50, 2)
    rop = reorder_point(30, 5, 100)
    status = classify_inventory_status(150, rop, eoq, 100)
    print(f"EOQ: {eoq:.2f}, ROP: {rop:.2f}")
    print("Status:", status)