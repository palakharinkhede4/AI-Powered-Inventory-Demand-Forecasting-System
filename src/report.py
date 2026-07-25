import io
import pandas as pd
from datetime import datetime

def create_purchase_order_excel(
    vendor_name: str,
    item_name: str,
    current_stock: float,
    rop: float,
    eoq: float,
    recommended_qty: int,
    unit_cost: float,
    avg_daily_demand: float,
    lead_time_days: int
) -> bytes:
    """Generates a professional Purchase Order Excel spreadsheet in memory."""
    po_number = f"PO-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    po_date = datetime.now().strftime("%Y-%m-%d")
    total_cost = recommended_qty * unit_cost

    # Summary Metadata Table
    header_data = {
        "Field": ["PO Number", "PO Date", "Vendor / Supplier", "Lead Time (Days)", "Order Status"],
        "Value": [po_number, po_date, vendor_name, lead_time_days, "APPROVED / PENDING TRANSMISSION"]
    }
    header_df = pd.DataFrame(header_data)

    # Line Items Table
    line_item_data = {
        "Item Description": [item_name],
        "Current Stock": [int(current_stock)],
        "Reorder Point (ROP)": [int(rop)],
        "Optimal EOQ": [int(eoq)],
        "Avg Daily Demand": [round(avg_daily_demand, 2)],
        "Order Qty (Units)": [int(recommended_qty)],
        "Unit Cost (₹)": [float(unit_cost)],
        "Subtotal Cost (₹)": [round(total_cost, 2)]
    }
    line_item_df = pd.DataFrame(line_item_data)

    # Summary Total Table
    totals_data = {
        "Metric": ["Total Units Ordered", "Total Estimated Order Cost (₹)"],
        "Value": [int(recommended_qty), f"₹{total_cost:,.2f}"]
    }
    totals_df = pd.DataFrame(totals_data)

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        header_df.to_excel(writer, sheet_name="Purchase Order", index=False, startrow=1, startcol=1)
        line_item_df.to_excel(writer, sheet_name="Purchase Order", index=False, startrow=8, startcol=1)
        totals_df.to_excel(writer, sheet_name="Purchase Order", index=False, startrow=12, startcol=1)

    return output.getvalue()
