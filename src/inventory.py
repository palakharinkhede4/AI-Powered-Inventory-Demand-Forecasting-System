import numpy as np

def calculate_eoq(D, S, H):
    return np.sqrt((2 * D * S) / H)

def reorder_point(daily_demand, lead_time, safety_stock):
    return (daily_demand * lead_time) + safety_stock

if __name__ == "__main__":
    print("EOQ:", calculate_eoq(10000, 50, 2))
    print("ROP:", reorder_point(30, 5, 100))