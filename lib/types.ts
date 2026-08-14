export interface ForecastRequest {
  days: number;
  current_stock: number;
  lead_time: number;
  ordering_cost: number;
  holding_cost: number;
  safety_stock: number;
  promo_boost_pct: number;
}

export interface KPIMetrics {
  total_expected_demand: number;
  avg_daily_demand: number;
  eoq: number;
  rop: number;
  recommended_reorder_qty: number;
  days_remaining: number;
  stockout_date: string;
}

export interface StatusInfo {
  status: 'CRITICAL_STOCKOUT' | 'REORDER_WARNING' | 'OPTIMAL_HEALTHY' | 'OVERSTOCKED';
  level: 'Red' | 'Yellow' | 'Green' | 'Blue';
  title: string;
  message: string;
  alert_type: 'error' | 'warning' | 'success' | 'info';
}

export interface HistoricalRecord {
  date: string;
  sales: number;
  onpromotion?: number;
}

export interface ForecastPoint {
  ds: string;
  yhat: number;
  yhat_lower: number;
  yhat_upper: number;
  trend: number;
  weekly?: number;
}

export interface ForecastResponse {
  success: boolean;
  metrics: KPIMetrics;
  status_info: StatusInfo;
  historical: { ds: string; y: number }[];
  forecast: ForecastPoint[];
}

export interface PORequest {
  vendor_name: string;
  item_name: string;
  current_stock: number;
  rop: number;
  eoq: number;
  recommended_qty: number;
  unit_cost: number;
  avg_daily_demand: number;
  lead_time_days: number;
}
