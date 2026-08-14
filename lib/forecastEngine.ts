import dataset from './data/dataset.json';
import { ForecastRequest, ForecastResponse, KPIMetrics, StatusInfo, ForecastPoint, HistoricalRecord } from './types';

export function getHistoricalData(): HistoricalRecord[] {
  return dataset.historical as HistoricalRecord[];
}

export function calculateEOQ(totalDemand: number, orderingCost: number, holdingCost: number): number {
  if (holdingCost <= 0 || totalDemand <= 0) return 0;
  return Math.sqrt((2 * totalDemand * orderingCost) / holdingCost);
}

export function calculateROP(dailyDemand: number, leadTime: number, safetyStock: number): number {
  return (dailyDemand * leadTime) + safetyStock;
}

export function classifyInventoryStatus(
  currentStock: number,
  rop: number,
  eoq: number,
  safetyStock: number = 0
): StatusInfo {
  const safetyBuffer = rop >= safetyStock ? rop - safetyStock : rop * 0.5;

  if (currentStock <= safetyBuffer) {
    return {
      status: 'CRITICAL_STOCKOUT',
      level: 'Red',
      title: 'Critical Stockout Risk',
      message: 'Immediate purchase order required. Current inventory is near or below the safety threshold buffer.',
      alert_type: 'error'
    };
  } else if (currentStock <= rop) {
    return {
      status: 'REORDER_WARNING',
      level: 'Yellow',
      title: 'Reorder Point Reached',
      message: 'Current stock has fallen below the Reorder Point (ROP). Place replenishment purchase order.',
      alert_type: 'warning'
    };
  } else if (currentStock <= rop + (eoq > 0 ? 1.5 * eoq : rop)) {
    return {
      status: 'OPTIMAL_HEALTHY',
      level: 'Green',
      title: 'Optimal Inventory Level',
      message: 'Stock levels are balanced within optimal operational demand parameters.',
      alert_type: 'success'
    };
  } else {
    return {
      status: 'OVERSTOCKED',
      level: 'Blue',
      title: 'Overstock Condition',
      message: 'Inventory significantly exceeds near-term demand horizon, increasing holding costs.',
      alert_type: 'info'
    };
  }
}

export function calculateRecommendedReorder(currentStock: number, rop: number, eoq: number): number {
  if (currentStock <= rop) {
    const needed = (rop + eoq) - currentStock;
    return Math.max(Math.ceil(eoq), Math.ceil(needed));
  }
  return 0;
}

export function runForecastSimulation(req: ForecastRequest): ForecastResponse {
  const horizon = Math.min(Math.max(req.days, 1), dataset.baseline_forecast.length);
  const baseline = dataset.baseline_forecast.slice(0, horizon);

  // Apply promotional multiplier modifier
  const promoMultiplier = 1.0 + (req.promo_boost_pct / 100.0) * 0.45;

  const adjustedForecast: ForecastPoint[] = baseline.map((pt) => {
    const yhat = Math.max(0, Math.round(pt.yhat * promoMultiplier * 100) / 100);
    const yhat_lower = Math.max(0, Math.round(pt.yhat_lower * promoMultiplier * 100) / 100);
    const yhat_upper = Math.max(0, Math.round(pt.yhat_upper * promoMultiplier * 100) / 100);
    return {
      ds: pt.ds,
      yhat,
      yhat_lower,
      yhat_upper,
      trend: Math.round(pt.trend * 100) / 100,
      weekly: pt.weekly
    };
  });

  const totalDemand = adjustedForecast.reduce((acc, curr) => acc + curr.yhat, 0);
  const avgDailyDemand = adjustedForecast.length > 0 ? totalDemand / adjustedForecast.length : 0;

  const eoq = calculateEOQ(totalDemand, req.ordering_cost, req.holding_cost);
  const rop = calculateROP(avgDailyDemand, req.lead_time, req.safety_stock);

  const statusInfo = classifyInventoryStatus(req.current_stock, rop, eoq, req.safety_stock);
  const recommendedReorder = calculateRecommendedReorder(req.current_stock, rop, eoq);

  const daysRemaining = avgDailyDemand > 0 ? Math.round((req.current_stock / avgDailyDemand) * 10) / 10 : 999;
  
  let stockoutDate = 'N/A';
  if (daysRemaining < 365) {
    const futureDate = new Date();
    futureDate.setDate(futureDate.getDate() + Math.round(daysRemaining));
    stockoutDate = futureDate.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
  }

  const metrics: KPIMetrics = {
    total_expected_demand: Math.round(totalDemand),
    avg_daily_demand: Math.round(avgDailyDemand),
    eoq: Math.round(eoq),
    rop: Math.round(rop),
    recommended_reorder_qty: recommendedReorder,
    days_remaining: daysRemaining,
    stockout_date: stockoutDate
  };

  return {
    success: true,
    metrics,
    status_info: statusInfo,
    historical: dataset.historical_tail_90,
    forecast: adjustedForecast
  };
}
