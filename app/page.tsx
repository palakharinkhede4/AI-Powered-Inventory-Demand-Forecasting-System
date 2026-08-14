'use client';

import React, { useState, useEffect, useMemo } from 'react';
import {
  TrendingUp,
  Package,
  Calendar,
  AlertTriangle,
  CheckCircle2,
  AlertCircle,
  Info,
  Download,
  FileSpreadsheet,
  Layers,
  Activity,
  Sliders,
  RefreshCw,
  Clock,
  ArrowRight,
  Database,
  Building2,
  FileText,
  Cpu,
  Search,
  ChevronLeft,
  ChevronRight,
  ShieldCheck
} from 'lucide-react';
import {
  ResponsiveContainer,
  ComposedChart,
  Line,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid,
  ReferenceLine
} from 'recharts';
import * as XLSX from 'xlsx';
import { ForecastRequest, ForecastResponse, HistoricalRecord, StatusInfo } from '@/lib/types';
import { runForecastSimulation, getHistoricalData } from '@/lib/forecastEngine';

export default function Dashboard() {
  // Operational Parameters State
  const [days, setDays] = useState<number>(30);
  const [currentStock, setCurrentStock] = useState<number>(5000);
  const [leadTime, setLeadTime] = useState<number>(5);
  const [promoScenario, setPromoScenario] = useState<number>(0);
  const [orderingCost, setOrderingCost] = useState<number>(50);
  const [holdingCost, setHoldingCost] = useState<number>(2);
  const [safetyStock, setSafetyStock] = useState<number>(500);

  // UI State
  const [activeTab, setActiveTab] = useState<'forecast' | 'history' | 'po' | 'architecture'>('forecast');
  const [showAdvanced, setShowAdvanced] = useState<boolean>(false);
  const [showValueExpander, setShowValueExpander] = useState<boolean>(false);
  const [showForecastTable, setShowForecastTable] = useState<boolean>(false);
  const [isLoading, setIsLoading] = useState<boolean>(false);

  // PO Generator Form State
  const [vendorName, setVendorName] = useState<string>('Global Logistics Supply Ltd');
  const [itemName, setItemName] = useState<string>('Aggregated Retail SKU #1');
  const [unitCost, setUnitCost] = useState<number>(25.0);
  const [isDownloadingPO, setIsDownloadingPO] = useState<boolean>(false);

  // Historical Data Table State
  const [histSearch, setHistSearch] = useState<string>('');
  const [histPage, setHistPage] = useState<number>(1);
  const rowsPerPage = 15;

  // Run calculation simulation
  const result: ForecastResponse = useMemo(() => {
    const params: ForecastRequest = {
      days,
      current_stock: currentStock,
      lead_time: leadTime,
      ordering_cost: orderingCost,
      holding_cost: holdingCost,
      safety_stock: safetyStock,
      promo_boost_pct: promoScenario,
    };
    return runForecastSimulation(params);
  }, [days, currentStock, leadTime, promoScenario, orderingCost, holdingCost, safetyStock]);

  const metrics = result.metrics;
  const statusInfo = result.status_info;
  const forecastData = result.forecast;
  const historicalTail = result.historical;

  // Load complete historical data
  const allHistorical = useMemo(() => getHistoricalData(), []);

  // Filtered historical data
  const filteredHistorical = useMemo(() => {
    if (!histSearch.trim()) return allHistorical;
    const query = histSearch.toLowerCase();
    return allHistorical.filter(
      (r) => r.date.includes(query) || (r.onpromotion ? 'promo promotion' : '').includes(query)
    );
  }, [allHistorical, histSearch]);

  const totalHistPages = Math.ceil(filteredHistorical.length / rowsPerPage) || 1;
  const pagedHistorical = useMemo(() => {
    const start = (histPage - 1) * rowsPerPage;
    return filteredHistorical.slice(start, start + rowsPerPage);
  }, [filteredHistorical, histPage]);

  // Combine historical tail and forecast for seamless Recharts visualization
  const chartData = useMemo(() => {
    const combined: any[] = [];
    const dailyRopBaseline = leadTime > 0 ? Math.round(metrics.rop / leadTime) : metrics.avg_daily_demand;

    // Historical data points
    historicalTail.forEach((item) => {
      combined.push({
        date: item.ds,
        historicalSales: Math.round(item.y),
        forecastDemand: null,
        uncertaintyLower: null,
        uncertaintyUpper: null,
        dailyRop: dailyRopBaseline,
      });
    });

    // Bridge point connecting last historical to first forecast
    if (historicalTail.length > 0 && forecastData.length > 0) {
      const lastHist = historicalTail[historicalTail.length - 1];
      // Forecast data points
      forecastData.forEach((item) => {
        combined.push({
          date: item.ds,
          historicalSales: null,
          forecastDemand: item.yhat,
          uncertaintyLower: item.yhat_lower,
          uncertaintyRange: [item.yhat_lower, item.yhat_upper],
          uncertaintyUpper: item.yhat_upper,
          dailyRop: dailyRopBaseline,
        });
      });
    }

    return combined;
  }, [historicalTail, forecastData, metrics.rop, leadTime, metrics.avg_daily_demand]);

  // Status Styling mappings
  const statusStyles = {
    Red: {
      border: 'border-rose-500/40',
      bg: 'bg-rose-950/30',
      text: 'text-rose-300',
      badge: 'bg-rose-500/20 text-rose-300 border-rose-500/30',
      icon: AlertCircle,
    },
    Yellow: {
      border: 'border-amber-500/40',
      bg: 'bg-amber-950/30',
      text: 'text-amber-300',
      badge: 'bg-amber-500/20 text-amber-300 border-amber-500/30',
      icon: AlertTriangle,
    },
    Green: {
      border: 'border-emerald-500/40',
      bg: 'bg-emerald-950/30',
      text: 'text-emerald-300',
      badge: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30',
      icon: CheckCircle2,
    },
    Blue: {
      border: 'border-sky-500/40',
      bg: 'bg-sky-950/30',
      text: 'text-sky-300',
      badge: 'bg-sky-500/20 text-sky-300 border-sky-500/30',
      icon: Info,
    },
  };

  const currentStatusStyle = statusStyles[statusInfo.level] || statusStyles.Green;
  const StatusIcon = currentStatusStyle.icon;

  // Download Historical CSV
  const handleDownloadHistoricalCSV = () => {
    const csvContent =
      'data:text/csv;charset=utf-8,' +
      ['date,sales,onpromotion', ...allHistorical.map((r) => `${r.date},${r.sales},${r.onpromotion || 0}`)].join('\n');
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', 'historical_store_sales.csv');
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // Download Forecast CSV
  const handleDownloadForecastCSV = () => {
    const csvContent =
      'data:text/csv;charset=utf-8,' +
      ['date,predicted_demand,lower_bound_95,upper_bound_95,trend',
        ...forecastData.map((f) => `${f.ds},${f.yhat},${f.yhat_lower},${f.yhat_upper},${f.trend}`)
      ].join('\n');
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', `forecast_${days}days.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // Download Purchase Order Excel (.xlsx)
  const handleDownloadPO = async () => {
    setIsDownloadingPO(true);
    try {
      const payload = {
        vendor_name: vendorName,
        item_name: itemName,
        current_stock: currentStock,
        rop: metrics.rop,
        eoq: metrics.eoq,
        recommended_qty: metrics.recommended_reorder_qty,
        unit_cost: unitCost,
        avg_daily_demand: metrics.avg_daily_demand,
        lead_time_days: leadTime,
      };

      const res = await fetch('/api/po', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (res.ok) {
        const blob = await res.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `Purchase_Order_${vendorName.replace(/\s+/g, '_')}.xlsx`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
      } else {
        // Client-side fallback with xlsx
        downloadPOClientFallback();
      }
    } catch (err) {
      downloadPOClientFallback();
    } finally {
      setIsDownloadingPO(false);
    }
  };

  const downloadPOClientFallback = () => {
    const poNumber = `PO-${Date.now().toString().slice(-8)}`;
    const poDate = new Date().toISOString().split('T')[0];
    const totalCost = metrics.recommended_reorder_qty * unitCost;

    const wb = XLSX.utils.book_new();
    const sheetData = [
      ['ENTERPRISE PURCHASE ORDER', '', '', '', ''],
      ['', '', '', '', ''],
      ['PO Number:', poNumber, '', 'Date:', poDate],
      ['Vendor / Supplier:', vendorName, '', 'Lead Time:', `${leadTime} Days`],
      ['Order Status:', 'APPROVED / PENDING TRANSMISSION', '', 'Generated By:', 'StockPulse Engine'],
      ['', '', '', '', ''],
      ['LINE ITEMS & INVENTORY METRICS', '', '', '', ''],
      ['Item Description', 'Current Stock', 'ROP Threshold', 'EOQ Batch', 'Daily Demand', 'Order Qty', 'Unit Price (INR)', 'Subtotal (INR)'],
      [itemName, currentStock, metrics.rop, metrics.eoq, metrics.avg_daily_demand, metrics.recommended_reorder_qty, unitCost, totalCost],
      ['', '', '', '', '', '', '', ''],
      ['Total Units Ordered:', metrics.recommended_reorder_qty, '', '', '', '', 'Total Order Cost (INR):', totalCost],
    ];

    const ws = XLSX.utils.aoa_to_sheet(sheetData);
    XLSX.utils.book_append_sheet(wb, ws, 'Purchase Order');
    XLSX.writeFile(wb, `Purchase_Order_${vendorName.replace(/\s+/g, '_')}.xlsx`);
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans">
      {/* Header Bar */}
      <header className="border-b border-slate-800/80 bg-slate-900/60 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-indigo-600/20 border border-indigo-500/30 rounded-lg text-indigo-400">
              <Layers className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <span className="font-bold text-base tracking-tight text-white">StockPulse AI</span>
                <span className="text-xs px-2 py-0.5 rounded bg-indigo-500/10 text-indigo-400 border border-indigo-500/20 font-medium">
                  v2.0 Enterprise
                </span>
              </div>
              <p className="text-xs text-slate-400 hidden sm:block">
                Time-Series Demand Forecasting & Replenishment Optimizer
              </p>
            </div>
          </div>

          <div className="flex items-center space-x-4">
            <div className="hidden md:flex items-center space-x-2 px-3 py-1.5 rounded-full bg-slate-900 border border-slate-800 text-xs">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
              <span className="text-slate-300 font-medium">Vercel Edge Active (24/7 Uptime)</span>
            </div>
            <button
              onClick={() => setActiveTab('po')}
              className="inline-flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold shadow-sm transition-all"
            >
              <FileSpreadsheet className="w-3.5 h-3.5" />
              <span>Generate PO</span>
            </button>
          </div>
        </div>
      </header>

      {/* Main Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6">
        {/* Business Value Collapsible Card */}
        <div className="rounded-xl border border-slate-800/80 bg-slate-900/40 p-4 transition-all">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="p-1.5 bg-slate-800 rounded text-indigo-400">
                <Activity className="w-4 h-4" />
              </div>
              <div>
                <h3 className="text-sm font-semibold text-slate-200">System Purpose & Business Impact</h3>
                <p className="text-xs text-slate-400">
                  Preventing costly retail stockouts and optimizing working capital holding costs.
                </p>
              </div>
            </div>
            <button
              onClick={() => setShowValueExpander(!showValueExpander)}
              className="text-xs text-indigo-400 hover:text-indigo-300 font-medium underline-offset-4 hover:underline"
            >
              {showValueExpander ? 'Collapse' : 'Learn More'}
            </button>
          </div>

          {showValueExpander && (
            <div className="mt-4 pt-4 border-t border-slate-800 grid grid-cols-1 md:grid-cols-3 gap-4 text-xs text-slate-300">
              <div className="p-3 bg-slate-900/70 rounded-lg border border-slate-800/60">
                <span className="font-semibold text-indigo-300 block mb-1">1. Demand Forecasting</span>
                Predicts daily customer sales volume for 7 to 60 days ahead with 95% confidence intervals using decomposed seasonality and promotional regressor modeling.
              </div>
              <div className="p-3 bg-slate-900/70 rounded-lg border border-slate-800/60">
                <span className="font-semibold text-amber-300 block mb-1">2. Stockout Risk Alert</span>
                Monitors warehouse burn rate against supplier lead time to establish real-time Reorder Points (ROP), eliminating lost sales revenue.
              </div>
              <div className="p-3 bg-slate-900/70 rounded-lg border border-slate-800/60">
                <span className="font-semibold text-emerald-300 block mb-1">3. Automated Replenishment</span>
                Computes Economic Order Quantity (EOQ) to minimize carrying and freight expenses, then generates formal supplier purchase orders.
              </div>
            </div>
          )}
        </div>

        {/* Top 4 KPI Cards */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="glass-card rounded-xl p-5 border border-slate-800/80 flex flex-col justify-between">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
                Projected {days}-Day Demand
              </span>
              <div className="p-2 rounded-lg bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
                <TrendingUp className="w-4 h-4" />
              </div>
            </div>
            <div className="mt-3">
              <div className="text-2xl sm:text-3xl font-bold tracking-tight text-white tabular-nums">
                {metrics.total_expected_demand.toLocaleString()}
              </div>
              <div className="text-xs text-indigo-400 mt-1 font-medium flex items-center space-x-1">
                <span>Cumulative expected units</span>
              </div>
            </div>
          </div>

          <div className="glass-card rounded-xl p-5 border border-slate-800/80 flex flex-col justify-between">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
                Daily Sales Velocity
              </span>
              <div className="p-2 rounded-lg bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                <Activity className="w-4 h-4" />
              </div>
            </div>
            <div className="mt-3">
              <div className="text-2xl sm:text-3xl font-bold tracking-tight text-white tabular-nums">
                {metrics.avg_daily_demand.toLocaleString()}
              </div>
              <div className="text-xs text-emerald-400 mt-1 font-medium">
                Units demanded per day
              </div>
            </div>
          </div>

          <div className="glass-card rounded-xl p-5 border border-slate-800/80 flex flex-col justify-between">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
                Stock Runway
              </span>
              <div className="p-2 rounded-lg bg-amber-500/10 text-amber-400 border border-amber-500/20">
                <Clock className="w-4 h-4" />
              </div>
            </div>
            <div className="mt-3">
              <div className="text-2xl sm:text-3xl font-bold tracking-tight text-white tabular-nums">
                {metrics.days_remaining} Days
              </div>
              <div className="text-xs text-amber-400 mt-1 font-medium truncate">
                Estimated stockout: {metrics.stockout_date}
              </div>
            </div>
          </div>

          <div className="glass-card rounded-xl p-5 border border-slate-800/80 flex flex-col justify-between">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
                Optimal Order Size (EOQ)
              </span>
              <div className="p-2 rounded-lg bg-sky-500/10 text-sky-400 border border-sky-500/20">
                <Package className="w-4 h-4" />
              </div>
            </div>
            <div className="mt-3">
              <div className="text-2xl sm:text-3xl font-bold tracking-tight text-white tabular-nums">
                {metrics.eoq.toLocaleString()}
              </div>
              <div className="text-xs text-sky-400 mt-1 font-medium">
                Units per replenishment batch
              </div>
            </div>
          </div>
        </div>

        {/* Inventory Status Banner */}
        <div className={`rounded-xl border p-4 sm:p-5 ${currentStatusStyle.border} ${currentStatusStyle.bg} transition-all`}>
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
            <div className="flex items-start space-x-3.5">
              <div className={`p-2 rounded-lg ${currentStatusStyle.badge} shrink-0 mt-0.5`}>
                <StatusIcon className="w-5 h-5" />
              </div>
              <div>
                <div className="flex items-center space-x-2">
                  <h4 className="text-base font-semibold text-white tracking-tight">
                    Inventory Status: {statusInfo.title}
                  </h4>
                  <span className={`text-xs px-2 py-0.5 rounded font-medium ${currentStatusStyle.badge}`}>
                    {statusInfo.level} Tier
                  </span>
                </div>
                <p className="text-xs sm:text-sm text-slate-300 mt-1">
                  {statusInfo.message}
                </p>
                <div className="flex flex-wrap items-center gap-x-4 gap-y-1 mt-2 text-xs text-slate-400">
                  <span>Current Inventory: <strong className="text-slate-200">{currentStock.toLocaleString()} units</strong></span>
                  <span>•</span>
                  <span>Reorder Point (ROP): <strong className="text-slate-200">{metrics.rop.toLocaleString()} units</strong></span>
                  <span>•</span>
                  <span>Recommended Order: <strong className="text-slate-200">{metrics.recommended_reorder_qty.toLocaleString()} units</strong></span>
                </div>
              </div>
            </div>

            {metrics.recommended_reorder_qty > 0 && (
              <button
                onClick={() => setActiveTab('po')}
                className="shrink-0 inline-flex items-center justify-center space-x-2 px-4 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold shadow-sm transition-all"
              >
                <span>Draft Purchase Order</span>
                <ArrowRight className="w-4 h-4" />
              </button>
            )}
          </div>
        </div>

        {/* Main Workspace Layout (Sidebar Controls + Main Tabbed Content) */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Left Column: Operational Controls Panel */}
          <div className="lg:col-span-4 space-y-5">
            <div className="glass-card rounded-xl p-5 border border-slate-800/80 space-y-5">
              <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                <div className="flex items-center space-x-2">
                  <Sliders className="w-4 h-4 text-indigo-400" />
                  <h3 className="text-sm font-semibold text-white">Operational Controls</h3>
                </div>
                <button
                  onClick={() => {
                    setDays(30);
                    setCurrentStock(5000);
                    setLeadTime(5);
                    setPromoScenario(0);
                    setOrderingCost(50);
                    setHoldingCost(2);
                    setSafetyStock(500);
                  }}
                  className="text-xs text-slate-400 hover:text-slate-200 flex items-center space-x-1"
                  title="Reset to defaults"
                >
                  <RefreshCw className="w-3 h-3" />
                  <span>Reset</span>
                </button>
              </div>

              {/* Forecast Horizon Slider */}
              <div className="space-y-2">
                <div className="flex items-center justify-between text-xs">
                  <label htmlFor="horizon-slider" className="font-medium text-slate-300">
                    Forecast Horizon
                  </label>
                  <span className="px-2 py-0.5 rounded bg-slate-800 text-indigo-300 font-semibold tabular-nums">
                    {days} Days
                  </span>
                </div>
                <input
                  id="horizon-slider"
                  type="range"
                  min={7}
                  max={60}
                  step={1}
                  value={days}
                  onChange={(e) => setDays(Number(e.target.value))}
                  className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-indigo-500"
                />
                <div className="flex justify-between text-[10px] text-slate-500 px-0.5">
                  <span>7 Days</span>
                  <span>14 Days</span>
                  <span>30 Days</span>
                  <span>60 Days</span>
                </div>
              </div>

              {/* Current Stock Input */}
              <div className="space-y-1.5">
                <div className="flex items-center justify-between text-xs">
                  <label htmlFor="stock-input" className="font-medium text-slate-300">
                    Current Stock Level
                  </label>
                  <span className="text-[11px] text-slate-400">Units on hand</span>
                </div>
                <div className="relative">
                  <input
                    id="stock-input"
                    type="number"
                    min={0}
                    step={100}
                    value={currentStock}
                    onChange={(e) => setCurrentStock(Math.max(0, Number(e.target.value)))}
                    className="w-full bg-slate-900 border border-slate-800 rounded-lg px-3 py-2 text-sm text-white placeholder-slate-500 focus:outline-none focus:border-indigo-500 transition-colors"
                  />
                  <div className="absolute right-2.5 top-2.5 text-xs text-slate-500">units</div>
                </div>
              </div>

              {/* Lead Time Input */}
              <div className="space-y-1.5">
                <div className="flex items-center justify-between text-xs">
                  <label htmlFor="lead-time-input" className="font-medium text-slate-300">
                    Supplier Lead Time
                  </label>
                  <span className="text-[11px] text-slate-400">Order to delivery</span>
                </div>
                <div className="relative">
                  <input
                    id="lead-time-input"
                    type="number"
                    min={1}
                    max={30}
                    step={1}
                    value={leadTime}
                    onChange={(e) => setLeadTime(Math.max(1, Number(e.target.value)))}
                    className="w-full bg-slate-900 border border-slate-800 rounded-lg px-3 py-2 text-sm text-white placeholder-slate-500 focus:outline-none focus:border-indigo-500 transition-colors"
                  />
                  <div className="absolute right-2.5 top-2.5 text-xs text-slate-500">days</div>
                </div>
              </div>

              {/* Promotional Scenario Selector */}
              <div className="space-y-1.5">
                <label htmlFor="promo-select" className="text-xs font-medium text-slate-300 block">
                  Promotional Event Scenario
                </label>
                <select
                  id="promo-select"
                  value={promoScenario}
                  onChange={(e) => setPromoScenario(Number(e.target.value))}
                  className="w-full bg-slate-900 border border-slate-800 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-indigo-500 transition-colors"
                >
                  <option value={0}>Baseline Demand (No Promotion)</option>
                  <option value={15}>Seasonal Promotion (+15% Demand)</option>
                  <option value={35}>Major Flash Campaign (+35% Demand)</option>
                </select>
              </div>

              {/* Advanced Parameters Accordion */}
              <div className="border-t border-slate-800/80 pt-3">
                <button
                  type="button"
                  onClick={() => setShowAdvanced(!showAdvanced)}
                  className="w-full flex items-center justify-between text-xs text-slate-400 hover:text-slate-200 transition-colors py-1"
                >
                  <span className="font-medium">Advanced Cost & Buffer Parameters</span>
                  <span className="text-slate-500">{showAdvanced ? '▲' : '▼'}</span>
                </button>

                {showAdvanced && (
                  <div className="space-y-3 mt-3 pt-3 border-t border-slate-850">
                    <div>
                      <label className="text-[11px] font-medium text-slate-400 block mb-1">
                        Ordering Cost S (INR / order)
                      </label>
                      <input
                        type="number"
                        min={1}
                        value={orderingCost}
                        onChange={(e) => setOrderingCost(Math.max(1, Number(e.target.value)))}
                        className="w-full bg-slate-900 border border-slate-800 rounded px-2.5 py-1.5 text-xs text-white focus:outline-none focus:border-indigo-500"
                      />
                    </div>
                    <div>
                      <label className="text-[11px] font-medium text-slate-400 block mb-1">
                        Holding Cost H (INR / unit / year)
                      </label>
                      <input
                        type="number"
                        min={0.1}
                        step={0.5}
                        value={holdingCost}
                        onChange={(e) => setHoldingCost(Math.max(0.1, Number(e.target.value)))}
                        className="w-full bg-slate-900 border border-slate-800 rounded px-2.5 py-1.5 text-xs text-white focus:outline-none focus:border-indigo-500"
                      />
                    </div>
                    <div>
                      <label className="text-[11px] font-medium text-slate-400 block mb-1">
                        Safety Stock Buffer (Units)
                      </label>
                      <input
                        type="number"
                        min={0}
                        step={50}
                        value={safetyStock}
                        onChange={(e) => setSafetyStock(Math.max(0, Number(e.target.value)))}
                        className="w-full bg-slate-900 border border-slate-800 rounded px-2.5 py-1.5 text-xs text-white focus:outline-none focus:border-indigo-500"
                      />
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Quick Metrics Summary Card */}
            <div className="glass-card rounded-xl p-5 border border-slate-800/80 space-y-3">
              <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
                Reorder Threshold Summary
              </h4>
              <div className="space-y-2 text-xs">
                <div className="flex justify-between py-1 border-b border-slate-800/60">
                  <span className="text-slate-400">Reorder Point (ROP)</span>
                  <span className="font-semibold text-white tabular-nums">{metrics.rop.toLocaleString()} units</span>
                </div>
                <div className="flex justify-between py-1 border-b border-slate-800/60">
                  <span className="text-slate-400">Economic Order Qty (EOQ)</span>
                  <span className="font-semibold text-white tabular-nums">{metrics.eoq.toLocaleString()} units</span>
                </div>
                <div className="flex justify-between py-1 border-b border-slate-800/60">
                  <span className="text-slate-400">Current Deficit / Buffer</span>
                  <span className={`font-semibold tabular-nums ${currentStock <= metrics.rop ? 'text-rose-400' : 'text-emerald-400'}`}>
                    {(currentStock - metrics.rop).toLocaleString()} units
                  </span>
                </div>
                <div className="flex justify-between py-1">
                  <span className="text-slate-400">Recommended Replenishment</span>
                  <span className="font-bold text-indigo-400 tabular-nums">
                    {metrics.recommended_reorder_qty.toLocaleString()} units
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Right Column: Interactive Tabs Container */}
          <div className="lg:col-span-8 space-y-4">
            {/* Navigation Tabs Header */}
            <div className="flex border-b border-slate-800 overflow-x-auto scrollbar-none space-x-2">
              <button
                onClick={() => setActiveTab('forecast')}
                className={`flex items-center space-x-2 px-4 py-3 text-xs font-semibold border-b-2 transition-all whitespace-nowrap ${
                  activeTab === 'forecast'
                    ? 'border-indigo-500 text-indigo-400 bg-indigo-500/5'
                    : 'border-transparent text-slate-400 hover:text-slate-200 hover:border-slate-700'
                }`}
              >
                <TrendingUp className="w-3.5 h-3.5" />
                <span>Demand Forecast & Analytics</span>
              </button>

              <button
                onClick={() => setActiveTab('history')}
                className={`flex items-center space-x-2 px-4 py-3 text-xs font-semibold border-b-2 transition-all whitespace-nowrap ${
                  activeTab === 'history'
                    ? 'border-indigo-500 text-indigo-400 bg-indigo-500/5'
                    : 'border-transparent text-slate-400 hover:text-slate-200 hover:border-slate-700'
                }`}
              >
                <Database className="w-3.5 h-3.5" />
                <span>Historical Sales Explorer</span>
              </button>

              <button
                onClick={() => setActiveTab('po')}
                className={`flex items-center space-x-2 px-4 py-3 text-xs font-semibold border-b-2 transition-all whitespace-nowrap ${
                  activeTab === 'po'
                    ? 'border-indigo-500 text-indigo-400 bg-indigo-500/5'
                    : 'border-transparent text-slate-400 hover:text-slate-200 hover:border-slate-700'
                }`}
              >
                <FileText className="w-3.5 h-3.5" />
                <span>Purchase Order Generator</span>
              </button>

              <button
                onClick={() => setActiveTab('architecture')}
                className={`flex items-center space-x-2 px-4 py-3 text-xs font-semibold border-b-2 transition-all whitespace-nowrap ${
                  activeTab === 'architecture'
                    ? 'border-indigo-500 text-indigo-400 bg-indigo-500/5'
                    : 'border-transparent text-slate-400 hover:text-slate-200 hover:border-slate-700'
                }`}
              >
                <Cpu className="w-3.5 h-3.5" />
                <span>System Architecture</span>
              </button>
            </div>

            {/* TAB 1: DEMAND FORECAST & VISUALS */}
            {activeTab === 'forecast' && (
              <div className="glass-card rounded-xl p-5 border border-slate-800/80 space-y-6">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-800 pb-4">
                  <div>
                    <h3 className="text-sm font-semibold text-white flex items-center space-x-2">
                      <span>Demand Forecast & 95% Confidence Interval</span>
                    </h3>
                    <p className="text-xs text-slate-400">
                      Log-transformed multiplicative Prophet forecast with historical training tail (last 90 days).
                    </p>
                  </div>
                  <div className="flex items-center space-x-2">
                    <button
                      onClick={handleDownloadForecastCSV}
                      className="inline-flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-slate-900 hover:bg-slate-850 border border-slate-800 text-xs text-slate-300 transition-colors"
                    >
                      <Download className="w-3.5 h-3.5 text-slate-400" />
                      <span>Export CSV</span>
                    </button>
                  </div>
                </div>

                {/* Recharts Visualizer */}
                <div className="h-80 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <ComposedChart data={chartData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                      <defs>
                        <linearGradient id="uncertaintyFill" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#6366f1" stopOpacity={0.25} />
                          <stop offset="95%" stopColor="#6366f1" stopOpacity={0.02} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                      <XAxis
                        dataKey="date"
                        stroke="#64748b"
                        fontSize={11}
                        tickLine={false}
                        minTickGap={40}
                      />
                      <YAxis
                        stroke="#64748b"
                        fontSize={11}
                        tickLine={false}
                        tickFormatter={(v) => `${(v / 1000).toFixed(0)}k`}
                      />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: '#0f172a',
                          borderColor: '#334155',
                          borderRadius: '8px',
                          fontSize: '12px',
                          color: '#f8fafc',
                        }}
                        formatter={(value: any, name: string) => {
                          if (value === null || value === undefined) return ['--', name];
                          const num = Number(value);
                          if (name === 'Historical Sales') return [`${num.toLocaleString()} units`, name];
                          if (name === 'Predicted Demand') return [`${num.toLocaleString()} units`, name];
                          if (name === 'Daily ROP Baseline') return [`${num.toLocaleString()} units/day`, name];
                          if (name === 'Upper 95% Bound') return [`${num.toLocaleString()} units`, name];
                          if (name === 'Lower 95% Bound') return [`${num.toLocaleString()} units`, name];
                          return [num.toLocaleString(), name];
                        }}
                      />
                      <Legend
                        verticalAlign="top"
                        height={36}
                        iconType="circle"
                        wrapperStyle={{ fontSize: '11px', color: '#94a3b8' }}
                      />

                      {/* Confidence Envelope Upper */}
                      <Area
                        type="monotone"
                        dataKey="uncertaintyUpper"
                        stroke="none"
                        fill="url(#uncertaintyFill)"
                        name="Upper 95% Bound"
                        connectNulls
                      />

                      {/* Historical Sales Trace */}
                      <Line
                        type="monotone"
                        dataKey="historicalSales"
                        stroke="#818cf8"
                        strokeWidth={2}
                        dot={false}
                        name="Historical Sales"
                        connectNulls
                      />

                      {/* Forecast Demand Centerline */}
                      <Line
                        type="monotone"
                        dataKey="forecastDemand"
                        stroke="#34d399"
                        strokeWidth={2.5}
                        dot={{ r: 2, fill: '#34d399' }}
                        name="Predicted Demand"
                        connectNulls
                      />

                      {/* Daily ROP Baseline */}
                      <Line
                        type="monotone"
                        dataKey="dailyRop"
                        stroke="#f87171"
                        strokeWidth={1.5}
                        strokeDasharray="4 4"
                        dot={false}
                        name="Daily ROP Baseline"
                      />
                    </ComposedChart>
                  </ResponsiveContainer>
                </div>

                <div className="p-3 bg-slate-900/60 rounded-lg border border-slate-800/60 text-xs text-slate-400 space-y-1">
                  <p>
                    <strong className="text-slate-300">Chart Interpretation:</strong> The red dashed line represents the <strong>Daily Reorder Rate</strong> ({Math.round(metrics.rop / leadTime).toLocaleString()} units/day).
                    The total cumulative <strong>{leadTime}-Day ROP Threshold</strong> is <strong>{metrics.rop.toLocaleString()} units</strong>.
                  </p>
                </div>

                {/* Collapsible Daily Forecast Table */}
                <div className="border-t border-slate-800 pt-4">
                  <button
                    onClick={() => setShowForecastTable(!showForecastTable)}
                    className="flex items-center justify-between w-full text-xs font-semibold text-slate-300 hover:text-white"
                  >
                    <span>View Tabular Forecast Values ({forecastData.length} Records)</span>
                    <span className="text-slate-500">{showForecastTable ? 'Hide Table' : 'Show Table'}</span>
                  </button>

                  {showForecastTable && (
                    <div className="mt-3 overflow-x-auto rounded-lg border border-slate-800">
                      <table className="w-full text-left text-xs">
                        <thead className="bg-slate-900/90 text-slate-400 border-b border-slate-800">
                          <tr>
                            <th className="p-2.5 font-medium">Date</th>
                            <th className="p-2.5 font-medium">Predicted Demand (yhat)</th>
                            <th className="p-2.5 font-medium">Lower 95% Bound</th>
                            <th className="p-2.5 font-medium">Upper 95% Bound</th>
                            <th className="p-2.5 font-medium">Underlying Trend</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-850 text-slate-300 font-mono text-[11px]">
                          {forecastData.map((row, idx) => (
                            <tr key={idx} className="hover:bg-slate-900/50">
                              <td className="p-2.5 font-sans text-slate-200">{row.ds}</td>
                              <td className="p-2.5 font-semibold text-emerald-400">{row.yhat.toLocaleString()}</td>
                              <td className="p-2.5 text-slate-400">{row.yhat_lower.toLocaleString()}</td>
                              <td className="p-2.5 text-slate-400">{row.yhat_upper.toLocaleString()}</td>
                              <td className="p-2.5 text-slate-400">{row.trend.toLocaleString()}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* TAB 2: HISTORICAL DATA EXPLORER */}
            {activeTab === 'history' && (
              <div className="glass-card rounded-xl p-5 border border-slate-800/80 space-y-5">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-800 pb-4">
                  <div>
                    <h3 className="text-sm font-semibold text-white">Historical Sales Dataset</h3>
                    <p className="text-xs text-slate-400">
                      Examining 1,684 daily retail sales observations (2013-2017) utilized to fit the predictive model.
                    </p>
                  </div>
                  <button
                    onClick={handleDownloadHistoricalCSV}
                    className="inline-flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold shadow-sm transition-colors"
                  >
                    <Download className="w-3.5 h-3.5" />
                    <span>Download Full CSV</span>
                  </button>
                </div>

                {/* Dataset Aggregate Metrics */}
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                  <div className="p-3 bg-slate-900 rounded-lg border border-slate-800">
                    <span className="text-[11px] text-slate-400 block">Total Records</span>
                    <span className="text-base font-bold text-white tabular-nums">{allHistorical.length.toLocaleString()} Days</span>
                  </div>
                  <div className="p-3 bg-slate-900 rounded-lg border border-slate-800">
                    <span className="text-[11px] text-slate-400 block">Date Range</span>
                    <span className="text-xs font-bold text-slate-200">2013-01 to 2017-08</span>
                  </div>
                  <div className="p-3 bg-slate-900 rounded-lg border border-slate-800">
                    <span className="text-[11px] text-slate-400 block">Daily Average Volume</span>
                    <span className="text-base font-bold text-indigo-400 tabular-nums">
                      {Math.round(allHistorical.reduce((a, c) => a + c.sales, 0) / allHistorical.length).toLocaleString()} units
                    </span>
                  </div>
                  <div className="p-3 bg-slate-900 rounded-lg border border-slate-800">
                    <span className="text-[11px] text-slate-400 block">Peak Daily Volume</span>
                    <span className="text-base font-bold text-emerald-400 tabular-nums">
                      {Math.max(...allHistorical.map((r) => r.sales)).toLocaleString()} units
                    </span>
                  </div>
                </div>

                {/* Search / Filter Bar */}
                <div className="flex items-center justify-between gap-4">
                  <div className="relative flex-1 max-w-sm">
                    <Search className="w-4 h-4 text-slate-500 absolute left-3 top-2.5" />
                    <input
                      type="text"
                      placeholder="Filter by date (e.g. 2017-05)..."
                      value={histSearch}
                      onChange={(e) => {
                        setHistSearch(e.target.value);
                        setHistPage(1);
                      }}
                      className="w-full bg-slate-900 border border-slate-800 rounded-lg pl-9 pr-3 py-2 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-indigo-500"
                    />
                  </div>
                  <span className="text-xs text-slate-400">
                    Showing {pagedHistorical.length} of {filteredHistorical.length} rows
                  </span>
                </div>

                {/* Paginated Historical Table */}
                <div className="overflow-x-auto rounded-lg border border-slate-800">
                  <table className="w-full text-left text-xs">
                    <thead className="bg-slate-900/90 text-slate-400 border-b border-slate-800">
                      <tr>
                        <th className="p-2.5 font-medium">Record Date</th>
                        <th className="p-2.5 font-medium">Sales Volume (Units)</th>
                        <th className="p-2.5 font-medium">Promotional Event</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-850 text-slate-300 font-mono text-[11px]">
                      {pagedHistorical.map((row, idx) => (
                        <tr key={idx} className="hover:bg-slate-900/50">
                          <td className="p-2.5 font-sans text-slate-200">{row.date}</td>
                          <td className="p-2.5 font-semibold text-white">{row.sales.toLocaleString()}</td>
                          <td className="p-2.5 font-sans">
                            {row.onpromotion && row.onpromotion > 0 ? (
                              <span className="px-2 py-0.5 rounded bg-indigo-500/20 text-indigo-300 text-[10px] font-medium border border-indigo-500/30">
                                Active Promotion ({row.onpromotion})
                              </span>
                            ) : (
                              <span className="text-slate-500 text-[10px]">Standard Day</span>
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                {/* Pagination Controls */}
                <div className="flex items-center justify-between text-xs text-slate-400 pt-2">
                  <span>Page {histPage} of {totalHistPages}</span>
                  <div className="flex items-center space-x-2">
                    <button
                      onClick={() => setHistPage((p) => Math.max(1, p - 1))}
                      disabled={histPage === 1}
                      className="p-1.5 rounded bg-slate-900 border border-slate-800 hover:bg-slate-850 disabled:opacity-40 disabled:cursor-not-allowed"
                    >
                      <ChevronLeft className="w-4 h-4" />
                    </button>
                    <button
                      onClick={() => setHistPage((p) => Math.min(totalHistPages, p + 1))}
                      disabled={histPage === totalHistPages}
                      className="p-1.5 rounded bg-slate-900 border border-slate-800 hover:bg-slate-850 disabled:opacity-40 disabled:cursor-not-allowed"
                    >
                      <ChevronRight className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              </div>
            )}

            {/* TAB 3: PURCHASE ORDER GENERATOR */}
            {activeTab === 'po' && (
              <div className="glass-card rounded-xl p-5 border border-slate-800/80 space-y-6">
                <div className="border-b border-slate-800 pb-4">
                  <h3 className="text-sm font-semibold text-white">Automated Supplier Purchase Order Generator</h3>
                  <p className="text-xs text-slate-400">
                    Instantly export formal, supplier-ready Excel spreadsheets (.xlsx) based on AI inventory optimization parameters.
                  </p>
                </div>

                {/* Form Controls */}
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  <div>
                    <label className="text-xs font-medium text-slate-300 block mb-1.5">Supplier / Vendor Name</label>
                    <input
                      type="text"
                      value={vendorName}
                      onChange={(e) => setVendorName(e.target.value)}
                      className="w-full bg-slate-900 border border-slate-800 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-indigo-500"
                    />
                  </div>
                  <div>
                    <label className="text-xs font-medium text-slate-300 block mb-1.5">Item / SKU Description</label>
                    <input
                      type="text"
                      value={itemName}
                      onChange={(e) => setItemName(e.target.value)}
                      className="w-full bg-slate-900 border border-slate-800 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-indigo-500"
                    />
                  </div>
                  <div>
                    <label className="text-xs font-medium text-slate-300 block mb-1.5">Unit Purchase Cost (INR)</label>
                    <input
                      type="number"
                      min={0.1}
                      step={1}
                      value={unitCost}
                      onChange={(e) => setUnitCost(Math.max(0.1, Number(e.target.value)))}
                      className="w-full bg-slate-900 border border-slate-800 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-indigo-500"
                    />
                  </div>
                  <div className="p-3 bg-slate-900 rounded-lg border border-slate-800 flex flex-col justify-center">
                    <span className="text-[11px] text-slate-400">Estimated Total Order Value</span>
                    <span className="text-lg font-bold text-emerald-400">
                      ₹{(metrics.recommended_reorder_qty * unitCost).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                    </span>
                  </div>
                </div>

                {/* Live PO Document Preview Box */}
                <div className="rounded-lg border border-slate-800 bg-slate-950/70 p-4 space-y-4">
                  <div className="flex items-center justify-between border-b border-slate-850 pb-3">
                    <div>
                      <span className="text-xs font-bold text-slate-300 uppercase tracking-wide">Purchase Order Document Preview</span>
                      <p className="text-[11px] text-slate-500">Document formatted for automated Excel transmission</p>
                    </div>
                    <span className="text-xs px-2.5 py-1 rounded bg-emerald-500/10 text-emerald-300 border border-emerald-500/20 font-medium">
                      APPROVED
                    </span>
                  </div>

                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
                    <div>
                      <span className="text-slate-500 block text-[11px]">Vendor:</span>
                      <span className="font-semibold text-slate-200">{vendorName}</span>
                    </div>
                    <div>
                      <span className="text-slate-500 block text-[11px]">Lead Time:</span>
                      <span className="font-semibold text-slate-200">{leadTime} Days</span>
                    </div>
                    <div>
                      <span className="text-slate-500 block text-[11px]">Item Name:</span>
                      <span className="font-semibold text-slate-200">{itemName}</span>
                    </div>
                    <div>
                      <span className="text-slate-500 block text-[11px]">Reorder Qty:</span>
                      <span className="font-bold text-indigo-400">{metrics.recommended_reorder_qty.toLocaleString()} units</span>
                    </div>
                  </div>
                </div>

                {/* Download Actions */}
                <div className="flex flex-col sm:flex-row items-center gap-3 pt-2">
                  <button
                    onClick={handleDownloadPO}
                    disabled={isDownloadingPO}
                    className="w-full sm:w-auto flex-1 inline-flex items-center justify-center space-x-2 px-5 py-2.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-semibold shadow-sm transition-all disabled:opacity-50"
                  >
                    <FileSpreadsheet className="w-4 h-4" />
                    <span>{isDownloadingPO ? 'Generating Excel...' : 'Download Official Purchase Order (.xlsx)'}</span>
                  </button>
                </div>
              </div>
            )}

            {/* TAB 4: SYSTEM ARCHITECTURE & METHODOLOGY */}
            {activeTab === 'architecture' && (
              <div className="glass-card rounded-xl p-5 border border-slate-800/80 space-y-6">
                <div className="border-b border-slate-800 pb-4">
                  <h3 className="text-sm font-semibold text-white">Technical Architecture & Mathematical Methodology</h3>
                  <p className="text-xs text-slate-400">
                    Detailed system flow, time-series forecasting formulation, and replenishment optimization mechanics.
                  </p>
                </div>

                {/* Architecture Flow Box */}
                <div className="space-y-3">
                  <h4 className="text-xs font-semibold text-indigo-300 uppercase tracking-wider">
                    Decoupled Edge / Serverless Architecture
                  </h4>
                  <div className="p-4 rounded-lg bg-slate-900/90 border border-slate-800 font-mono text-[11px] text-slate-300 overflow-x-auto leading-relaxed">
                    {`┌───────────────────────────┐      REST API      ┌─────────────────────────────┐
│  Next.js Modern Dashboard │ ─────────────────► │  Vercel Edge API Routes     │
│  (TypeScript, Tailwind)   │ ◄───────────────── │  (/api/forecast, /api/po)   │
└───────────────────────────┘    JSON Payload    └──────────────┬──────────────┘
                                                                │
                                        ┌───────────────────────┼───────────────────────┐
                                        ▼                       ▼                       ▼
                                ┌──────────────┐        ┌──────────────┐        ┌──────────────┐
                                │ Decomposed   │        │ Inventory    │        │ Excel Sheet  │
                                │ Meta Prophet │        │ Optimization │        │ Generator    │
                                │ Inference    │        │ (EOQ & ROP)  │        │ (SheetJS)    │
                                └──────────────┘        └──────────────┘        └──────────────┘`}
                  </div>
                </div>

                {/* Mathematical Formulas */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
                  <div className="p-4 rounded-lg bg-slate-900/70 border border-slate-800 space-y-2">
                    <h5 className="font-semibold text-white flex items-center space-x-1.5">
                      <ShieldCheck className="w-4 h-4 text-emerald-400" />
                      <span>Economic Order Quantity (EOQ)</span>
                    </h5>
                    <p className="text-slate-400 text-[11px]">
                      Calculates the batch order size that minimizes the sum of inventory carrying costs and purchase order setup fees:
                    </p>
                    <div className="p-2 rounded bg-slate-950 font-mono text-emerald-300 text-center font-semibold">
                      EOQ = √ ( (2 · D · S) / H )
                    </div>
                    <p className="text-[10px] text-slate-500">
                      Where D = Total forecasted demand, S = Ordering fixed cost, H = Unit holding cost per period.
                    </p>
                  </div>

                  <div className="p-4 rounded-lg bg-slate-900/70 border border-slate-800 space-y-2">
                    <h5 className="font-semibold text-white flex items-center space-x-1.5">
                      <AlertCircle className="w-4 h-4 text-amber-400" />
                      <span>Reorder Point (ROP)</span>
                    </h5>
                    <p className="text-slate-400 text-[11px]">
                      Establishes the minimum stock trigger level ensuring operations never breach safety buffer during supplier lead time:
                    </p>
                    <div className="p-2 rounded bg-slate-950 font-mono text-amber-300 text-center font-semibold">
                      ROP = (d · L) + SS
                    </div>
                    <p className="text-[10px] text-slate-500">
                      Where d = Daily demand velocity, L = Lead time in days, SS = Safety stock buffer.
                    </p>
                  </div>
                </div>

                {/* API Specs Table */}
                <div className="space-y-3">
                  <h4 className="text-xs font-semibold text-slate-300 uppercase tracking-wider">
                    Serverless API Endpoints
                  </h4>
                  <div className="overflow-x-auto rounded-lg border border-slate-800">
                    <table className="w-full text-left text-xs">
                      <thead className="bg-slate-900/90 text-slate-400 border-b border-slate-800">
                        <tr>
                          <th className="p-2.5 font-medium">Endpoint</th>
                          <th className="p-2.5 font-medium">Method</th>
                          <th className="p-2.5 font-medium">Payload / Parameters</th>
                          <th className="p-2.5 font-medium">Response Format</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-850 text-slate-300 text-[11px]">
                        <tr>
                          <td className="p-2.5 font-mono text-indigo-300">/api/forecast</td>
                          <td className="p-2.5"><span className="px-1.5 py-0.5 rounded bg-emerald-500/20 text-emerald-300 text-[10px] font-bold">POST</span></td>
                          <td className="p-2.5 font-mono text-slate-400">&#123; days, current_stock, lead_time, ... &#125;</td>
                          <td className="p-2.5 text-slate-300">JSON (Metrics, Forecast Array, ROP/EOQ Status)</td>
                        </tr>
                        <tr>
                          <td className="p-2.5 font-mono text-indigo-300">/api/po</td>
                          <td className="p-2.5"><span className="px-1.5 py-0.5 rounded bg-emerald-500/20 text-emerald-300 text-[10px] font-bold">POST</span></td>
                          <td className="p-2.5 font-mono text-slate-400">&#123; vendor_name, item_name, unit_cost, ... &#125;</td>
                          <td className="p-2.5 text-slate-300">Binary Stream (.xlsx Excel spreadsheet)</td>
                        </tr>
                        <tr>
                          <td className="p-2.5 font-mono text-indigo-300">/api/history</td>
                          <td className="p-2.5"><span className="px-1.5 py-0.5 rounded bg-sky-500/20 text-sky-300 text-[10px] font-bold">GET</span></td>
                          <td className="p-2.5 font-mono text-slate-400">None</td>
                          <td className="p-2.5 text-slate-300">JSON (1,684 Historical Observations)</td>
                        </tr>
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-850 mt-12 py-6 text-center text-xs text-slate-500">
        <p>StockPulse AI • Enterprise Demand Forecasting & Supply Chain Replenishment Engine</p>
      </footer>
    </div>
  );
}
