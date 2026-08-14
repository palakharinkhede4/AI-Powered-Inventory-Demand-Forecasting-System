'use client';

import React, { useState, useEffect, useMemo } from 'react';
import {
  TrendingUp,
  Package,
  Clock,
  Activity,
  AlertTriangle,
  CheckCircle2,
  AlertCircle,
  Info,
  Download,
  FileSpreadsheet,
  Sliders,
  RotateCcw,
  Search,
  ChevronLeft,
  ChevronRight,
  Sun,
  Moon,
  Laptop,
  ArrowUpRight,
  Database,
  FileText,
  HelpCircle,
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
} from 'recharts';
import * as XLSX from 'xlsx';
import { ForecastRequest, ForecastResponse, StatusInfo } from '@/lib/types';
import { runForecastSimulation, getHistoricalData } from '@/lib/forecastEngine';

type ThemeMode = 'light' | 'dark' | 'system';

export default function Dashboard() {
  // Theme State
  const [theme, setTheme] = useState<ThemeMode>('system');
  const [isDark, setIsDark] = useState<boolean>(false);

  // Initialize theme from localStorage or system preference
  useEffect(() => {
    try {
      const savedTheme = (localStorage.getItem('stockpulse_theme') as ThemeMode) || 'system';
      setTheme(savedTheme);
      applyTheme(savedTheme);
    } catch (_) {}

    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    const handleSystemChange = (e: MediaQueryListEvent) => {
      const currentSaved = localStorage.getItem('stockpulse_theme') as ThemeMode;
      if (!currentSaved || currentSaved === 'system') {
        if (e.matches) {
          document.documentElement.classList.add('dark');
          setIsDark(true);
        } else {
          document.documentElement.classList.remove('dark');
          setIsDark(false);
        }
      }
    };

    mediaQuery.addEventListener('change', handleSystemChange);
    return () => mediaQuery.removeEventListener('change', handleSystemChange);
  }, []);

  const applyTheme = (newTheme: ThemeMode) => {
    try {
      localStorage.setItem('stockpulse_theme', newTheme);
    } catch (_) {}

    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    const shouldBeDark = newTheme === 'dark' || (newTheme === 'system' && prefersDark);

    if (shouldBeDark) {
      document.documentElement.classList.add('dark');
      setIsDark(true);
    } else {
      document.documentElement.classList.remove('dark');
      setIsDark(false);
    }
  };

  const handleThemeToggle = (newTheme: ThemeMode) => {
    setTheme(newTheme);
    applyTheme(newTheme);
  };

  // Operational Parameters State
  const [days, setDays] = useState<number>(30);
  const [currentStock, setCurrentStock] = useState<number>(5000);
  const [leadTime, setLeadTime] = useState<number>(5);
  const [promoScenario, setPromoScenario] = useState<number>(0);
  const [orderingCost, setOrderingCost] = useState<number>(50);
  const [holdingCost, setHoldingCost] = useState<number>(2);
  const [safetyStock, setSafetyStock] = useState<number>(500);

  // UI State
  const [activeTab, setActiveTab] = useState<'forecast' | 'history' | 'po' | 'methodology'>('forecast');
  const [showAdvanced, setShowAdvanced] = useState<boolean>(false);
  const [showForecastTable, setShowForecastTable] = useState<boolean>(false);

  // PO Generator Form State
  const [vendorName, setVendorName] = useState<string>('Global Logistics Supply Ltd');
  const [itemName, setItemName] = useState<string>('Aggregated Retail SKU #1');
  const [unitCost, setUnitCost] = useState<number>(25.0);
  const [isDownloadingPO, setIsDownloadingPO] = useState<boolean>(false);

  // Historical Data Table State
  const [histSearch, setHistSearch] = useState<string>('');
  const [histPage, setHistPage] = useState<number>(1);
  const rowsPerPage = 12;

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

  // Combine historical tail and forecast for Recharts
  const chartData = useMemo(() => {
    const combined: any[] = [];
    const dailyRopBaseline = leadTime > 0 ? Math.round(metrics.rop / leadTime) : metrics.avg_daily_demand;

    historicalTail.forEach((item) => {
      combined.push({
        date: item.ds,
        historicalSales: Math.round(item.y),
        forecastDemand: null,
        uncertaintyUpper: null,
        uncertaintyLower: null,
        dailyRop: dailyRopBaseline,
      });
    });

    forecastData.forEach((item) => {
      combined.push({
        date: item.ds,
        historicalSales: null,
        forecastDemand: item.yhat,
        uncertaintyUpper: item.yhat_upper,
        uncertaintyLower: item.yhat_lower,
        dailyRop: dailyRopBaseline,
      });
    });

    return combined;
  }, [historicalTail, forecastData, metrics.rop, leadTime, metrics.avg_daily_demand]);

  // Status Styling mapping (Clean, muted, non-aggressive)
  const statusConfig = {
    Red: {
      border: 'border-rose-200 dark:border-rose-900/60',
      bg: 'bg-rose-50/70 dark:bg-rose-950/20',
      text: 'text-rose-900 dark:text-rose-200',
      badge: 'bg-rose-100 text-rose-800 dark:bg-rose-900/40 dark:text-rose-300',
      icon: AlertCircle,
      iconColor: 'text-rose-600 dark:text-rose-400',
    },
    Yellow: {
      border: 'border-amber-200 dark:border-amber-900/60',
      bg: 'bg-amber-50/70 dark:bg-amber-950/20',
      text: 'text-amber-900 dark:text-amber-200',
      badge: 'bg-amber-100 text-amber-800 dark:bg-amber-900/40 dark:text-amber-300',
      icon: AlertTriangle,
      iconColor: 'text-amber-600 dark:text-amber-400',
    },
    Green: {
      border: 'border-emerald-200 dark:border-emerald-900/60',
      bg: 'bg-emerald-50/70 dark:bg-emerald-950/20',
      text: 'text-emerald-900 dark:text-emerald-200',
      badge: 'bg-emerald-100 text-emerald-800 dark:bg-emerald-900/40 dark:text-emerald-300',
      icon: CheckCircle2,
      iconColor: 'text-emerald-600 dark:text-emerald-400',
    },
    Blue: {
      border: 'border-sky-200 dark:border-sky-900/60',
      bg: 'bg-sky-50/70 dark:bg-sky-950/20',
      text: 'text-sky-900 dark:text-sky-200',
      badge: 'bg-sky-100 text-sky-800 dark:bg-sky-900/40 dark:text-sky-300',
      icon: Info,
      iconColor: 'text-sky-600 dark:text-sky-400',
    },
  };

  const currentStatus = statusConfig[statusInfo.level] || statusConfig.Green;
  const StatusIcon = currentStatus.icon;

  // Download Historical CSV
  const handleDownloadHistoricalCSV = () => {
    const csvContent =
      'data:text/csv;charset=utf-8,' +
      ['date,sales,onpromotion', ...allHistorical.map((r) => `${r.date},${r.sales},${r.onpromotion || 0}`)].join('\n');
    const link = document.createElement('a');
    link.href = encodeURI(csvContent);
    link.download = 'historical_sales.csv';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // Download Forecast CSV
  const handleDownloadForecastCSV = () => {
    const csvContent =
      'data:text/csv;charset=utf-8,' +
      ['date,predicted_demand,lower_95,upper_95,trend',
        ...forecastData.map((f) => `${f.ds},${f.yhat},${f.yhat_lower},${f.yhat_upper},${f.trend}`)
      ].join('\n');
    const link = document.createElement('a');
    link.href = encodeURI(csvContent);
    link.download = `forecast_${days}d.csv`;
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
        downloadPOClientFallback();
      }
    } catch (_) {
      downloadPOClientFallback();
    } finally {
      setIsDownloadingPO(false);
    }
  };

  const downloadPOClientFallback = () => {
    const poNumber = `PO-${Date.now().toString().slice(-6)}`;
    const poDate = new Date().toISOString().split('T')[0];
    const totalCost = metrics.recommended_reorder_qty * unitCost;

    const wb = XLSX.utils.book_new();
    const sheetData = [
      ['PURCHASE ORDER', '', '', '', ''],
      ['PO Number:', poNumber, '', 'Date:', poDate],
      ['Vendor:', vendorName, '', 'Lead Time:', `${leadTime} Days`],
      ['', '', '', '', ''],
      ['Item Description', 'Current Stock', 'ROP', 'EOQ', 'Daily Demand', 'Order Qty', 'Unit Price', 'Total'],
      [itemName, currentStock, metrics.rop, metrics.eoq, metrics.avg_daily_demand, metrics.recommended_reorder_qty, unitCost, totalCost],
    ];

    const ws = XLSX.utils.aoa_to_sheet(sheetData);
    XLSX.utils.book_append_sheet(wb, ws, 'Purchase Order');
    XLSX.writeFile(wb, `Purchase_Order_${vendorName.replace(/\s+/g, '_')}.xlsx`);
  };

  return (
    <div className="min-h-screen bg-zinc-50 dark:bg-zinc-950 text-zinc-900 dark:text-zinc-100 flex flex-col font-sans transition-colors duration-150">
      {/* Clean Minimalist Header */}
      <header className="border-b border-zinc-200 dark:border-zinc-800 bg-white/80 dark:bg-zinc-900/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-14 flex items-center justify-between">
          <div className="flex items-center space-x-2.5">
            <div className="p-1.5 bg-zinc-900 dark:bg-zinc-100 text-zinc-100 dark:text-zinc-900 rounded-md">
              <TrendingUp className="w-4 h-4" />
            </div>
            <div>
              <span className="font-semibold text-sm tracking-tight text-zinc-900 dark:text-zinc-100">
                StockPulse
              </span>
              <span className="text-zinc-400 dark:text-zinc-500 text-xs ml-2 hidden sm:inline">
                Demand & Inventory Planning
              </span>
            </div>
          </div>

          {/* Header Controls: Theme Switcher & Action */}
          <div className="flex items-center space-x-3">
            {/* Theme Toggle Button Group */}
            <div className="flex items-center p-0.5 rounded-lg bg-zinc-100 dark:bg-zinc-800 border border-zinc-200/80 dark:border-zinc-700/80 text-xs">
              <button
                onClick={() => handleThemeToggle('light')}
                className={`p-1.5 rounded-md transition-all ${
                  theme === 'light'
                    ? 'bg-white dark:bg-zinc-700 text-zinc-900 dark:text-zinc-100 shadow-sm'
                    : 'text-zinc-500 hover:text-zinc-900 dark:hover:text-zinc-200'
                }`}
                title="Light mode"
              >
                <Sun className="w-3.5 h-3.5" />
              </button>
              <button
                onClick={() => handleThemeToggle('dark')}
                className={`p-1.5 rounded-md transition-all ${
                  theme === 'dark'
                    ? 'bg-white dark:bg-zinc-700 text-zinc-900 dark:text-zinc-100 shadow-sm'
                    : 'text-zinc-500 hover:text-zinc-900 dark:hover:text-zinc-200'
                }`}
                title="Dark mode"
              >
                <Moon className="w-3.5 h-3.5" />
              </button>
              <button
                onClick={() => handleThemeToggle('system')}
                className={`p-1.5 rounded-md transition-all ${
                  theme === 'system'
                    ? 'bg-white dark:bg-zinc-700 text-zinc-900 dark:text-zinc-100 shadow-sm'
                    : 'text-zinc-500 hover:text-zinc-900 dark:hover:text-zinc-200'
                }`}
                title="System default"
              >
                <Laptop className="w-3.5 h-3.5" />
              </button>
            </div>

            <button
              onClick={() => setActiveTab('po')}
              className="inline-flex items-center space-x-1.5 px-3 py-1.5 rounded-md bg-zinc-900 hover:bg-zinc-800 dark:bg-zinc-100 dark:hover:bg-zinc-200 text-zinc-100 dark:text-zinc-900 text-xs font-medium transition-colors shadow-sm"
            >
              <FileSpreadsheet className="w-3.5 h-3.5" />
              <span>Create PO</span>
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6">
        {/* KPI Metric Strip */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 sm:gap-4">
          <div className="bg-white dark:bg-zinc-900/60 rounded-lg p-4 border border-zinc-200 dark:border-zinc-800 shadow-xs">
            <span className="text-xs text-zinc-500 dark:text-zinc-400 font-medium block">
              Forecasted {days}-Day Demand
            </span>
            <div className="text-xl sm:text-2xl font-bold text-zinc-900 dark:text-zinc-100 mt-1 tabular-nums">
              {metrics.total_expected_demand.toLocaleString()}
            </div>
            <span className="text-[11px] text-zinc-400 dark:text-zinc-500 mt-0.5 block">
              Units across planning horizon
            </span>
          </div>

          <div className="bg-white dark:bg-zinc-900/60 rounded-lg p-4 border border-zinc-200 dark:border-zinc-800 shadow-xs">
            <span className="text-xs text-zinc-500 dark:text-zinc-400 font-medium block">
              Daily Sales Velocity
            </span>
            <div className="text-xl sm:text-2xl font-bold text-zinc-900 dark:text-zinc-100 mt-1 tabular-nums">
              {metrics.avg_daily_demand.toLocaleString()}
            </div>
            <span className="text-[11px] text-zinc-400 dark:text-zinc-500 mt-0.5 block">
              Units demanded per day
            </span>
          </div>

          <div className="bg-white dark:bg-zinc-900/60 rounded-lg p-4 border border-zinc-200 dark:border-zinc-800 shadow-xs">
            <span className="text-xs text-zinc-500 dark:text-zinc-400 font-medium block">
              Stock Runway
            </span>
            <div className="text-xl sm:text-2xl font-bold text-zinc-900 dark:text-zinc-100 mt-1 tabular-nums">
              {metrics.days_remaining} Days
            </div>
            <span className="text-[11px] text-zinc-400 dark:text-zinc-500 mt-0.5 block truncate">
              Est. stockout {metrics.stockout_date}
            </span>
          </div>

          <div className="bg-white dark:bg-zinc-900/60 rounded-lg p-4 border border-zinc-200 dark:border-zinc-800 shadow-xs">
            <span className="text-xs text-zinc-500 dark:text-zinc-400 font-medium block">
              Optimal Order Batch (EOQ)
            </span>
            <div className="text-xl sm:text-2xl font-bold text-zinc-900 dark:text-zinc-100 mt-1 tabular-nums">
              {metrics.eoq.toLocaleString()}
            </div>
            <span className="text-[11px] text-zinc-400 dark:text-zinc-500 mt-0.5 block">
              Units to minimize carrying cost
            </span>
          </div>
        </div>

        {/* Clean Status Alert Banner */}
        <div className={`rounded-lg border p-3.5 sm:p-4 ${currentStatus.border} ${currentStatus.bg} transition-colors`}>
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
            <div className="flex items-start space-x-3">
              <StatusIcon className={`w-5 h-5 shrink-0 mt-0.5 ${currentStatus.iconColor}`} />
              <div>
                <div className="flex items-center space-x-2">
                  <span className="text-xs font-semibold uppercase tracking-wider text-zinc-900 dark:text-zinc-100">
                    {statusInfo.title}
                  </span>
                  <span className={`text-[10px] px-2 py-0.2 rounded font-medium ${currentStatus.badge}`}>
                    {statusInfo.level}
                  </span>
                </div>
                <p className="text-xs text-zinc-600 dark:text-zinc-300 mt-0.5">
                  {statusInfo.message}
                </p>
                <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-zinc-500 dark:text-zinc-400 mt-1">
                  <span>Current: <strong>{currentStock.toLocaleString()} units</strong></span>
                  <span>•</span>
                  <span>Reorder Point (ROP): <strong>{metrics.rop.toLocaleString()} units</strong></span>
                  <span>•</span>
                  <span>Recommended Order: <strong>{metrics.recommended_reorder_qty.toLocaleString()} units</strong></span>
                </div>
              </div>
            </div>

            {metrics.recommended_reorder_qty > 0 && (
              <button
                onClick={() => setActiveTab('po')}
                className="shrink-0 text-xs font-medium text-zinc-900 dark:text-zinc-100 hover:underline flex items-center space-x-1"
              >
                <span>Draft Purchase Order</span>
                <ArrowUpRight className="w-3.5 h-3.5" />
              </button>
            )}
          </div>
        </div>

        {/* Two-Column Clean Layout */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Left: Operational Parameters Panel */}
          <div className="lg:col-span-4 space-y-4">
            <div className="bg-white dark:bg-zinc-900/60 rounded-lg p-5 border border-zinc-200 dark:border-zinc-800 shadow-xs space-y-4">
              <div className="flex items-center justify-between border-b border-zinc-100 dark:border-zinc-800 pb-3">
                <span className="text-xs font-semibold text-zinc-900 dark:text-zinc-100 uppercase tracking-wide">
                  Parameters
                </span>
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
                  className="text-xs text-zinc-400 hover:text-zinc-700 dark:hover:text-zinc-200 flex items-center space-x-1"
                  title="Reset to default values"
                >
                  <RotateCcw className="w-3 h-3" />
                  <span>Reset</span>
                </button>
              </div>

              {/* Horizon Segmented Options */}
              <div className="space-y-2">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-zinc-600 dark:text-zinc-300 font-medium">Forecast Horizon</span>
                  <span className="font-semibold text-zinc-900 dark:text-zinc-100">{days} Days</span>
                </div>
                <div className="grid grid-cols-4 gap-1.5 p-1 rounded-md bg-zinc-100 dark:bg-zinc-800 text-xs text-center">
                  {[7, 14, 30, 60].map((d) => (
                    <button
                      key={d}
                      onClick={() => setDays(d)}
                      className={`py-1 rounded font-medium transition-all ${
                        days === d
                          ? 'bg-white dark:bg-zinc-700 text-zinc-900 dark:text-zinc-100 shadow-xs'
                          : 'text-zinc-500 hover:text-zinc-900 dark:hover:text-zinc-200'
                      }`}
                    >
                      {d}D
                    </button>
                  ))}
                </div>
                <input
                  type="range"
                  min={7}
                  max={60}
                  step={1}
                  value={days}
                  onChange={(e) => setDays(Number(e.target.value))}
                  className="w-full h-1 bg-zinc-200 dark:bg-zinc-800 rounded-lg appearance-none cursor-pointer accent-zinc-900 dark:accent-zinc-100"
                />
              </div>

              {/* Current Stock Input */}
              <div className="space-y-1">
                <label className="text-xs text-zinc-600 dark:text-zinc-300 font-medium block">
                  Current Stock Level (Units)
                </label>
                <input
                  type="number"
                  min={0}
                  step={100}
                  value={currentStock}
                  onChange={(e) => setCurrentStock(Math.max(0, Number(e.target.value)))}
                  className="w-full bg-zinc-50 dark:bg-zinc-800/60 border border-zinc-200 dark:border-zinc-700/80 rounded-md px-3 py-1.5 text-xs text-zinc-900 dark:text-zinc-100 focus:outline-none focus:border-zinc-400 dark:focus:border-zinc-500"
                />
              </div>

              {/* Supplier Lead Time Input */}
              <div className="space-y-1">
                <label className="text-xs text-zinc-600 dark:text-zinc-300 font-medium block">
                  Supplier Lead Time (Days)
                </label>
                <input
                  type="number"
                  min={1}
                  max={30}
                  step={1}
                  value={leadTime}
                  onChange={(e) => setLeadTime(Math.max(1, Number(e.target.value)))}
                  className="w-full bg-zinc-50 dark:bg-zinc-800/60 border border-zinc-200 dark:border-zinc-700/80 rounded-md px-3 py-1.5 text-xs text-zinc-900 dark:text-zinc-100 focus:outline-none focus:border-zinc-400 dark:focus:border-zinc-500"
                />
              </div>

              {/* Promotional Scenario */}
              <div className="space-y-1">
                <label className="text-xs text-zinc-600 dark:text-zinc-300 font-medium block">
                  Promotional Event Scenario
                </label>
                <select
                  value={promoScenario}
                  onChange={(e) => setPromoScenario(Number(e.target.value))}
                  className="w-full bg-zinc-50 dark:bg-zinc-800/60 border border-zinc-200 dark:border-zinc-700/80 rounded-md px-3 py-1.5 text-xs text-zinc-900 dark:text-zinc-100 focus:outline-none focus:border-zinc-400 dark:focus:border-zinc-500"
                >
                  <option value={0}>Baseline Demand (No Promotion)</option>
                  <option value={15}>Seasonal Campaign (+15% Demand)</option>
                  <option value={35}>Flash Sale (+35% Demand)</option>
                </select>
              </div>

              {/* Advanced Parameters */}
              <div className="border-t border-zinc-100 dark:border-zinc-800 pt-3">
                <button
                  type="button"
                  onClick={() => setShowAdvanced(!showAdvanced)}
                  className="w-full flex items-center justify-between text-xs text-zinc-500 hover:text-zinc-800 dark:hover:text-zinc-200 transition-colors"
                >
                  <span>Advanced Parameters</span>
                  <span>{showAdvanced ? '−' : '+'}</span>
                </button>

                {showAdvanced && (
                  <div className="space-y-3 mt-3 pt-2">
                    <div>
                      <span className="text-[11px] text-zinc-500 block mb-1">Ordering Cost S (INR / order)</span>
                      <input
                        type="number"
                        min={1}
                        value={orderingCost}
                        onChange={(e) => setOrderingCost(Math.max(1, Number(e.target.value)))}
                        className="w-full bg-zinc-50 dark:bg-zinc-800/60 border border-zinc-200 dark:border-zinc-700/80 rounded px-2.5 py-1 text-xs text-zinc-900 dark:text-zinc-100"
                      />
                    </div>
                    <div>
                      <span className="text-[11px] text-zinc-500 block mb-1">Holding Cost H (INR / unit / year)</span>
                      <input
                        type="number"
                        min={0.1}
                        step={0.5}
                        value={holdingCost}
                        onChange={(e) => setHoldingCost(Math.max(0.1, Number(e.target.value)))}
                        className="w-full bg-zinc-50 dark:bg-zinc-800/60 border border-zinc-200 dark:border-zinc-700/80 rounded px-2.5 py-1 text-xs text-zinc-900 dark:text-zinc-100"
                      />
                    </div>
                    <div>
                      <span className="text-[11px] text-zinc-500 block mb-1">Safety Stock Buffer (Units)</span>
                      <input
                        type="number"
                        min={0}
                        step={50}
                        value={safetyStock}
                        onChange={(e) => setSafetyStock(Math.max(0, Number(e.target.value)))}
                        className="w-full bg-zinc-50 dark:bg-zinc-800/60 border border-zinc-200 dark:border-zinc-700/80 rounded px-2.5 py-1 text-xs text-zinc-900 dark:text-zinc-100"
                      />
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Right: Tabbed Workspace */}
          <div className="lg:col-span-8 space-y-4">
            {/* Clean Tab Switcher */}
            <div className="flex border-b border-zinc-200 dark:border-zinc-800 space-x-1 sm:space-x-4 overflow-x-auto text-xs font-medium">
              <button
                onClick={() => setActiveTab('forecast')}
                className={`py-2.5 px-3 border-b-2 transition-all whitespace-nowrap ${
                  activeTab === 'forecast'
                    ? 'border-zinc-900 dark:border-zinc-100 text-zinc-900 dark:text-zinc-100'
                    : 'border-transparent text-zinc-500 hover:text-zinc-800 dark:hover:text-zinc-200'
                }`}
              >
                Demand Forecast
              </button>

              <button
                onClick={() => setActiveTab('po')}
                className={`py-2.5 px-3 border-b-2 transition-all whitespace-nowrap ${
                  activeTab === 'po'
                    ? 'border-zinc-900 dark:border-zinc-100 text-zinc-900 dark:text-zinc-100'
                    : 'border-transparent text-zinc-500 hover:text-zinc-800 dark:hover:text-zinc-200'
                }`}
              >
                Purchase Order
              </button>

              <button
                onClick={() => setActiveTab('history')}
                className={`py-2.5 px-3 border-b-2 transition-all whitespace-nowrap ${
                  activeTab === 'history'
                    ? 'border-zinc-900 dark:border-zinc-100 text-zinc-900 dark:text-zinc-100'
                    : 'border-transparent text-zinc-500 hover:text-zinc-800 dark:hover:text-zinc-200'
                }`}
              >
                Historical Data
              </button>

              <button
                onClick={() => setActiveTab('methodology')}
                className={`py-2.5 px-3 border-b-2 transition-all whitespace-nowrap ${
                  activeTab === 'methodology'
                    ? 'border-zinc-900 dark:border-zinc-100 text-zinc-900 dark:text-zinc-100'
                    : 'border-transparent text-zinc-500 hover:text-zinc-800 dark:hover:text-zinc-200'
                }`}
              >
                Model & Methodology
              </button>
            </div>

            {/* TAB 1: DEMAND FORECAST */}
            {activeTab === 'forecast' && (
              <div className="bg-white dark:bg-zinc-900/60 rounded-lg p-5 border border-zinc-200 dark:border-zinc-800 shadow-xs space-y-5">
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="text-xs font-semibold text-zinc-900 dark:text-zinc-100 uppercase tracking-wide">
                      Demand Trajectory & Uncertainty
                    </h3>
                    <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-0.5">
                      Log-transformed Prophet model with 95% confidence intervals and historical actuals.
                    </p>
                  </div>
                  <button
                    onClick={handleDownloadForecastCSV}
                    className="inline-flex items-center space-x-1 px-2.5 py-1 rounded bg-zinc-100 dark:bg-zinc-800 hover:bg-zinc-200 dark:hover:bg-zinc-700 text-zinc-700 dark:text-zinc-300 text-xs font-medium transition-colors"
                  >
                    <Download className="w-3 h-3" />
                    <span>Export CSV</span>
                  </button>
                </div>

                {/* Recharts Component */}
                <div className="h-72 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <ComposedChart data={chartData} margin={{ top: 5, right: 5, left: 0, bottom: 0 }}>
                      <defs>
                        <linearGradient id="uncertaintyFill" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor={isDark ? '#6366f1' : '#818cf8'} stopOpacity={0.2} />
                          <stop offset="95%" stopColor={isDark ? '#6366f1' : '#818cf8'} stopOpacity={0.02} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid
                        strokeDasharray="3 3"
                        stroke={isDark ? '#27272a' : '#e4e4e7'}
                        vertical={false}
                      />
                      <XAxis
                        dataKey="date"
                        stroke={isDark ? '#71717a' : '#a1a1aa'}
                        fontSize={11}
                        tickLine={false}
                        minTickGap={35}
                      />
                      <YAxis
                        stroke={isDark ? '#71717a' : '#a1a1aa'}
                        fontSize={11}
                        tickLine={false}
                        tickFormatter={(v) => `${(v / 1000).toFixed(0)}k`}
                      />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: isDark ? '#18181b' : '#ffffff',
                          borderColor: isDark ? '#27272a' : '#e4e4e7',
                          borderRadius: '6px',
                          fontSize: '11px',
                          color: isDark ? '#f4f4f5' : '#09090b',
                          boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
                        }}
                        formatter={(value: any, name: string) => {
                          if (value === null || value === undefined) return ['--', name];
                          const num = Number(value);
                          if (name === 'Historical Sales') return [`${num.toLocaleString()} units`, name];
                          if (name === 'Predicted Demand') return [`${num.toLocaleString()} units`, name];
                          if (name === 'Daily ROP Baseline') return [`${num.toLocaleString()} units/day`, name];
                          if (name === '95% Upper Bound') return [`${num.toLocaleString()} units`, name];
                          return [num.toLocaleString(), name];
                        }}
                      />
                      <Legend
                        verticalAlign="top"
                        height={32}
                        iconType="circle"
                        wrapperStyle={{ fontSize: '11px', color: isDark ? '#a1a1aa' : '#71717a' }}
                      />

                      <Area
                        type="monotone"
                        dataKey="uncertaintyUpper"
                        stroke="none"
                        fill="url(#uncertaintyFill)"
                        name="95% Upper Bound"
                        connectNulls
                      />

                      <Line
                        type="monotone"
                        dataKey="historicalSales"
                        stroke="#6366f1"
                        strokeWidth={1.8}
                        dot={false}
                        name="Historical Sales"
                        connectNulls
                      />

                      <Line
                        type="monotone"
                        dataKey="forecastDemand"
                        stroke="#10b981"
                        strokeWidth={2}
                        dot={{ r: 2, fill: '#10b981' }}
                        name="Predicted Demand"
                        connectNulls
                      />

                      <Line
                        type="monotone"
                        dataKey="dailyRop"
                        stroke="#f43f5e"
                        strokeWidth={1.5}
                        strokeDasharray="3 3"
                        dot={false}
                        name="Daily ROP Baseline"
                      />
                    </ComposedChart>
                  </ResponsiveContainer>
                </div>

                <div className="flex items-center justify-between text-xs text-zinc-500 dark:text-zinc-400 pt-2 border-t border-zinc-100 dark:border-zinc-800">
                  <span>Daily ROP: <strong>{Math.round(metrics.rop / leadTime).toLocaleString()} units/day</strong></span>
                  <button
                    onClick={() => setShowForecastTable(!showForecastTable)}
                    className="text-zinc-700 dark:text-zinc-300 hover:underline"
                  >
                    {showForecastTable ? 'Hide Table' : 'Show Breakdown Table'}
                  </button>
                </div>

                {showForecastTable && (
                  <div className="overflow-x-auto rounded border border-zinc-200 dark:border-zinc-800">
                    <table className="w-full text-left text-xs">
                      <thead className="bg-zinc-50 dark:bg-zinc-800/80 text-zinc-500 dark:text-zinc-400 border-b border-zinc-200 dark:border-zinc-800">
                        <tr>
                          <th className="p-2 font-medium">Date</th>
                          <th className="p-2 font-medium">Predicted (yhat)</th>
                          <th className="p-2 font-medium">Lower 95%</th>
                          <th className="p-2 font-medium">Upper 95%</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-zinc-100 dark:divide-zinc-800 text-zinc-700 dark:text-zinc-300 tabular-nums">
                        {forecastData.map((row, idx) => (
                          <tr key={idx} className="hover:bg-zinc-50 dark:hover:bg-zinc-800/40">
                            <td className="p-2">{row.ds}</td>
                            <td className="p-2 font-medium text-emerald-600 dark:text-emerald-400">{row.yhat.toLocaleString()}</td>
                            <td className="p-2 text-zinc-500">{row.yhat_lower.toLocaleString()}</td>
                            <td className="p-2 text-zinc-500">{row.yhat_upper.toLocaleString()}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            )}

            {/* TAB 2: PURCHASE ORDER */}
            {activeTab === 'po' && (
              <div className="bg-white dark:bg-zinc-900/60 rounded-lg p-5 border border-zinc-200 dark:border-zinc-800 shadow-xs space-y-5">
                <div>
                  <h3 className="text-xs font-semibold text-zinc-900 dark:text-zinc-100 uppercase tracking-wide">
                    Supplier Purchase Order
                  </h3>
                  <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-0.5">
                    Generate an official supplier Excel spreadsheet based on current inventory recommendations.
                  </p>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                  <div>
                    <span className="text-xs text-zinc-500 block mb-1">Supplier / Vendor</span>
                    <input
                      type="text"
                      value={vendorName}
                      onChange={(e) => setVendorName(e.target.value)}
                      className="w-full bg-zinc-50 dark:bg-zinc-800/60 border border-zinc-200 dark:border-zinc-700/80 rounded px-2.5 py-1.5 text-xs text-zinc-900 dark:text-zinc-100"
                    />
                  </div>
                  <div>
                    <span className="text-xs text-zinc-500 block mb-1">Item Description</span>
                    <input
                      type="text"
                      value={itemName}
                      onChange={(e) => setItemName(e.target.value)}
                      className="w-full bg-zinc-50 dark:bg-zinc-800/60 border border-zinc-200 dark:border-zinc-700/80 rounded px-2.5 py-1.5 text-xs text-zinc-900 dark:text-zinc-100"
                    />
                  </div>
                  <div>
                    <span className="text-xs text-zinc-500 block mb-1">Unit Price (INR)</span>
                    <input
                      type="number"
                      min={0.1}
                      value={unitCost}
                      onChange={(e) => setUnitCost(Math.max(0.1, Number(e.target.value)))}
                      className="w-full bg-zinc-50 dark:bg-zinc-800/60 border border-zinc-200 dark:border-zinc-700/80 rounded px-2.5 py-1.5 text-xs text-zinc-900 dark:text-zinc-100"
                    />
                  </div>
                </div>

                {/* PO Document Preview Box */}
                <div className="rounded border border-zinc-200 dark:border-zinc-800 p-4 space-y-3 bg-zinc-50/50 dark:bg-zinc-950/40 text-xs">
                  <div className="flex justify-between items-center border-b border-zinc-200 dark:border-zinc-800 pb-2">
                    <span className="font-semibold text-zinc-900 dark:text-zinc-100">PO Document Preview</span>
                    <span className="text-emerald-700 dark:text-emerald-400 font-medium">Ready for Transmission</span>
                  </div>
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-zinc-600 dark:text-zinc-300">
                    <div>
                      <span className="text-zinc-400 block text-[11px]">Vendor</span>
                      <span className="font-medium text-zinc-900 dark:text-zinc-100">{vendorName}</span>
                    </div>
                    <div>
                      <span className="text-zinc-400 block text-[11px]">Lead Time</span>
                      <span className="font-medium text-zinc-900 dark:text-zinc-100">{leadTime} Days</span>
                    </div>
                    <div>
                      <span className="text-zinc-400 block text-[11px]">Reorder Qty</span>
                      <span className="font-bold text-zinc-900 dark:text-zinc-100">{metrics.recommended_reorder_qty.toLocaleString()} units</span>
                    </div>
                    <div>
                      <span className="text-zinc-400 block text-[11px]">Estimated Cost</span>
                      <span className="font-bold text-emerald-600 dark:text-emerald-400">
                        ₹{(metrics.recommended_reorder_qty * unitCost).toLocaleString(undefined, { minimumFractionDigits: 2 })}
                      </span>
                    </div>
                  </div>
                </div>

                <button
                  onClick={handleDownloadPO}
                  disabled={isDownloadingPO}
                  className="w-full sm:w-auto inline-flex items-center justify-center space-x-1.5 px-4 py-2 rounded-md bg-zinc-900 hover:bg-zinc-800 dark:bg-zinc-100 dark:hover:bg-zinc-200 text-zinc-100 dark:text-zinc-900 text-xs font-medium transition-colors shadow-sm"
                >
                  <FileSpreadsheet className="w-3.5 h-3.5" />
                  <span>{isDownloadingPO ? 'Generating Excel...' : 'Download Official Purchase Order (.xlsx)'}</span>
                </button>
              </div>
            )}

            {/* TAB 3: HISTORICAL DATA */}
            {activeTab === 'history' && (
              <div className="bg-white dark:bg-zinc-900/60 rounded-lg p-5 border border-zinc-200 dark:border-zinc-800 shadow-xs space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="text-xs font-semibold text-zinc-900 dark:text-zinc-100 uppercase tracking-wide">
                      Historical Sales Records
                    </h3>
                    <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-0.5">
                      1,684 daily store observations (2013–2017) used for model calibration.
                    </p>
                  </div>
                  <button
                    onClick={handleDownloadHistoricalCSV}
                    className="inline-flex items-center space-x-1 px-2.5 py-1 rounded bg-zinc-100 dark:bg-zinc-800 hover:bg-zinc-200 dark:hover:bg-zinc-700 text-zinc-700 dark:text-zinc-300 text-xs font-medium transition-colors"
                  >
                    <Download className="w-3 h-3" />
                    <span>Download CSV</span>
                  </button>
                </div>

                {/* Filter and Table */}
                <div className="flex items-center justify-between gap-3">
                  <div className="relative flex-1 max-w-xs">
                    <Search className="w-3.5 h-3.5 text-zinc-400 absolute left-2.5 top-2" />
                    <input
                      type="text"
                      placeholder="Filter by date (e.g. 2017-05)..."
                      value={histSearch}
                      onChange={(e) => {
                        setHistSearch(e.target.value);
                        setHistPage(1);
                      }}
                      className="w-full bg-zinc-50 dark:bg-zinc-800/60 border border-zinc-200 dark:border-zinc-700/80 rounded pl-8 pr-2 py-1 text-xs text-zinc-900 dark:text-zinc-100 focus:outline-none"
                    />
                  </div>
                  <span className="text-xs text-zinc-400">
                    {filteredHistorical.length} Records
                  </span>
                </div>

                <div className="overflow-x-auto rounded border border-zinc-200 dark:border-zinc-800">
                  <table className="w-full text-left text-xs">
                    <thead className="bg-zinc-50 dark:bg-zinc-800/80 text-zinc-500 dark:text-zinc-400 border-b border-zinc-200 dark:border-zinc-800">
                      <tr>
                        <th className="p-2 font-medium">Date</th>
                        <th className="p-2 font-medium">Sales (Units)</th>
                        <th className="p-2 font-medium">Promotion Status</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-zinc-100 dark:divide-zinc-800 text-zinc-700 dark:text-zinc-300 tabular-nums">
                      {pagedHistorical.map((row, idx) => (
                        <tr key={idx} className="hover:bg-zinc-50 dark:hover:bg-zinc-800/40">
                          <td className="p-2">{row.date}</td>
                          <td className="p-2 font-medium text-zinc-900 dark:text-zinc-100">{row.sales.toLocaleString()}</td>
                          <td className="p-2">
                            {row.onpromotion && row.onpromotion > 0 ? (
                              <span className="text-[10px] px-1.5 py-0.5 rounded bg-zinc-200 dark:bg-zinc-700 text-zinc-800 dark:text-zinc-200 font-medium">
                                Active ({row.onpromotion})
                              </span>
                            ) : (
                              <span className="text-zinc-400 text-[11px]">Normal</span>
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                {/* Pagination */}
                <div className="flex items-center justify-between text-xs text-zinc-500 pt-1">
                  <span>Page {histPage} of {totalHistPages}</span>
                  <div className="flex items-center space-x-1">
                    <button
                      onClick={() => setHistPage((p) => Math.max(1, p - 1))}
                      disabled={histPage === 1}
                      className="p-1 rounded border border-zinc-200 dark:border-zinc-800 disabled:opacity-40"
                    >
                      <ChevronLeft className="w-3.5 h-3.5" />
                    </button>
                    <button
                      onClick={() => setHistPage((p) => Math.min(totalHistPages, p + 1))}
                      disabled={histPage === totalHistPages}
                      className="p-1 rounded border border-zinc-200 dark:border-zinc-800 disabled:opacity-40"
                    >
                      <ChevronRight className="w-3.5 h-3.5" />
                    </button>
                  </div>
                </div>
              </div>
            )}

            {/* TAB 4: MODEL & METHODOLOGY */}
            {activeTab === 'methodology' && (
              <div className="bg-white dark:bg-zinc-900/60 rounded-lg p-5 border border-zinc-200 dark:border-zinc-800 shadow-xs space-y-4 text-xs">
                <div>
                  <h3 className="text-xs font-semibold text-zinc-900 dark:text-zinc-100 uppercase tracking-wide">
                    Optimization Formulations
                  </h3>
                  <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-0.5">
                    Mathematical definitions powering replenishment and batch ordering.
                  </p>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-2">
                  <div className="p-3.5 rounded border border-zinc-200 dark:border-zinc-800 bg-zinc-50/50 dark:bg-zinc-950/40 space-y-1.5">
                    <span className="font-semibold text-zinc-900 dark:text-zinc-100 block">
                      Economic Order Quantity (EOQ)
                    </span>
                    <p className="text-zinc-500 text-[11px]">
                      Minimizes the sum of fixed order placement fees and inventory holding expenses:
                    </p>
                    <div className="p-2 bg-white dark:bg-zinc-900 rounded font-mono text-center font-medium text-zinc-900 dark:text-zinc-100">
                      EOQ = √ ( (2 · D · S) / H )
                    </div>
                  </div>

                  <div className="p-3.5 rounded border border-zinc-200 dark:border-zinc-800 bg-zinc-50/50 dark:bg-zinc-950/40 space-y-1.5">
                    <span className="font-semibold text-zinc-900 dark:text-zinc-100 block">
                      Reorder Point (ROP)
                    </span>
                    <p className="text-zinc-500 text-[11px]">
                      Establishes the minimum stock trigger level ensuring operations never breach safety buffer:
                    </p>
                    <div className="p-2 bg-white dark:bg-zinc-900 rounded font-mono text-center font-medium text-zinc-900 dark:text-zinc-100">
                      ROP = (d · L) + SS
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </main>

      {/* Clean Minimalist Footer */}
      <footer className="border-t border-zinc-200 dark:border-zinc-800 py-4 text-center text-xs text-zinc-400 dark:text-zinc-600">
        <span>StockPulse • Demand Forecasting & Inventory Optimizer</span>
      </footer>
    </div>
  );
}
