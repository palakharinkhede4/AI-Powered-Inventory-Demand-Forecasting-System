"use client";

import React, { useState, useEffect, useMemo } from "react";
import Link from "next/link";
import {
  TrendingUp,
  Activity as ActivityIcon,
  Package,
  Clock,
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
  ArrowUpRight,
  Database,
  BookOpen,
  LayoutDashboard,
  ArrowRight,
  Boxes,
} from "lucide-react";
import {
  ResponsiveContainer,
  ComposedChart,
  Line,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ReferenceLine,
} from "recharts";
import ThemePicker from "./ThemePicker";
import * as XLSX from "xlsx";
import { ForecastRequest, ForecastResponse } from "@/lib/types";
import { runForecastSimulation, getHistoricalData } from "@/lib/forecastEngine";

type ThemeMode = "light" | "dark" | "system";

export default function Dashboard() {
  // Theme State
  const [theme, setTheme] = useState<ThemeMode>("system");
  const [isDark, setIsDark] = useState<boolean>(false);

  // Initialize theme from localStorage or system preference
  useEffect(() => {
    try {
      const savedTheme =
        (localStorage.getItem("stockpulse_theme") as ThemeMode) || "system";
      setTheme(savedTheme);
      applyTheme(savedTheme);
    } catch (_) {}

    const mediaQuery = window.matchMedia("(prefers-color-scheme: dark)");
    const handleSystemChange = (e: MediaQueryListEvent) => {
      const currentSaved = localStorage.getItem(
        "stockpulse_theme",
      ) as ThemeMode;
      if (!currentSaved || currentSaved === "system") {
        if (e.matches) {
          document.documentElement.classList.add("dark");
          setIsDark(true);
        } else {
          document.documentElement.classList.remove("dark");
          setIsDark(false);
        }
      }
    };

    mediaQuery.addEventListener("change", handleSystemChange);
    return () => mediaQuery.removeEventListener("change", handleSystemChange);
  }, []);

  const applyTheme = (newTheme: ThemeMode) => {
    try {
      localStorage.setItem("stockpulse_theme", newTheme);
    } catch (_) {}

    const prefersDark = window.matchMedia(
      "(prefers-color-scheme: dark)",
    ).matches;
    const shouldBeDark =
      newTheme === "dark" || (newTheme === "system" && prefersDark);

    if (shouldBeDark) {
      document.documentElement.classList.add("dark");
      setIsDark(true);
    } else {
      document.documentElement.classList.remove("dark");
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
  const [activeTab, setActiveTab] = useState<
    "forecast" | "history" | "po" | "methodology"
  >("forecast");
  const [showAdvanced, setShowAdvanced] = useState<boolean>(false);
  const [showForecastTable, setShowForecastTable] = useState<boolean>(false);

  // PO Generator Form State
  const [vendorName, setVendorName] = useState<string>(
    "Global Logistics Supply Ltd",
  );
  const [itemName, setItemName] = useState<string>(
    "Store 1 — aggregated sales (demo)",
  );
  const [unitCost, setUnitCost] = useState<number>(25.0);
  const [isDownloadingPO, setIsDownloadingPO] = useState<boolean>(false);

  // Historical Data Table State
  const [histSearch, setHistSearch] = useState<string>("");
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
  }, [
    days,
    currentStock,
    leadTime,
    promoScenario,
    orderingCost,
    holdingCost,
    safetyStock,
  ]);

  const metrics = result.metrics;
  const statusInfo = result.status_info;
  const forecastData = result.forecast;
  const historicalTail = result.historical;

  // Load complete historical data
  const allHistorical = useMemo(
    () => [...getHistoricalData()].sort((a, b) => b.date.localeCompare(a.date)),
    [],
  );

  // Filtered historical data
  const filteredHistorical = useMemo(() => {
    if (!histSearch.trim()) return allHistorical;
    const query = histSearch.toLowerCase();
    return allHistorical.filter(
      (r) =>
        r.date.includes(query) ||
        (r.onpromotion ? "promo promotion" : "").includes(query),
    );
  }, [allHistorical, histSearch]);

  const totalHistPages =
    Math.ceil(filteredHistorical.length / rowsPerPage) || 1;
  const pagedHistorical = useMemo(() => {
    const start = (histPage - 1) * rowsPerPage;
    return filteredHistorical.slice(start, start + rowsPerPage);
  }, [filteredHistorical, histPage]);

  // Combine historical tail and forecast for Recharts
  const chartData = useMemo(() => {
    const combined: any[] = [];
    const dailyRopBaseline =
      leadTime > 0
        ? Math.round(metrics.rop / leadTime)
        : metrics.avg_daily_demand;

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
        uncertaintyRange: [item.yhat_lower, item.yhat_upper],
        dailyRop: dailyRopBaseline,
      });
    });

    return combined;
  }, [
    historicalTail,
    forecastData,
    metrics.rop,
    leadTime,
    metrics.avg_daily_demand,
  ]);

  const StatusIcon = {
    Red: AlertCircle,
    Yellow: AlertTriangle,
    Green: CheckCircle2,
    Blue: Info,
  }[statusInfo.level];

  const statusCopy = {
    CRITICAL_STOCKOUT: {
      title: "Stock needs attention",
      message:
        "Stock may run out before your next delivery. Prepare a replenishment order.",
    },
    REORDER_WARNING: {
      title: "Time to reorder",
      message: "Stock has reached the reorder point. Plan your next delivery.",
    },
    OPTIMAL_HEALTHY: {
      title: "Stock is in a healthy range",
      message:
        "You have enough stock to cover expected demand during the supplier lead time.",
    },
    OVERSTOCKED: {
      title: "More stock than needed",
      message:
        "Stock is above the target range. Review demand before placing another order.",
    },
  }[statusInfo.status];
  const tabs = [
    { id: "forecast", label: "Overview", Icon: LayoutDashboard },
    { id: "history", label: "Sales history", Icon: Database },
    { id: "po", label: "Purchase order", Icon: FileSpreadsheet },
    { id: "methodology", label: "How it works", Icon: BookOpen },
  ] as const;
  const formatDate = (date: string) =>
    new Date(date + "T00:00:00").toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
      year: "numeric",
    });
  const historyStart = allHistorical[allHistorical.length - 1]?.date;
  const historyEnd = allHistorical[0]?.date;
  const statusTone = statusInfo.level.toLowerCase();

  // Download Historical CSV
  const handleDownloadHistoricalCSV = () => {
    const csvContent =
      "data:text/csv;charset=utf-8," +
      [
        "date,sales,onpromotion",
        ...getHistoricalData().map(
          (r) => `${r.date},${r.sales},${r.onpromotion || 0}`,
        ),
      ].join("\n");
    const link = document.createElement("a");
    link.href = encodeURI(csvContent);
    link.download = "historical_sales.csv";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // Download Forecast CSV
  const handleDownloadForecastCSV = () => {
    const csvContent =
      "data:text/csv;charset=utf-8," +
      [
        "date,predicted_demand,lower_95,upper_95,trend",
        ...forecastData.map(
          (f) => `${f.ds},${f.yhat},${f.yhat_lower},${f.yhat_upper},${f.trend}`,
        ),
      ].join("\n");
    const link = document.createElement("a");
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

      const res = await fetch("/api/po", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (res.ok) {
        const blob = await res.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `Purchase_Order_${vendorName.replace(/\s+/g, "_")}.xlsx`;
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
    const poDate = new Date().toISOString().split("T")[0];
    const totalCost = metrics.recommended_reorder_qty * unitCost;

    const wb = XLSX.utils.book_new();
    const sheetData = [
      ["PURCHASE ORDER", "", "", "", ""],
      ["PO Number:", poNumber, "", "Date:", poDate],
      ["Vendor:", vendorName, "", "Lead Time:", `${leadTime} Days`],
      ["", "", "", "", ""],
      [
        "Item Description",
        "Current Stock",
        "ROP",
        "EOQ",
        "Daily Demand",
        "Order Qty",
        "Unit Price",
        "Total",
      ],
      [
        itemName,
        currentStock,
        metrics.rop,
        metrics.eoq,
        metrics.avg_daily_demand,
        metrics.recommended_reorder_qty,
        unitCost,
        totalCost,
      ],
    ];

    const ws = XLSX.utils.aoa_to_sheet(sheetData);
    XLSX.utils.book_append_sheet(wb, ws, "Purchase Order");
    XLSX.writeFile(
      wb,
      `Purchase_Order_${vendorName.replace(/\s+/g, "_")}.xlsx`,
    );
  };

  return (
    <div className="dashboard-shell">
      <a className="skip-link" href="#workspace">
        Skip to workspace
      </a>
      <aside className="sidebar">
        <Link className="brand" href="/" aria-label="StockPulse home">
          <span className="brand-icon">
            <Boxes size={22} />
          </span>
          <span>
            StockPulse<span className="brand-caption">INVENTORY PLANNING</span>
          </span>
        </Link>
        <div className="sidebar-section-label">WORKSPACE</div>
        <nav className="workspace-nav" aria-label="Workspace">
          {tabs.map(({ id, label, Icon }) => (
            <button
              key={id}
              aria-label={label}
              title={label}
              onClick={() => setActiveTab(id)}
              aria-current={activeTab === id ? "page" : undefined}
              className={activeTab === id ? "active" : ""}
            >
              <Icon size={18} />
              <span>{label}</span>
              {activeTab === id && <span className="nav-indicator" />}
            </button>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <Link href="/about">
            <Info size={17} />
            About the project
            <ArrowUpRight size={14} />
          </Link>
          <span>Built by Palak Harinkhede</span>
        </div>
      </aside>

      <div className="dashboard-body">
        <header className="topbar">
          <div className="breadcrumb">
            Workspace <ChevronRight size={13} />
            <span>{tabs.find((tab) => tab.id === activeTab)?.label}</span>
          </div>
          <div className="topbar-actions">
            <span className="demo-badge">
              <span />
              Demo data
            </span>
            <ThemePicker theme={theme} onChange={handleThemeToggle} />
            <Link
              href="/about"
              className="about-mobile"
              aria-label="About project"
            >
              <Info size={18} />
            </Link>
          </div>
        </header>
        <main id="workspace" className="dashboard-main">
          <section className="page-heading">
            <div>
              <h1>
                {activeTab === "forecast"
                  ? "Inventory overview"
                  : activeTab === "history"
                    ? "Sales history"
                    : activeTab === "po"
                      ? "Purchase order"
                      : "How it works"}
              </h1>
            </div>
            {activeTab !== "po" && (            <button
              className="btn btn-primary"
              onClick={() => setActiveTab("po")}
            >
              <FileSpreadsheet size={16} />
              Create purchase order
              <ArrowUpRight size={15} />
            </button>)}
          </section>

          <section className="demo-context" aria-label="Demo dataset">
            <div className="context-store">
              <span className="context-icon">
                <Database size={19} />
              </span>
              <div>
                <strong>
                  Store 1 <span> / Sample retail store</span>
                </strong>
                <p>
                  Daily sales across all product families ·{" "}
                  {allHistorical.length.toLocaleString("en-US")} records
                </p>
              </div>
            </div>
            <div className="context-dates">
              <span>
                HISTORICAL DATA<strong>Jan 2013 – Aug 2017</strong>
              </span>
              <span>
                FORECAST START<strong>{formatDate(forecastData[0].ds)}</strong>
              </span>
            </div>
          </section>

          {activeTab === "forecast" && (
            <section className="metric-grid" aria-label="Planning summary">
              {[
                {
                  label: days + "-day demand",
                  value: metrics.total_expected_demand.toLocaleString("en-US"),
                  unit: "units",
                  detail: "",
                  Icon: TrendingUp,
                  tone: "violet",
                },
                {
                  label: "Average daily demand",
                  value: metrics.avg_daily_demand.toLocaleString("en-US"),
                  unit: "units / day",
                  detail: "",
                  Icon: ActivityIcon,
                  tone: "mint",
                },
                {
                  label: "Stock coverage",
                  value: metrics.days_remaining,
                  unit: "days",
                  detail: leadTime + "-day supplier lead time",
                  Icon: Clock,
                  tone: "amber",
                },
                {
                  label: "Optimal order size",
                  value: metrics.eoq.toLocaleString("en-US"),
                  unit: "units",
                  detail: "Economic order quantity (EOQ)",
                  Icon: Package,
                  tone: "blue",
                },
              ].map(({ label, value, unit, detail, Icon, tone }) => (
                <article className={"metric-card " + tone} key={label}>
                  <div className="metric-top">
                    <span>{label}</span>
                    <span className="metric-icon">
                      <Icon size={17} />
                    </span>
                  </div>
                  <div className="metric-value">
                    {value}
                    <span>{unit}</span>
                  </div>
                  {detail && <p>{detail}</p>}
                </article>
              ))}
            </section>
          )}

          <div
            className={
              "workspace-grid" +
              (activeTab === "history" || activeTab === "methodology"
                ? " wide-workspace"
                : "")
            }
          >
            <div className="workspace-content">
              {activeTab === "forecast" && (
                <section
                  className={"stock-alert " + statusTone}
                  aria-label="Inventory status"
                >
                  <div className="alert-heading">
                    <span className="alert-icon">
                      <StatusIcon size={20} />
                    </span>
                    <div>
                      <h2>{statusCopy.title}</h2>
                      <p>{statusCopy.message}</p>
                    </div>
                  </div>
                  <div className="stock-summary">
                    <div>
                      <span>On hand</span>
                      <strong>
                        {currentStock.toLocaleString("en-US")}{" "}
                        <small>units</small>
                      </strong>
                    </div>
                    <div>
                      <span>Reorder at</span>
                      <strong>
                        {metrics.rop.toLocaleString("en-US")}{" "}
                        <small>units</small>
                      </strong>
                    </div>
                    <div>
                      <span>Suggested order</span>
                      <strong>
                        {metrics.recommended_reorder_qty.toLocaleString(
                          "en-US",
                        )}{" "}
                        <small>units</small>
                      </strong>
                    </div>
                  </div>
                  {metrics.recommended_reorder_qty > 0 && (
                    <button
                      className="alert-action"
                      onClick={() => setActiveTab("po")}
                    >
                      Review order
                      <ArrowRight size={15} />
                    </button>
                  )}
                </section>
              )}

              {activeTab === "forecast" && (
                <>
                  <section className="panel forecast-panel">
                    <div className="panel-heading">
                      <div>
                        <h2>Sales & forecast</h2>
                        <p>
                          90 days of actual sales, followed by your {days}-day
                          forecast.
                        </p>
                      </div>
                      <button
                        className="btn btn-secondary btn-small"
                        onClick={handleDownloadForecastCSV}
                      >
                        <Download size={14} />
                        Export CSV
                      </button>
                    </div>
                    <div className="chart-legend">
                      <span>
                        <i className="legend-dot actual" />
                        Actual sales
                      </span>
                      <span>
                        <i className="legend-dot predicted" />
                        Forecast
                      </span>
                      <span>
                        <i className="legend-area" />
                        95% range
                      </span>
                      <span>
                        <i className="legend-line" />
                        Daily reorder baseline
                      </span>
                    </div>
                    <div className="chart-axis-label">
                      Sales · units per day
                    </div>
                    <div
                      className="forecast-chart"
                      role="img"
                      aria-label={
                        "Daily sales and " +
                        days +
                        "-day demand forecast with a 95% uncertainty range"
                      }
                    >
                      <ResponsiveContainer width="100%" height="100%">
                        <ComposedChart
                          data={chartData}
                          margin={{ top: 10, right: 16, left: -14, bottom: 8 }}
                        >
                          <CartesianGrid
                            strokeDasharray="3 5"
                            stroke={isDark ? "#30364b" : "#e8ebf2"}
                            vertical={false}
                          />
                          <XAxis
                            dataKey="date"
                            stroke={isDark ? "#9ba5bc" : "#818a9e"}
                            axisLine={false}
                            fontSize={12}
                            tickLine={false}
                            minTickGap={38}
                            tickMargin={12}
                            tickFormatter={(v) =>
                              new Date(v + "T00:00:00").toLocaleDateString(
                                "en-US",
                                { month: "short", day: "numeric" },
                              )
                            }
                          />
                          <YAxis
                            stroke={isDark ? "#9ba5bc" : "#818a9e"}
                            axisLine={false}
                            fontSize={12}
                            tickLine={false}
                            tickFormatter={(v) => v / 1000 + "k"}
                          />
                          <Tooltip
                            labelFormatter={(v) => formatDate(String(v))}
                            contentStyle={{
                              background: isDark ? "#20263a" : "#fff",
                              border:
                                "1px solid " + (isDark ? "#3b435b" : "#e5e8f0"),
                              borderRadius: 10,
                              fontSize: 12,
                              color: isDark ? "#eef1fa" : "#20253d",
                            }}
                            formatter={(value: any, name: string) => [
                              Array.isArray(value)
                                ? value
                                    .map((v) =>
                                      Math.round(v).toLocaleString("en-US"),
                                    )
                                    .join(" – ") + " units"
                                : Number(value).toLocaleString("en-US") +
                                  " units",
                              name,
                            ]}
                          />
                          <ReferenceLine
                            x={forecastData[0].ds}
                            stroke={isDark ? "#8790b0" : "#adb4c9"}
                            strokeDasharray="4 4"
                          />
                          <Area
                            isAnimationActive={false}
                            type="monotone"
                            dataKey="uncertaintyRange"
                            fill={isDark ? "#353d64" : "#eceafb"}
                            fillOpacity={1}
                            stroke="none"
                            name="95% range"
                          />
                          <Line
                            isAnimationActive={false}
                            type="monotone"
                            dataKey="historicalSales"
                            stroke={isDark ? "#a69af8" : "#7763ce"}
                            strokeWidth={2}
                            dot={false}
                            name="Actual sales"
                          />
                          <Line
                            isAnimationActive={false}
                            type="monotone"
                            dataKey="forecastDemand"
                            stroke={isDark ? "#62d2b1" : "#258b73"}
                            strokeWidth={2.5}
                            dot={false}
                            name="Forecast"
                          />
                          <Line
                            isAnimationActive={false}
                            type="monotone"
                            dataKey="dailyRop"
                            stroke={isDark ? "#f3be71" : "#c59643"}
                            strokeWidth={1.5}
                            strokeDasharray="5 5"
                            dot={false}
                            name="Daily reorder baseline"
                          />
                        </ComposedChart>
                      </ResponsiveContainer>
                    </div>
                    <div className="chart-footer">
                      <span>
                        Forecast begins{" "}
                        <strong>{formatDate(forecastData[0].ds)}</strong>
                      </span>
                      <button
                        className="text-button"
                        aria-expanded={showForecastTable}
                        onClick={() => setShowForecastTable(!showForecastTable)}
                      >
                        {showForecastTable
                          ? "Hide daily breakdown"
                          : "View daily breakdown"}
                        <ChevronRight size={14} />
                      </button>
                    </div>
                    {showForecastTable && (
                      <div className="table-scroll">
                        <table>
                          <caption className="sr-only">
                            Daily forecast, in units
                          </caption>
                          <thead>
                            <tr>
                              <th>Date</th>
                              <th className="numeric">Expected units</th>
                              <th className="numeric">Lower 95%</th>
                              <th className="numeric">Upper 95%</th>
                            </tr>
                          </thead>
                          <tbody>
                            {forecastData.map((row) => (
                              <tr key={row.ds}>
                                <td>{formatDate(row.ds)}</td>
                                <td className="numeric forecast-number">
                                  {row.yhat.toLocaleString("en-US")}
                                </td>
                                <td className="numeric">
                                  {row.yhat_lower.toLocaleString("en-US")}
                                </td>
                                <td className="numeric">
                                  {row.yhat_upper.toLocaleString("en-US")}
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    )}
                  </section>
                </>
              )}

              {activeTab === "history" && (
                <section className="panel">
                  <div className="panel-heading">
                    <div>
                      <h2>Daily sales</h2>
                      <p>
                        {formatDate(historyStart)} – {formatDate(historyEnd)} ·
                        newest first
                      </p>
                    </div>
                    <button
                      className="btn btn-secondary btn-small"
                      onClick={handleDownloadHistoricalCSV}
                    >
                      <Download size={14} />
                      Export CSV
                    </button>
                  </div>
                  <div className="history-note">
                    Each row is one day of Store 1 sales. “On promotion” counts
                    promoted items in the source data.
                  </div>
                  <div className="history-toolbar">
                    <label className="search-field">
                      <Search size={16} />
                      <input
                        aria-label="Search sales history"
                        placeholder="Search date or promotion…"
                        value={histSearch}
                        onChange={(e) => {
                          setHistSearch(e.target.value);
                          setHistPage(1);
                        }}
                      />
                    </label>
                    <span>
                      {filteredHistorical.length.toLocaleString("en-US")}{" "}
                      records
                    </span>
                  </div>
                  <div className="table-scroll">
                    <table>
                      <thead>
                        <tr>
                          <th>Date</th>
                          <th className="numeric">Units sold</th>
                          <th className="numeric">Items on promotion</th>
                        </tr>
                      </thead>
                      <tbody>
                        {pagedHistorical.map((row) => (
                          <tr key={row.date}>
                            <td>{formatDate(row.date)}</td>
                            <td className="numeric strong">
                              {row.sales.toLocaleString("en-US")}
                            </td>
                            <td className="numeric">
                              <span
                                className={
                                  row.onpromotion ? "promo-pill" : "muted"
                                }
                              >
                                {row.onpromotion
                                  ? row.onpromotion.toLocaleString("en-US")
                                  : "None"}
                              </span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                    {pagedHistorical.length === 0 && (
                      <div className="empty-state">
                        <Search size={24} />
                        <h3>No matching sales records</h3>
                        <p>
                          Try a date such as 2017-08 or the word “promotion”.
                        </p>
                        <button
                          className="text-button"
                          onClick={() => {
                            setHistSearch("");
                            setHistPage(1);
                          }}
                        >
                          Clear search
                        </button>
                      </div>
                    )}
                  </div>
                  <div className="pagination">
                    <span>
                      Page {histPage} of {totalHistPages}
                    </span>
                    <div>
                      <button
                        aria-label="Previous page"
                        className="icon-button"
                        disabled={histPage === 1}
                        onClick={() => setHistPage((p) => Math.max(1, p - 1))}
                      >
                        <ChevronLeft size={16} />
                      </button>
                      <button
                        aria-label="Next page"
                        className="icon-button"
                        disabled={histPage === totalHistPages}
                        onClick={() =>
                          setHistPage((p) => Math.min(totalHistPages, p + 1))
                        }
                      >
                        <ChevronRight size={16} />
                      </button>
                    </div>
                  </div>
                </section>
              )}

              {activeTab === "po" && (
                <section className="panel">
                  <div className="panel-heading">
                    <div>
                      <h2>Order details</h2>
                      <p>
                        Sample supplier details. Edit them before exporting.
                      </p>
                    </div>
                    <span className="neutral-pill">Draft preview</span>
                  </div>
                  <div className="po-form">
                    <label className="field">
                      Supplier
                      <input
                        value={vendorName}
                        onChange={(e) => setVendorName(e.target.value)}
                      />
                    </label>
                    <label className="field">
                      Item description
                      <input
                        value={itemName}
                        onChange={(e) => setItemName(e.target.value)}
                      />
                    </label>
                    <label className="field">
                      Unit price (₹)
                      <input
                        type="number"
                        min={0.1}
                        value={unitCost}
                        onChange={(e) =>
                          setUnitCost(Math.max(0.1, Number(e.target.value)))
                        }
                      />
                    </label>
                  </div>
                  <div className="order-preview">
                    <div className="order-preview-heading">
                      <span className="brand-icon">
                        <Boxes size={20} />
                      </span>
                      <div>
                        <strong>PURCHASE ORDER</strong>
                        <span>Store 1 · Demo workspace</span>
                      </div>
                      <FileSpreadsheet size={24} />
                    </div>
                    <div className="order-parties">
                      <div>
                        <span>SUPPLIER</span>
                        <strong>{vendorName || "Add a supplier"}</strong>
                      </div>
                      <div>
                        <span>DELIVERY LEAD TIME</span>
                        <strong>{leadTime} days</strong>
                      </div>
                    </div>
                    <div className="order-line">
                      <div>
                        <span>ITEM</span>
                        <strong>{itemName || "Add an item"}</strong>
                      </div>
                      <div>
                        <span>QUANTITY</span>
                        <strong>
                          {metrics.recommended_reorder_qty.toLocaleString(
                            "en-US",
                          )}{" "}
                          units
                        </strong>
                      </div>
                      <div>
                        <span>UNIT PRICE</span>
                        <strong>₹{unitCost.toLocaleString("en-US")}</strong>
                      </div>
                    </div>
                    <div className="order-total">
                      <span>Estimated order value</span>
                      <strong>
                        ₹
                        {(
                          metrics.recommended_reorder_qty * unitCost
                        ).toLocaleString("en-IN", { minimumFractionDigits: 2 })}
                      </strong>
                    </div>
                  </div>
                  {metrics.recommended_reorder_qty === 0 && (
                    <p className="history-note">
                      No replenishment is currently recommended. The export will
                      show an order quantity of zero.
                    </p>
                  )}
                  <div className="order-download">
                    <p>Downloads an Excel file for review. No order is sent.</p>
                    <button
                      className="btn btn-primary"
                      onClick={handleDownloadPO}
                      disabled={isDownloadingPO}
                    >
                      <Download size={16} />
                      {isDownloadingPO
                        ? "Preparing file…"
                        : "Download order (.xlsx)"}
                    </button>
                  </div>
                </section>
              )}

              {activeTab === "methodology" && (
                <section className="panel methodology-panel">
                  <div className="panel-heading">
                    <div>
                      <h2>From sales to your next order</h2>
                      <p>
                        Three steps connect the demo data to a replenishment
                        decision.
                      </p>
                    </div>
                    <BookOpen size={22} />
                  </div>
                  <div className="guide-steps">
                    {[
                      {
                        title: "Understand demand",
                        text: "The demo uses historical daily sales from Store 1 and a saved forecast. Choose a planning window and promotion scenario to explore expected sales.",
                      },
                      {
                        title: "Compare stock with demand",
                        text: "Stock coverage is your current stock divided by expected daily demand. The reorder point covers supplier lead time plus your safety stock.",
                      },
                      {
                        title: "Review a suggested order",
                        text: "When stock reaches the reorder point, the recommended quantity restores stock to the reorder point plus an optimal order batch. Review the draft in Purchase order.",
                      },
                    ].map((step, i) => (
                      <div key={step.title}>
                        <span className="demo-step">0{i + 1}</span>
                        <div>
                          <h3>{step.title}</h3>
                          <p>{step.text}</p>
                        </div>
                      </div>
                    ))}
                  </div>
                  <div className="formula-grid">
                    <article>
                      <h3>
                        Optimal order size <span>EOQ</span>
                      </h3>
                      <code>√((2 × D × S) / H)</code>
                      <p>
                        D = demand in the selected window
                        <br />S = cost per order · H = holding cost
                      </p>
                    </article>
                    <article>
                      <h3>
                        Reorder point <span>ROP</span>
                      </h3>
                      <code>(d × L) + SS</code>
                      <p>
                        d = daily demand · L = lead time
                        <br />
                        SS = safety stock buffer
                      </p>
                    </article>
                  </div>
                  <div className="history-note">
                    <strong>Reading the chart</strong>
                    <p>
                      The shaded band shows the lower and upper 95% forecast
                      bounds. The dashed baseline is the reorder point divided
                      by supplier lead time.
                    </p>
                  </div>
                  <div className="history-note">
                    <strong>About this demo</strong>
                    <p>
                      The sales and forecast dates are historical. The estimated
                      stockout date uses today plus stock coverage:{" "}
                      {metrics.stockout_date}. Aggregated store sales illustrate
                      the workflow; the sample order is illustrative.
                    </p>
                  </div>
                  <Link className="text-button" href="/about">
                    Explore the project & architecture
                    <ArrowUpRight size={15} />
                  </Link>
                </section>
              )}
            </div>

            {(activeTab === "forecast" || activeTab === "po") && (
              <aside
                className="planning-panel panel"
                aria-label="Planning inputs"
              >
                <div className="planning-heading">
                  <span className="planning-icon">
                    <Sliders size={18} />
                  </span>
                  <div>
                    <h2>Planning inputs</h2>
                    <span>Updates instantly</span>
                  </div>
                  <button
                    className="icon-button"
                    title="Reset to default values"
                    aria-label="Reset planning inputs"
                    onClick={() => {
                      setDays(30);
                      setCurrentStock(5000);
                      setLeadTime(5);
                      setPromoScenario(0);
                      setOrderingCost(50);
                      setHoldingCost(2);
                      setSafetyStock(500);
                    }}
                  >
                    <RotateCcw size={15} />
                  </button>
                </div>
                <div className="planning-fields">
                  <div className="field">
                    <label htmlFor="horizon">
                      Planning window <span>{days} days</span>
                    </label>
                    <div className="horizon-options">
                      {[7, 14, 30, 60].map((d) => (
                        <button
                          key={d}
                          aria-pressed={days === d}
                          onClick={() => setDays(d)}
                        >
                          {d}d
                        </button>
                      ))}
                    </div>
                    <input
                      id="horizon"
                      aria-label="Planning window in days"
                      type="range"
                      min={7}
                      max={60}
                      step={1}
                      value={days}
                      onChange={(e) => setDays(Number(e.target.value))}
                    />
                  </div>
                  <label className="field">
                    Stock on hand <span className="input-unit">units</span>
                    <input
                      type="number"
                      min={0}
                      step={100}
                      value={currentStock}
                      onChange={(e) =>
                        setCurrentStock(Math.max(0, Number(e.target.value)))
                      }
                    />
                  </label>
                  <label className="field">
                    Supplier lead time <span className="input-unit">days</span>
                    <input
                      type="number"
                      min={1}
                      max={30}
                      step={1}
                      value={leadTime}
                      onChange={(e) =>
                        setLeadTime(Math.max(1, Number(e.target.value)))
                      }
                    />
                  </label>
                  <label className="field">
                    Promotion scenario
                    <select
                      value={promoScenario}
                      onChange={(e) => setPromoScenario(Number(e.target.value))}
                    >
                      <option value={0}>No promotion</option>
                      <option value={15}>Seasonal campaign</option>
                      <option value={35}>Flash sale</option>
                    </select>
                  </label>
                  <div className="advanced-fields">
                    <button
                      className="advanced-toggle"
                      aria-expanded={showAdvanced}
                      aria-controls="advanced-inputs"
                      onClick={() => setShowAdvanced(!showAdvanced)}
                    >
                      Costs & safety stock
                      <span>{showAdvanced ? "−" : "+"}</span>
                    </button>
                    {showAdvanced && (
                      <div id="advanced-inputs">
                        <label className="field">
                          Ordering cost (₹ / order)
                          <input
                            type="number"
                            min={1}
                            value={orderingCost}
                            onChange={(e) =>
                              setOrderingCost(
                                Math.max(1, Number(e.target.value)),
                              )
                            }
                          />
                        </label>
                        <label className="field">
                          Holding cost (₹ / unit / period)
                          <input
                            type="number"
                            min={0.1}
                            step={0.5}
                            value={holdingCost}
                            onChange={(e) =>
                              setHoldingCost(
                                Math.max(0.1, Number(e.target.value)),
                              )
                            }
                          />
                        </label>
                        <label className="field">
                          Safety stock (units)
                          <input
                            type="number"
                            min={0}
                            step={50}
                            value={safetyStock}
                            onChange={(e) =>
                              setSafetyStock(
                                Math.max(0, Number(e.target.value)),
                              )
                            }
                          />
                        </label>
                      </div>
                    )}
                  </div>
                </div>
              </aside>
            )}
          </div>
          <footer className="dashboard-footer">
            <span>StockPulse</span>
            <div>
              <Link href="/about">About</Link>
              <a
                href="https://palakharinkhede4.github.io/"
                target="_blank"
                rel="noopener noreferrer"
              >
                Portfolio
              </a>
              <a
                href="https://www.linkedin.com/in/palakharinkhede/"
                target="_blank"
                rel="noopener noreferrer"
              >
                LinkedIn
              </a>
              <a
                href="https://github.com/palakharinkhede4/AI-Powered-Inventory-Demand-Forecasting-System"
                target="_blank"
                rel="noopener noreferrer"
              >
                GitHub
              </a>
            </div>
          </footer>
        </main>
      </div>
    </div>
  );
}
