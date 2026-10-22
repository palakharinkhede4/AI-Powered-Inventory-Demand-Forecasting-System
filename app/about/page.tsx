"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import {
  ArrowLeft,
  ArrowRight,
  ArrowUpRight,
  Github,
  Linkedin,
  ExternalLink,
  Layers,
  Cpu,
  Server,
  Database,
  FileSpreadsheet,
  Package,
  Clock,
  Copy,
  Check,
  Boxes,
} from "lucide-react";
import ThemePicker, { ThemeMode } from "../ThemePicker";

type ArchView = "system" | "pipeline" | "lifecycle";

export default function AboutPage() {
  // Theme State
  const [theme, setTheme] = useState<ThemeMode>("system");
  const [copiedLink, setCopiedLink] = useState<string | null>(null);
  const [activeArchView, setActiveArchView] = useState<ArchView>("system");

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
        } else {
          document.documentElement.classList.remove("dark");
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
    } else {
      document.documentElement.classList.remove("dark");
    }
  };

  const handleThemeToggle = (newTheme: ThemeMode) => {
    setTheme(newTheme);
    applyTheme(newTheme);
  };

  const copyToClipboard = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopiedLink(id);
    setTimeout(() => setCopiedLink(null), 2000);
  };

  const repository =
    "https://github.com/palakharinkhede4/AI-Powered-Inventory-Demand-Forecasting-System";
  const architecture = {
    system: [
      {
        title: "Interactive workspace",
        text: "Next.js, React and Recharts display forecasts, stock health and planning inputs.",
        Icon: Layers,
      },
      {
        title: "API routes",
        text: "Forecast, sales history and Excel purchase-order endpoints.",
        Icon: Server,
      },
      {
        title: "Forecast engine",
        text: "Saved Prophet forecasts, scenario adjustments and EOQ / ROP calculations.",
        Icon: Cpu,
      },
      {
        title: "Historical dataset",
        text: "1,684 daily Store 1 observations, with sales and promotion counts.",
        Icon: Database,
      },
    ],
    pipeline: [
      {
        title: "Prepare sales",
        text: "Aggregate Store 1 sales by day and apply a log transform for model training.",
        Icon: Database,
      },
      {
        title: "Train & forecast",
        text: "Prophet estimates trend and seasonality, then saves baseline forecasts and bounds.",
        Icon: Cpu,
      },
      {
        title: "Explore a scenario",
        text: "The workspace applies the selected horizon and promotional adjustment.",
        Icon: Layers,
      },
      {
        title: "Plan replenishment",
        text: "Calculate reorder point, optimal batch size and recommended quantity.",
        Icon: Package,
      },
    ],
    lifecycle: [
      {
        title: "Track stock",
        text: "Compare stock on hand with the forecast daily sales rate.",
        Icon: Package,
      },
      {
        title: "Check the trigger",
        text: "Reorder point accounts for supplier lead time and safety stock.",
        Icon: Clock,
      },
      {
        title: "Prepare an order",
        text: "Review supplier, quantity and cost in the purchase-order preview.",
        Icon: FileSpreadsheet,
      },
      {
        title: "Export for review",
        text: "Download an Excel file. Sending an order remains a separate step.",
        Icon: Check,
      },
    ],
  };
  const links = [
    {
      id: "repo",
      title: "Project repository",
      label: "Source code & documentation",
      url: repository,
      Icon: Github,
    },
    {
      id: "linkedin",
      title: "LinkedIn",
      label: "Connect with Palak",
      url: "https://www.linkedin.com/in/palakharinkhede/",
      Icon: Linkedin,
    },
    {
      id: "portfolio",
      title: "Portfolio",
      label: "More projects & work",
      url: "https://palakharinkhede4.github.io/",
      Icon: ExternalLink,
    },
  ];

  return (
    <div className="about-page">
      <header className="about-header">
        <Link href="/" className="brand">
          <span className="brand-icon">
            <Boxes size={21} />
          </span>
          <span>StockPulse</span>
        </Link>
        <div>
          <Link href="/" className="btn btn-secondary btn-small">
            <ArrowLeft size={14} />
            Workspace
          </Link>
          <ThemePicker theme={theme} onChange={handleThemeToggle} />
        </div>
      </header>
      <main className="about-main">
        <section className="about-hero">
          <div>
            <span className="eyebrow">BEHIND THE WORKSPACE</span>
            <h1>
              A clearer path from
              <br />
              sales to stock decisions.
            </h1>
            <p>
              StockPulse brings demand forecasts, inventory health and
              purchase-order planning into one workspace.
            </p>
            <a
              href={repository}
              target="_blank"
              rel="noopener noreferrer"
              className="btn btn-primary"
            >
              <Github size={16} />
              Explore the source
              <ArrowUpRight size={15} />
            </a>
          </div>
          <div className="about-hero-mark" aria-hidden="true">
            <Boxes size={90} strokeWidth={1} />
            <span>
              LESS GUESSWORK.
              <br />
              BETTER PLANNING.
            </span>
          </div>
        </section>
        <nav className="about-jump-nav" aria-label="On this page">
          {[
            { id: "owner-profile", label: "Creator" },
            { id: "architecture", label: "Architecture" },
            { id: "how-it-works", label: "Workflow" },
            { id: "math-models", label: "Formulas" },
            { id: "api-specs", label: "API" },
            { id: "links-hub", label: "Links" },
          ].map((item) => (
            <a key={item.id} href={"#" + item.id}>
              {item.label}
            </a>
          ))}
        </nav>

        <section id="owner-profile" className="about-section">
          <div className="section-title">
            <span>01</span>
            <h2>The creator</h2>
          </div>
          <div className="creator-card panel">
            <img
              src="/palak.jpg"
              alt="Palak Harinkhede"
              width={90}
              height={90}
            />
            <div>
              <span className="eyebrow">PROJECT LEAD & DEVELOPER</span>
              <h3>Palak Harinkhede</h3>
              <p>AI / ML systems engineer & full-stack developer.</p>
              <span className="creator-tags">
                Time-series forecasting · Inventory optimization · Web
                applications
              </span>
            </div>
            <a
              href="https://palakharinkhede4.github.io/"
              target="_blank"
              rel="noopener noreferrer"
              className="text-button"
            >
              View portfolio
              <ArrowUpRight size={15} />
            </a>
          </div>
        </section>

        <section id="architecture" className="about-section">
          <div className="section-title">
            <span>02</span>
            <h2>How the pieces connect</h2>
          </div>
          <div className="panel architecture-panel">
            <div
              className="architecture-tabs"
              role="group"
              aria-label="Architecture view"
            >
              {(
                [
                  { id: "system", label: "System layers" },
                  { id: "pipeline", label: "Model pipeline" },
                  { id: "lifecycle", label: "Replenishment cycle" },
                ] as const
              ).map((view) => (
                <button
                  key={view.id}
                  aria-pressed={activeArchView === view.id}
                  onClick={() => setActiveArchView(view.id)}
                >
                  {view.label}
                </button>
              ))}
            </div>
            <div className="architecture-cards">
              {architecture[activeArchView].map(({ title, text, Icon }, i) => (
                <article key={title}>
                  <div>
                    <span>0{i + 1}</span>
                    <Icon size={21} />
                  </div>
                  <h3>{title}</h3>
                  <p>{text}</p>
                </article>
              ))}
            </div>
            <div className="architecture-flow">
              <span>Sales data</span>
              <ArrowRight size={14} />
              <span>Forecast</span>
              <ArrowRight size={14} />
              <span>Stock health</span>
              <ArrowRight size={14} />
              <span>Order draft</span>
            </div>
          </div>
        </section>

        <section id="how-it-works" className="about-section">
          <div className="section-title">
            <span>03</span>
            <h2>One planning workflow</h2>
          </div>
          <div className="about-workflow">
            {[
              {
                Icon: Database,
                title: "Start with the sample",
                text: "The demo contains aggregated Store 1 sales from January 2013 to August 2017. The saved forecast begins August 16, 2017.",
              },
              {
                Icon: Layers,
                title: "Adjust your inputs",
                text: "Explore 7–60 days of demand. Change current stock, lead time, promotions, costs and safety stock to see the recommendation update.",
              },
              {
                Icon: FileSpreadsheet,
                title: "Review & export",
                text: "Inspect daily forecasts, search sales history, export CSVs or prepare an Excel purchase-order draft.",
              },
            ].map(({ Icon, title, text }) => (
              <article className="panel" key={title}>
                <span className="context-icon">
                  <Icon size={20} />
                </span>
                <h3>{title}</h3>
                <p>{text}</p>
              </article>
            ))}
          </div>
        </section>

        <section id="math-models" className="about-section">
          <div className="section-title">
            <span>04</span>
            <h2>The numbers, explained</h2>
          </div>
          <div className="about-formulas">
            {[
              {
                title: "Optimal order size",
                tag: "EOQ",
                formula: "√((2 × D × S) / H)",
                text: "Balances ordering and holding costs.",
                legend:
                  "D = demand in the selected window · S = cost per order · H = holding cost per unit per period",
              },
              {
                title: "Reorder point",
                tag: "ROP",
                formula: "(d × L) + SS",
                text: "The stock level that triggers replenishment.",
                legend:
                  "d = average daily demand · L = supplier lead time · SS = safety stock",
              },
              {
                title: "Stock coverage",
                tag: "DAYS",
                formula: "Stock on hand / daily demand",
                text: "Estimated time until stock runs out.",
                legend:
                  "The estimated stockout date uses today plus stock coverage. Forecast dates remain historical.",
              },
              {
                title: "Suggested order",
                tag: "UNITS",
                formula: "max(EOQ, ROP + EOQ − stock)",
                text: "Applies when stock is at or below the reorder point.",
                legend:
                  "Rounded up to whole units. When stock is above the reorder point, the recommended order is zero.",
              },
            ].map((f) => (
              <article className="panel" key={f.tag}>
                <div>
                  <h3>{f.title}</h3>
                  <span className="neutral-pill">{f.tag}</span>
                </div>
                <p>{f.text}</p>
                <code>{f.formula}</code>
                <small>{f.legend}</small>
              </article>
            ))}
          </div>
          <div className="about-status-guide">
            <strong>Inventory health</strong>
            <p>
              <span>Critical</span> Stock does not cover lead-time demand.{" "}
              <span>Reorder</span> Stock is above lead-time demand but at or
              below ROP. <span>Healthy</span> Stock is above ROP, up to ROP +
              1.5 × EOQ. <span>Overstocked</span> Stock exceeds that range.
            </p>
          </div>
        </section>

        <section id="api-specs" className="about-section">
          <div className="section-title">
            <span>05</span>
            <h2>API reference</h2>
          </div>
          <div className="panel api-reference">
            {[
              {
                method: "POST",
                path: "/api/forecast",
                text: "Returns demand forecasts, stock metrics and inventory status.",
                code: "{ days, current_stock, lead_time, ordering_cost, holding_cost, safety_stock, promo_boost_pct }",
              },
              {
                method: "GET",
                path: "/api/history",
                text: "Returns all historical sales records and a record count.",
                code: "{ success, total_count, data: [{ date, sales, onpromotion }] }",
              },
              {
                method: "POST",
                path: "/api/po",
                text: "Returns an Excel purchase-order file.",
                code: "{ vendor_name, item_name, current_stock, rop, eoq, recommended_qty, unit_cost, avg_daily_demand, lead_time_days }",
              },
            ].map((api) => (
              <article key={api.path}>
                <div>
                  <span className="method-badge">{api.method}</span>
                  <code>{api.path}</code>
                </div>
                <p>{api.text}</p>
                <pre>{api.code}</pre>
              </article>
            ))}
          </div>
        </section>

        <section id="links-hub" className="about-section">
          <div className="section-title">
            <span>06</span>
            <h2>Explore & connect</h2>
          </div>
          <div className="about-links">
            {links.map(({ id, title, label, url, Icon }) => (
              <article className="panel" key={id}>
                <Icon size={21} />
                <a href={url} target="_blank" rel="noopener noreferrer">
                  <h3>
                    {title}
                    <ArrowUpRight size={15} />
                  </h3>
                  <p>{label}</p>
                </a>
                <button
                  className="icon-button"
                  aria-label={"Copy " + title + " link"}
                  title={"Copy " + title + " link"}
                  onClick={() => copyToClipboard(url, id)}
                >
                  {copiedLink === id ? <Check size={15} /> : <Copy size={15} />}
                </button>
              </article>
            ))}
          </div>
          <span className="copy-feedback" role="status" aria-live="polite">
            {copiedLink ? "Link copied" : ""}
          </span>
        </section>
        <footer className="dashboard-footer">
          <span>StockPulse · Developed by Palak Harinkhede</span>
          <Link className="text-button" href="/">
            Back to workspace
            <ArrowRight size={14} />
          </Link>
        </footer>
      </main>
    </div>
  );
}
