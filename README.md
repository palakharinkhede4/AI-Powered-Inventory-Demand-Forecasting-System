# StockPulse AI | Demand Forecasting & Inventory Replenishment Platform

An enterprise-grade, full-stack AI Inventory Optimization and Time-Series Demand Forecasting platform engineered for high-performance serverless deployment on **Vercel** with 24/7 global availability, zero sleep downtime, and sub-10ms response latency.

---

## Business Impact & Executive Summary

Global retail and supply chain organizations lose over **$1.1 Trillion annually** due to inventory imbalances:
- **Overstocking**: Traps working capital, inflates warehousing holding expenses, and causes product depreciation.
- **Stockouts / Understocking**: Triggers lost sales revenue, backorders, and customer churn.

### Operational Answers Provided by StockPulse AI
1. **Demand Forecasting**: Predicts exact consumer demand across 7 to 60-day horizons with decomposed seasonal trends and 95% confidence intervals.
2. **Stockout Risk Runway**: Dynamically monitors warehouse burn velocity against supplier lead time to identify replenishment triggers before inventory reaches critical safety buffers.
3. **Automated Purchase Order Generation**: Computes cost-minimizing Economic Order Quantities (EOQ) and produces formal, supplier-ready Excel spreadsheets (`.xlsx`) in one click.

---

## Core Capabilities

- **Decomposed Multiplicative Time-Series Modeling**: Log-transformed Prophet algorithm estimating baseline trend, weekly cycles, yearly seasonality, and promotional modifiers with 95% uncertainty intervals (`yhat_lower`, `yhat_upper`).
- **Real-Time Inventory Health Classification**: Four-tier health status classification (Optimal Healthy, Reorder Point Reached, Critical Stockout Risk, Overstocked) with dynamic ROP and EOQ calculations.
- **Interactive Recharts Visualizer**: High-performance interactive chart displaying historical observations, projected demand centerlines, uncertainty envelope shading, and daily ROP thresholds.
- **Historical Sales Data Explorer**: Comprehensive dataset viewer with search filtering, aggregate statistical metrics, pagination, and 1-click CSV dataset export.
- **Automated Purchase Order Builder**: Interactive document preview and instant generation of supplier-ready Excel spreadsheets (`.xlsx`) using SheetJS.
- **Zero-Sleep Serverless Architecture**: Built on Next.js App Router and deployed to Vercel's global edge network, guaranteeing 100% uptime with zero cold starts.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Vercel Edge Platform                     │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │   Next.js Enterprise Frontend (TypeScript / Tailwind) │  │
│  │   - Executive KPI Grid & Runway Countdown             │  │
│  │   - Demand Forecast & Confidence Bounds (Recharts)    │  │
│  │   - Operational Parameter Controls & Scenarios        │  │
│  │   - Historical Sales Data Explorer & CSV Export       │  │
│  │   - Automated Purchase Order Builder (.xlsx Export)   │  │
│  │   - System Architecture & Methodology Documentation   │  │
│  └───────────────────────────┬───────────────────────────┘  │
│                              │ Next.js API Routes           │
│  ┌───────────────────────────▼───────────────────────────┐  │
│  │   High-Performance Serverless Engine                  │  │
│  │   - Time-Series Demand Simulation & Multipliers       │  │
│  │   - EOQ & ROP Inventory Optimization Formulas         │  │
│  │   - Excel (.xlsx) Purchase Order Streamer (SheetJS)   │  │
│  │   - Historical Dataset Query & Filtering Provider     │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## How to Deploy to Vercel (Step-by-Step)

### Option A: 1-Click Deployment via GitHub (Recommended)

1. **Commit and Push Changes to GitHub**:
   ```bash
   git add .
   git commit -m "Migrate to Next.js modern enterprise UI for Vercel"
   git push origin main
   ```

2. **Connect to Vercel**:
   - Go to [vercel.com](https://vercel.com) and log in.
   - Click **"Add New..."** > **"Project"**.
   - Select your GitHub repository (`AI-Inventory-System` / `AI-Powered-Inventory-Demand-Forecasting-System`).
   - Leave the default settings (Framework Preset: **Next.js**, Root Directory: `./`).
   - Click **"Deploy"**.

3. **Deployment Complete**:
   - Your application will be live on a production URL (e.g. `https://ai-inventory-system.vercel.app`) in under 1 minute.
   - It runs 24/7 with zero sleeping and zero cold starts.

---

### Option B: Deploy via Vercel CLI from Terminal

1. **Install Vercel CLI (if not already installed)**:
   ```bash
   npm i -g vercel
   ```

2. **Deploy directly from your project directory**:
   ```bash
   vercel
   ```
   - Follow the interactive prompts (Accept default project settings and link to your Vercel account).

3. **Deploy to Production**:
   ```bash
   vercel --prod
   ```

---

## Local Development Setup

### 1. Install Node.js Dependencies
```bash
npm install
```

### 2. Start the Local Next.js Development Server
```bash
npm run dev
```
Open [http://localhost:3000](http://localhost:3000) in your browser.

### 3. Build for Production Locally
```bash
npm run build
npm run start
```

---

## Technical Specifications & Mathematical Formulations

### 1. Economic Order Quantity (EOQ)
Calculates the optimal order quantity that minimizes the sum of holding costs and ordering costs:
$$\text{EOQ} = \sqrt{\frac{2 \cdot D \cdot S}{H}}$$
- $D$ = Total expected demand over the planning horizon (units)
- $S$ = Fixed cost per purchase order (INR)
- $H$ = Holding cost per unit per period (INR)

### 2. Reorder Point (ROP)
Establishes the minimum stock level that triggers a replenishment order to ensure inventory is never depleted during supplier lead time:
$$\text{ROP} = (d \cdot L) + SS$$
- $d$ = Average daily demand velocity ($\text{units/day}$)
- $L$ = Supplier lead time in days
- $SS$ = Safety stock buffer buffer (units)

---

## Serverless REST API Endpoints

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/api/forecast` | `POST` | Executes demand forecasting simulation, ROP/EOQ optimization, and status classification |
| `/api/po` | `POST` | Generates and streams formal supplier Excel spreadsheets (`.xlsx`) |
| `/api/history` | `GET` | Returns 1,684 daily historical store sales observations with promotion metadata |

---

## Technology Stack

- **Framework**: Next.js 14 (App Router) & React 18
- **Language**: TypeScript
- **Styling**: Tailwind CSS & Glassmorphism Design System
- **Icons**: Lucide React (Clean SVG icons)
- **Data Visualization**: Recharts
- **Spreadsheet Generation**: SheetJS (`xlsx`)
- **Deployment Platform**: Vercel (Edge / Serverless)

---

## Project Creator & Connect

**Palak Harinkhede**  
*AI & Full-Stack Systems Engineer*

- **Portfolio**: [palakharinkhede4.github.io](https://palakharinkhede4.github.io/)
- **LinkedIn**: [linkedin.com/in/palakharinkhede](https://www.linkedin.com/in/palakharinkhede/)
- **GitHub**: [github.com/palakharinkhede4](https://github.com/palakharinkhede4)
- **Repository**: [github.com/palakharinkhede4/AI-Powered-Inventory-Demand-Forecasting-System](https://github.com/palakharinkhede4/AI-Powered-Inventory-Demand-Forecasting-System)
- **Dedicated About Page**: Open `/about` in the application or click **"About Project"** on the top right of the dashboard.