'use client';

import React, { useState, useEffect } from 'react';
import Link from 'next/link';
import {
  TrendingUp,
  ArrowLeft,
  Sun,
  Moon,
  Laptop,
  Github,
  Linkedin,
  ExternalLink,
  Layers,
  Cpu,
  Server,
  Database,
  FileSpreadsheet,
  Activity,
  CheckCircle2,
  AlertTriangle,
  AlertCircle,
  Info,
  Sliders,
  Workflow,
  Sparkles,
  Code2,
  GitBranch,
  Zap,
  ShieldCheck,
  ArrowUpRight,
  BookOpen,
  User,
  Package,
  Clock,
  Copy,
  Check,
  FileText,
  BarChart3,
  Calendar,
  Terminal,
} from 'lucide-react';

type ThemeMode = 'light' | 'dark' | 'system';
type ArchView = 'system' | 'pipeline' | 'lifecycle';

export default function AboutPage() {
  // Theme State
  const [theme, setTheme] = useState<ThemeMode>('system');
  const [isDark, setIsDark] = useState<boolean>(false);
  const [copiedLink, setCopiedLink] = useState<string | null>(null);
  const [activeArchView, setActiveArchView] = useState<ArchView>('system');

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

  const copyToClipboard = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopiedLink(id);
    setTimeout(() => setCopiedLink(null), 2000);
  };

  return (
    <div className="min-h-screen bg-zinc-50 dark:bg-zinc-950 text-zinc-900 dark:text-zinc-100 flex flex-col font-sans transition-colors duration-150">
      {/* Navigation Header */}
      <header className="border-b border-zinc-200 dark:border-zinc-800 bg-white/80 dark:bg-zinc-900/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-14 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <Link
              href="/"
              className="inline-flex items-center space-x-1.5 px-2.5 py-1 rounded-md bg-zinc-100 dark:bg-zinc-800 hover:bg-zinc-200 dark:hover:bg-zinc-700 text-zinc-700 dark:text-zinc-300 text-xs font-medium transition-colors"
            >
              <ArrowLeft className="w-3.5 h-3.5" />
              <span>Dashboard</span>
            </Link>
            <div className="h-4 w-px bg-zinc-200 dark:bg-zinc-800 hidden sm:block" />
            <div className="flex items-center space-x-2">
              <div className="p-1 bg-zinc-900 dark:bg-zinc-100 text-zinc-100 dark:text-zinc-900 rounded">
                <TrendingUp className="w-3.5 h-3.5" />
              </div>
              <span className="font-semibold text-xs sm:text-sm tracking-tight text-zinc-900 dark:text-zinc-100">
                StockPulse AI
              </span>
              <span className="text-[11px] px-2 py-0.5 rounded bg-zinc-100 dark:bg-zinc-800 text-zinc-600 dark:text-zinc-400 font-medium border border-zinc-200 dark:border-zinc-700/60 hidden md:inline">
                About & Architecture
              </span>
            </div>
          </div>

          {/* Header Actions: Theme & Links */}
          <div className="flex items-center space-x-2 sm:space-x-3">
            {/* Theme Toggle */}
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

            {/* GitHub Repo Button */}
            <a
              href="https://github.com/palakharinkhede4/AI-Powered-Inventory-Demand-Forecasting-System"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center space-x-1.5 px-3 py-1.5 rounded-md bg-zinc-900 hover:bg-zinc-800 dark:bg-zinc-100 dark:hover:bg-zinc-200 text-zinc-100 dark:text-zinc-900 text-xs font-medium transition-colors shadow-sm"
            >
              <Github className="w-3.5 h-3.5" />
              <span className="hidden sm:inline">GitHub Repo</span>
            </a>
          </div>
        </div>
      </header>

      {/* Main Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-10">
        {/* Hero Section */}
        <section className="space-y-4">
          <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-zinc-100 dark:bg-zinc-800/80 border border-zinc-200 dark:border-zinc-700/80 text-xs text-zinc-700 dark:text-zinc-300">
            <Sparkles className="w-3.5 h-3.5 text-amber-500 dark:text-amber-400" />
            <span className="font-medium">Enterprise AI Demand Forecasting & Inventory Replenishment Platform</span>
          </div>

          <h1 className="text-2xl sm:text-4xl font-bold tracking-tight text-zinc-900 dark:text-zinc-100">
            StockPulse AI • Technical Specification & System Overview
          </h1>

          <p className="text-sm sm:text-base text-zinc-600 dark:text-zinc-400 max-w-3xl leading-relaxed">
            StockPulse AI is an enterprise-grade, high-performance time-series demand forecasting and automated 
            inventory replenishment optimization platform. Engineered with a serverless edge architecture, mathematical 
            optimization matrices (EOQ / ROP), and log-transformed multiplicative decomposition to eliminate stockouts 
            and minimize carrying costs.
          </p>

          {/* Quick Jump Anchor Pills */}
          <div className="flex flex-wrap gap-2 pt-2 text-xs">
            <a
              href="#owner-profile"
              className="px-3 py-1 rounded-md bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 text-zinc-700 dark:text-zinc-300 hover:border-zinc-400 dark:hover:border-zinc-600 transition-colors"
            >
              👤 Project Owner & Creator
            </a>
            <a
              href="#architecture"
              className="px-3 py-1 rounded-md bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 text-zinc-700 dark:text-zinc-300 hover:border-zinc-400 dark:hover:border-zinc-600 transition-colors"
            >
              🏛️ Architecture Diagrams
            </a>
            <a
              href="#how-it-works"
              className="px-3 py-1 rounded-md bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 text-zinc-700 dark:text-zinc-300 hover:border-zinc-400 dark:hover:border-zinc-600 transition-colors"
            >
              ⚙️ How the System Operates
            </a>
            <a
              href="#math-models"
              className="px-3 py-1 rounded-md bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 text-zinc-700 dark:text-zinc-300 hover:border-zinc-400 dark:hover:border-zinc-600 transition-colors"
            >
              📐 Mathematical Formulations
            </a>
            <a
              href="#api-specs"
              className="px-3 py-1 rounded-md bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 text-zinc-700 dark:text-zinc-300 hover:border-zinc-400 dark:hover:border-zinc-600 transition-colors"
            >
              ⚡ Serverless REST API
            </a>
            <a
              href="#links-hub"
              className="px-3 py-1 rounded-md bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 text-zinc-700 dark:text-zinc-300 hover:border-zinc-400 dark:hover:border-zinc-600 transition-colors"
            >
              🔗 Repository & Links
            </a>
          </div>
        </section>

        {/* SECTION 1: OWNER / CREATOR PROFILE */}
        <section id="owner-profile" className="scroll-mt-20 space-y-4">
          <div className="flex items-center space-x-2 border-b border-zinc-200 dark:border-zinc-800 pb-3">
            <User className="w-5 h-5 text-zinc-700 dark:text-zinc-300" />
            <h2 className="text-lg font-bold text-zinc-900 dark:text-zinc-100">
              Project Creator & Owner
            </h2>
          </div>

          <div className="bg-white dark:bg-zinc-900/70 rounded-xl p-6 border border-zinc-200 dark:border-zinc-800 shadow-sm space-y-6">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-6">
              <div className="flex items-start space-x-4 sm:space-x-5">
                <div className="relative group shrink-0">
                  <div className="w-20 h-20 sm:w-24 sm:h-24 rounded-2xl bg-gradient-to-br from-indigo-500 via-purple-500 to-pink-500 p-0.5 shadow-lg overflow-hidden">
                    <img
                      src="/palak.jpg"
                      alt="Palak Harinkhede"
                      className="w-full h-full object-cover rounded-[14px] group-hover:scale-105 transition-transform duration-300"
                    />
                  </div>
                  <div className="absolute -bottom-1 -right-1 p-1 bg-emerald-500 rounded-full border-2 border-white dark:border-zinc-900 shadow-sm" title="Active Project Lead">
                    <div className="w-2 h-2 rounded-full bg-white" />
                  </div>
                </div>
                <div className="space-y-1">
                  <div className="flex items-center space-x-2.5 flex-wrap gap-y-1">
                    <h3 className="text-xl sm:text-2xl font-bold text-zinc-900 dark:text-zinc-100">
                      Palak Harinkhede
                    </h3>
                    <span className="text-[11px] px-2.5 py-0.5 rounded-full bg-emerald-100 dark:bg-emerald-950/60 text-emerald-800 dark:text-emerald-300 font-semibold border border-emerald-200 dark:border-emerald-800/60">
                      Project Lead & Architect
                    </span>
                  </div>
                  <p className="text-xs sm:text-sm text-zinc-600 dark:text-zinc-400 font-medium">
                    AI/ML Systems Engineer • Supply Chain Analytics Specialist • Full-Stack Developer
                  </p>
                  <p className="text-xs text-zinc-500 dark:text-zinc-400 max-w-2xl pt-1">
                    Specialized in time-series predictive modeling, statistical forecasting algorithms, serverless 
                    cloud architectures, and full-stack web applications for operational research and logistics optimization.
                  </p>
                </div>
              </div>

              {/* Direct Profile Action Badges */}
              <div className="flex flex-wrap md:flex-col gap-2 shrink-0">
                <a
                  href="https://palakharinkhede4.github.io/"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center space-x-2 px-3.5 py-2 rounded-lg bg-purple-600 hover:bg-purple-700 text-white text-xs font-medium transition-all shadow-xs"
                >
                  <User className="w-4 h-4" />
                  <span>Personal Portfolio</span>
                  <ExternalLink className="w-3 h-3 opacity-70" />
                </a>

                <a
                  href="https://www.linkedin.com/in/palakharinkhede/"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center space-x-2 px-3.5 py-2 rounded-lg bg-[#0077B5] hover:bg-[#006097] text-white text-xs font-medium transition-all shadow-xs"
                >
                  <Linkedin className="w-4 h-4" />
                  <span>LinkedIn Profile</span>
                  <ExternalLink className="w-3 h-3 opacity-70" />
                </a>

                <a
                  href="https://github.com/palakharinkhede4"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center space-x-2 px-3.5 py-2 rounded-lg bg-zinc-900 hover:bg-zinc-800 dark:bg-zinc-100 dark:hover:bg-zinc-200 text-zinc-100 dark:text-zinc-900 text-xs font-medium transition-all shadow-xs"
                >
                  <Github className="w-4 h-4" />
                  <span>GitHub Profile</span>
                  <ExternalLink className="w-3 h-3 opacity-70" />
                </a>

                <a
                  href="https://github.com/palakharinkhede4/AI-Powered-Inventory-Demand-Forecasting-System"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center space-x-2 px-3.5 py-2 rounded-lg bg-zinc-100 dark:bg-zinc-800 hover:bg-zinc-200 dark:hover:bg-zinc-700 text-zinc-800 dark:text-zinc-200 text-xs font-medium border border-zinc-200 dark:border-zinc-700 transition-all"
                >
                  <Code2 className="w-4 h-4 text-indigo-500" />
                  <span>Project Repository</span>
                  <ExternalLink className="w-3 h-3 opacity-70" />
                </a>
              </div>
            </div>

            {/* Core Domain Skills Strip */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-4 border-t border-zinc-100 dark:border-zinc-800 text-xs">
              <div className="p-3 rounded-lg bg-zinc-50 dark:bg-zinc-800/40 border border-zinc-200/70 dark:border-zinc-800">
                <span className="text-[11px] text-zinc-500 block font-medium">Predictive ML</span>
                <span className="font-semibold text-zinc-900 dark:text-zinc-100 mt-0.5 block">Time-Series & Prophet</span>
              </div>
              <div className="p-3 rounded-lg bg-zinc-50 dark:bg-zinc-800/40 border border-zinc-200/70 dark:border-zinc-800">
                <span className="text-[11px] text-zinc-500 block font-medium">Operations Research</span>
                <span className="font-semibold text-zinc-900 dark:text-zinc-100 mt-0.5 block">EOQ, ROP & Safety Buffer</span>
              </div>
              <div className="p-3 rounded-lg bg-zinc-50 dark:bg-zinc-800/40 border border-zinc-200/70 dark:border-zinc-800">
                <span className="text-[11px] text-zinc-500 block font-medium">Full-Stack Tech</span>
                <span className="font-semibold text-zinc-900 dark:text-zinc-100 mt-0.5 block">Next.js 14, React & TS</span>
              </div>
              <div className="p-3 rounded-lg bg-zinc-50 dark:bg-zinc-800/40 border border-zinc-200/70 dark:border-zinc-800">
                <span className="text-[11px] text-zinc-500 block font-medium">Deployment Target</span>
                <span className="font-semibold text-zinc-900 dark:text-zinc-100 mt-0.5 block">Vercel Edge (Zero Sleep)</span>
              </div>
            </div>
          </div>
        </section>

        {/* SECTION 2: ARCHITECTURE DIAGRAMS (INTERACTIVE MULTI-VIEW) */}
        <section id="architecture" className="scroll-mt-20 space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-zinc-200 dark:border-zinc-800 pb-3">
            <div className="flex items-center space-x-2">
              <Layers className="w-5 h-5 text-indigo-600 dark:text-indigo-400" />
              <h2 className="text-lg font-bold text-zinc-900 dark:text-zinc-100">
                System Architecture & Visual Diagrams
              </h2>
            </div>

            {/* Architecture View Selector */}
            <div className="flex items-center p-1 rounded-lg bg-zinc-100 dark:bg-zinc-800 text-xs font-medium">
              <button
                onClick={() => setActiveArchView('system')}
                className={`px-3 py-1 rounded-md transition-all ${
                  activeArchView === 'system'
                    ? 'bg-white dark:bg-zinc-700 text-zinc-900 dark:text-zinc-100 shadow-xs'
                    : 'text-zinc-500 hover:text-zinc-900 dark:hover:text-zinc-200'
                }`}
              >
                1. System Layers
              </button>
              <button
                onClick={() => setActiveArchView('pipeline')}
                className={`px-3 py-1 rounded-md transition-all ${
                  activeArchView === 'pipeline'
                    ? 'bg-white dark:bg-zinc-700 text-zinc-900 dark:text-zinc-100 shadow-xs'
                    : 'text-zinc-500 hover:text-zinc-900 dark:hover:text-zinc-200'
                }`}
              >
                2. Algorithmic Pipeline
              </button>
              <button
                onClick={() => setActiveArchView('lifecycle')}
                className={`px-3 py-1 rounded-md transition-all ${
                  activeArchView === 'lifecycle'
                    ? 'bg-white dark:bg-zinc-700 text-zinc-900 dark:text-zinc-100 shadow-xs'
                    : 'text-zinc-500 hover:text-zinc-900 dark:hover:text-zinc-200'
                }`}
              >
                3. Replenishment Cycle
              </button>
            </div>
          </div>

          {/* VIEW 1: SYSTEM CLOUD & EDGE ARCHITECTURE */}
          {activeArchView === 'system' && (
            <div className="bg-white dark:bg-zinc-900/70 rounded-xl p-6 border border-zinc-200 dark:border-zinc-800 shadow-sm space-y-6">
              <div className="space-y-1">
                <span className="text-xs font-semibold text-indigo-600 dark:text-indigo-400 uppercase tracking-wider">
                  Tiered Serverless Architecture
                </span>
                <h3 className="text-base font-bold text-zinc-900 dark:text-zinc-100">
                  Vercel Edge Global Infrastructure & Next.js App Router
                </h3>
                <p className="text-xs text-zinc-500 dark:text-zinc-400">
                  Zero cold-start serverless platform with sub-10ms edge distribution and client-side reactive rendering.
                </p>
              </div>

              {/* Visual Architecture Diagram Graphic */}
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4 pt-2">
                {/* Layer 1: Client UI */}
                <div className="p-4 rounded-xl border border-blue-200 dark:border-blue-900/60 bg-blue-50/40 dark:bg-blue-950/20 space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-[11px] font-bold px-2 py-0.5 rounded bg-blue-100 dark:bg-blue-900/60 text-blue-800 dark:text-blue-300">
                      Tier 1: Client
                    </span>
                    <Laptop className="w-4 h-4 text-blue-600 dark:text-blue-400" />
                  </div>
                  <h4 className="text-xs font-bold text-zinc-900 dark:text-zinc-100">Next.js UI & Recharts</h4>
                  <ul className="text-[11px] text-zinc-600 dark:text-zinc-400 space-y-1.5 list-disc list-inside">
                    <li>Interactive KPI Strip</li>
                    <li>Recharts Forecast Area</li>
                    <li>Parametric Simulators</li>
                    <li>XLSX PO Builder</li>
                    <li>System Theme Sync</li>
                  </ul>
                </div>

                {/* Layer 2: Edge Routing */}
                <div className="p-4 rounded-xl border border-purple-200 dark:border-purple-900/60 bg-purple-50/40 dark:bg-purple-950/20 space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-[11px] font-bold px-2 py-0.5 rounded bg-purple-100 dark:bg-purple-900/60 text-purple-800 dark:text-purple-300">
                      Tier 2: Edge API
                    </span>
                    <Zap className="w-4 h-4 text-purple-600 dark:text-purple-400" />
                  </div>
                  <h4 className="text-xs font-bold text-zinc-900 dark:text-zinc-100">Serverless REST API</h4>
                  <ul className="text-[11px] text-zinc-600 dark:text-zinc-400 space-y-1.5 list-disc list-inside">
                    <li><code className="text-[10px] bg-purple-100/80 dark:bg-purple-900/50 px-1 py-0.2 rounded">/api/forecast</code> (POST)</li>
                    <li><code className="text-[10px] bg-purple-100/80 dark:bg-purple-900/50 px-1 py-0.2 rounded">/api/po</code> (POST)</li>
                    <li><code className="text-[10px] bg-purple-100/80 dark:bg-purple-900/50 px-1 py-0.2 rounded">/api/history</code> (GET)</li>
                    <li>Vercel Edge Global CDN</li>
                  </ul>
                </div>

                {/* Layer 3: Simulation & ML Engine */}
                <div className="p-4 rounded-xl border border-indigo-200 dark:border-indigo-900/60 bg-indigo-50/40 dark:bg-indigo-950/20 space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-[11px] font-bold px-2 py-0.5 rounded bg-indigo-100 dark:bg-indigo-900/60 text-indigo-800 dark:text-indigo-300">
                      Tier 3: Analytics
                    </span>
                    <Cpu className="w-4 h-4 text-indigo-600 dark:text-indigo-400" />
                  </div>
                  <h4 className="text-xs font-bold text-zinc-900 dark:text-zinc-100">Forecasting & Optimization</h4>
                  <ul className="text-[11px] text-zinc-600 dark:text-zinc-400 space-y-1.5 list-disc list-inside">
                    <li>Prophet Time-Series Math</li>
                    <li>Fourier Yearly Harmonics</li>
                    <li>EOQ Batch Optimizer</li>
                    <li>ROP Dynamic Trigger</li>
                    <li>SheetJS Excel Streamer</li>
                  </ul>
                </div>

                {/* Layer 4: Dataset & Persistence */}
                <div className="p-4 rounded-xl border border-emerald-200 dark:border-emerald-900/60 bg-emerald-50/40 dark:bg-emerald-950/20 space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-[11px] font-bold px-2 py-0.5 rounded bg-emerald-100 dark:bg-emerald-900/60 text-emerald-800 dark:text-emerald-300">
                      Tier 4: Data
                    </span>
                    <Database className="w-4 h-4 text-emerald-600 dark:text-emerald-400" />
                  </div>
                  <h4 className="text-xs font-bold text-zinc-900 dark:text-zinc-100">Historical Observations</h4>
                  <ul className="text-[11px] text-zinc-600 dark:text-zinc-400 space-y-1.5 list-disc list-inside">
                    <li>1,684 Daily Store Records</li>
                    <li>Promotion Signal Flags</li>
                    <li>In-Memory Static Cache</li>
                    <li>Sub-Millisecond Query</li>
                  </ul>
                </div>
              </div>

              {/* Data Flow Directional Indicators */}
              <div className="p-4 rounded-lg bg-zinc-50 dark:bg-zinc-800/40 border border-zinc-200/80 dark:border-zinc-800 text-xs text-zinc-600 dark:text-zinc-300 flex flex-col md:flex-row items-center justify-between gap-2">
                <span className="font-semibold text-zinc-900 dark:text-zinc-100">Data Flow:</span>
                <div className="flex items-center flex-wrap gap-2 text-[11px]">
                  <span className="px-2 py-0.5 rounded bg-zinc-200 dark:bg-zinc-700">User Parameters</span>
                  <span>➔</span>
                  <span className="px-2 py-0.5 rounded bg-zinc-200 dark:bg-zinc-700">Forecast Engine</span>
                  <span>➔</span>
                  <span className="px-2 py-0.5 rounded bg-zinc-200 dark:bg-zinc-700">Uncertainty Envelope</span>
                  <span>➔</span>
                  <span className="px-2 py-0.5 rounded bg-zinc-200 dark:bg-zinc-700">Health Classifier</span>
                  <span>➔</span>
                  <span className="px-2 py-0.5 rounded bg-zinc-200 dark:bg-zinc-700">Automated PO (.xlsx)</span>
                </div>
              </div>
            </div>
          )}

          {/* VIEW 2: ALGORITHMIC & ML PIPELINE */}
          {activeArchView === 'pipeline' && (
            <div className="bg-white dark:bg-zinc-900/70 rounded-xl p-6 border border-zinc-200 dark:border-zinc-800 shadow-sm space-y-6">
              <div className="space-y-1">
                <span className="text-xs font-semibold text-purple-600 dark:text-purple-400 uppercase tracking-wider">
                  Mathematical Pipeline
                </span>
                <h3 className="text-base font-bold text-zinc-900 dark:text-zinc-100">
                  Decomposed Multiplicative Time-Series & Demand Estimation
                </h3>
                <p className="text-xs text-zinc-500 dark:text-zinc-400">
                  Step-by-step computational journey from raw historical sales to 95% confidence demand trajectories.
                </p>
              </div>

              {/* Pipeline Step Cards */}
              <div className="space-y-3">
                <div className="p-3.5 rounded-lg bg-zinc-50 dark:bg-zinc-800/40 border border-zinc-200 dark:border-zinc-800 flex items-start space-x-3">
                  <div className="p-1.5 rounded bg-zinc-900 dark:bg-zinc-100 text-zinc-100 dark:text-zinc-900 text-xs font-bold shrink-0 mt-0.5">
                    01
                  </div>
                  <div className="space-y-1">
                    <h4 className="text-xs font-bold text-zinc-900 dark:text-zinc-100">
                      Historical Sales Ingestion & Log Normalization
                    </h4>
                    <p className="text-[11px] text-zinc-600 dark:text-zinc-400">
                      Ingests 1,684 daily store sales records. Applies log transform <code className="font-mono bg-zinc-200 dark:bg-zinc-700 px-1 py-0.2 rounded">ln(1 + y)</code> to stabilize variance across high-volume promotional spikes.
                    </p>
                  </div>
                </div>

                <div className="p-3.5 rounded-lg bg-zinc-50 dark:bg-zinc-800/40 border border-zinc-200 dark:border-zinc-800 flex items-start space-x-3">
                  <div className="p-1.5 rounded bg-zinc-900 dark:bg-zinc-100 text-zinc-100 dark:text-zinc-900 text-xs font-bold shrink-0 mt-0.5">
                    02
                  </div>
                  <div className="space-y-1">
                    <h4 className="text-xs font-bold text-zinc-900 dark:text-zinc-100">
                      Multiplicative Trend & Seasonality Decomposition
                    </h4>
                    <p className="text-[11px] text-zinc-600 dark:text-zinc-400">
                      Decomposes signal into three orthogonal components:
                      <strong className="text-zinc-900 dark:text-zinc-100"> Linear Growth Trend g(t)</strong>, 
                      <strong className="text-zinc-900 dark:text-zinc-100"> Fourier Yearly Cycles s(t)</strong>, and 
                      <strong className="text-zinc-900 dark:text-zinc-100"> Weekly Day-of-Week Weights w(t)</strong>.
                    </p>
                  </div>
                </div>

                <div className="p-3.5 rounded-lg bg-zinc-50 dark:bg-zinc-800/40 border border-zinc-200 dark:border-zinc-800 flex items-start space-x-3">
                  <div className="p-1.5 rounded bg-zinc-900 dark:bg-zinc-100 text-zinc-100 dark:text-zinc-900 text-xs font-bold shrink-0 mt-0.5">
                    03
                  </div>
                  <div className="space-y-1">
                    <h4 className="text-xs font-bold text-zinc-900 dark:text-zinc-100">
                      Scenario Multiplier & Uncertainty Envelope Generation
                    </h4>
                    <p className="text-[11px] text-zinc-600 dark:text-zinc-400">
                      Injects promotional demand lift modifiers (+15% campaign, +35% flash sale) and produces 
                      asymmetric 95% uncertainty confidence intervals <code className="font-mono bg-zinc-200 dark:bg-zinc-700 px-1 py-0.2 rounded">[yhat_lower, yhat_upper]</code>.
                    </p>
                  </div>
                </div>

                <div className="p-3.5 rounded-lg bg-zinc-50 dark:bg-zinc-800/40 border border-zinc-200 dark:border-zinc-800 flex items-start space-x-3">
                  <div className="p-1.5 rounded bg-zinc-900 dark:bg-zinc-100 text-zinc-100 dark:text-zinc-900 text-xs font-bold shrink-0 mt-0.5">
                    04
                  </div>
                  <div className="space-y-1">
                    <h4 className="text-xs font-bold text-zinc-900 dark:text-zinc-100">
                      Dynamic Replenishment Formulation (ROP & EOQ)
                    </h4>
                    <p className="text-[11px] text-zinc-600 dark:text-zinc-400">
                      Computes average daily demand velocity <code className="font-mono bg-zinc-200 dark:bg-zinc-700 px-1 py-0.2 rounded">d_avg</code>, 
                      lead-time protective buffer <code className="font-mono bg-zinc-200 dark:bg-zinc-700 px-1 py-0.2 rounded">ROP = (d · L) + SS</code>, 
                      and optimal cost-minimizing order batch <code className="font-mono bg-zinc-200 dark:bg-zinc-700 px-1 py-0.2 rounded">EOQ = √((2·D·S)/H)</code>.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* VIEW 3: INVENTORY REPLENISHMENT LIFECYCLE */}
          {activeArchView === 'lifecycle' && (
            <div className="bg-white dark:bg-zinc-900/70 rounded-xl p-6 border border-zinc-200 dark:border-zinc-800 shadow-sm space-y-6">
              <div className="space-y-1">
                <span className="text-xs font-semibold text-emerald-600 dark:text-emerald-400 uppercase tracking-wider">
                  Supply Chain Operations
                </span>
                <h3 className="text-base font-bold text-zinc-900 dark:text-zinc-100">
                  Continuous Review Inventory Replenishment Loop
                </h3>
                <p className="text-xs text-zinc-500 dark:text-zinc-400">
                  Deterministic flow maintaining continuous operational availability while eliminating stockout risks.
                </p>
              </div>

              {/* Replenishment Grid */}
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                <div className="p-4 rounded-lg bg-zinc-50 dark:bg-zinc-800/40 border border-zinc-200 dark:border-zinc-800 space-y-2">
                  <div className="w-8 h-8 rounded bg-blue-100 dark:bg-blue-900/50 text-blue-700 dark:text-blue-300 flex items-center justify-center font-bold text-xs">
                    1
                  </div>
                  <h4 className="text-xs font-bold text-zinc-900 dark:text-zinc-100">Daily Demand Consumption</h4>
                  <p className="text-[11px] text-zinc-500 dark:text-zinc-400">
                    Warehouse stock depletes each day at rate <code className="font-mono">d_avg</code> units/day based on consumer transactions.
                  </p>
                </div>

                <div className="p-4 rounded-lg bg-zinc-50 dark:bg-zinc-800/40 border border-zinc-200 dark:border-zinc-800 space-y-2">
                  <div className="w-8 h-8 rounded bg-amber-100 dark:bg-amber-900/50 text-amber-700 dark:text-amber-300 flex items-center justify-center font-bold text-xs">
                    2
                  </div>
                  <h4 className="text-xs font-bold text-zinc-900 dark:text-zinc-100">Reorder Trigger (ROP)</h4>
                  <p className="text-[11px] text-zinc-500 dark:text-zinc-400">
                    When Current Stock drops below <code className="font-mono">ROP = (d·L)+SS</code>, system issues replenishment alert.
                  </p>
                </div>

                <div className="p-4 rounded-lg bg-zinc-50 dark:bg-zinc-800/40 border border-zinc-200 dark:border-zinc-800 space-y-2">
                  <div className="w-8 h-8 rounded bg-indigo-100 dark:bg-indigo-900/50 text-indigo-700 dark:text-indigo-300 flex items-center justify-center font-bold text-xs">
                    3
                  </div>
                  <h4 className="text-xs font-bold text-zinc-900 dark:text-zinc-100">Automated PO Dispatch</h4>
                  <p className="text-[11px] text-zinc-500 dark:text-zinc-400">
                    Supplier PO generated with batch size <code className="font-mono">EOQ</code> or deficit coverage to restore optimal inventory.
                  </p>
                </div>

                <div className="p-4 rounded-lg bg-zinc-50 dark:bg-zinc-800/40 border border-zinc-200 dark:border-zinc-800 space-y-2">
                  <div className="w-8 h-8 rounded bg-emerald-100 dark:bg-emerald-900/50 text-emerald-700 dark:text-emerald-300 flex items-center justify-center font-bold text-xs">
                    4
                  </div>
                  <h4 className="text-xs font-bold text-zinc-900 dark:text-zinc-100">Safety Buffer Protected</h4>
                  <p className="text-[11px] text-zinc-500 dark:text-zinc-400">
                    Supplier delivers order in <code className="font-mono">L</code> days without stock level ever penetrating safety buffer <code className="font-mono">SS</code>.
                  </p>
                </div>
              </div>
            </div>
          )}
        </section>

        {/* SECTION 3: HOW THE SYSTEM WORKS (TECHNICAL BREAKDOWN) */}
        <section id="how-it-works" className="scroll-mt-20 space-y-4">
          <div className="flex items-center space-x-2 border-b border-zinc-200 dark:border-zinc-800 pb-3">
            <Workflow className="w-5 h-5 text-zinc-700 dark:text-zinc-300" />
            <h2 className="text-lg font-bold text-zinc-900 dark:text-zinc-100">
              How StockPulse AI Works • Operational Breakdown
            </h2>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Capability 1 */}
            <div className="bg-white dark:bg-zinc-900/70 rounded-xl p-5 border border-zinc-200 dark:border-zinc-800 space-y-3">
              <div className="flex items-center space-x-2">
                <div className="p-1.5 rounded bg-blue-100 dark:bg-blue-950/60 text-blue-700 dark:text-blue-400">
                  <BarChart3 className="w-4 h-4" />
                </div>
                <h3 className="text-xs font-bold text-zinc-900 dark:text-zinc-100 uppercase tracking-wide">
                  1. Multiplicative Decomposed Demand Forecasting
                </h3>
              </div>
              <p className="text-xs text-zinc-600 dark:text-zinc-400 leading-relaxed">
                StockPulse decomposes store retail time-series using Fourier terms for annual seasonality 
                and explicit regression weights for day-of-week demand surges (e.g. weekend retail spikes). 
                Each prediction provides a 95% Bayesian uncertainty interval to protect against unforeseen demand volatility.
              </p>
              <div className="text-[11px] p-2.5 rounded bg-zinc-50 dark:bg-zinc-800/60 font-mono text-zinc-700 dark:text-zinc-300">
                y(t) = g(t) × s(t) × w(t) × (1 + promo_boost) + ε_t
              </div>
            </div>

            {/* Capability 2 */}
            <div className="bg-white dark:bg-zinc-900/70 rounded-xl p-5 border border-zinc-200 dark:border-zinc-800 space-y-3">
              <div className="flex items-center space-x-2">
                <div className="p-1.5 rounded bg-emerald-100 dark:bg-emerald-950/60 text-emerald-700 dark:text-emerald-400">
                  <ShieldCheck className="w-4 h-4" />
                </div>
                <h3 className="text-xs font-bold text-zinc-900 dark:text-zinc-100 uppercase tracking-wide">
                  2. Dynamic 4-Tier Health Status Classification
                </h3>
              </div>
              <p className="text-xs text-zinc-600 dark:text-zinc-400 leading-relaxed">
                Inventory health is continually evaluated against the dynamic Reorder Point (ROP) and daily burn rate:
              </p>
              <div className="grid grid-cols-2 gap-2 text-[11px]">
                <div className="p-2 rounded bg-rose-50 dark:bg-rose-950/30 border border-rose-200 dark:border-rose-900/40 text-rose-800 dark:text-rose-300">
                  <strong>🔴 Critical Stockout:</strong> Stock ≤ 50% lead demand
                </div>
                <div className="p-2 rounded bg-amber-50 dark:bg-amber-950/30 border border-amber-200 dark:border-amber-900/40 text-amber-800 dark:text-amber-300">
                  <strong>🟡 Reorder Point:</strong> Stock ≤ ROP
                </div>
                <div className="p-2 rounded bg-emerald-50 dark:bg-emerald-950/30 border border-emerald-200 dark:border-emerald-900/40 text-emerald-800 dark:text-emerald-300">
                  <strong>🟢 Optimal Healthy:</strong> ROP &lt; Stock ≤ 2.5× ROP
                </div>
                <div className="p-2 rounded bg-sky-50 dark:bg-sky-950/30 border border-sky-200 dark:border-sky-900/40 text-sky-800 dark:text-sky-300">
                  <strong>🔵 Overstocked:</strong> Stock &gt; 2.5× ROP
                </div>
              </div>
            </div>

            {/* Capability 3 */}
            <div className="bg-white dark:bg-zinc-900/70 rounded-xl p-5 border border-zinc-200 dark:border-zinc-800 space-y-3">
              <div className="flex items-center space-x-2">
                <div className="p-1.5 rounded bg-purple-100 dark:bg-purple-950/60 text-purple-700 dark:text-purple-400">
                  <Clock className="w-4 h-4" />
                </div>
                <h3 className="text-xs font-bold text-zinc-900 dark:text-zinc-100 uppercase tracking-wide">
                  3. Stockout Risk Runway Countdown
                </h3>
              </div>
              <p className="text-xs text-zinc-600 dark:text-zinc-400 leading-relaxed">
                By dividing current stock units by the forecast daily sales velocity, StockPulse computes an exact 
                day countdown and calendar date when the warehouse will hit zero inventory if no purchase order is dispatched.
              </p>
              <div className="text-[11px] p-2.5 rounded bg-zinc-50 dark:bg-zinc-800/60 font-mono text-zinc-700 dark:text-zinc-300">
                Runway_Days = ⌊ Current_Stock / Daily_Velocity ⌋
              </div>
            </div>

            {/* Capability 4 */}
            <div className="bg-white dark:bg-zinc-900/70 rounded-xl p-5 border border-zinc-200 dark:border-zinc-800 space-y-3">
              <div className="flex items-center space-x-2">
                <div className="p-1.5 rounded bg-amber-100 dark:bg-amber-950/60 text-amber-700 dark:text-amber-400">
                  <FileSpreadsheet className="w-4 h-4" />
                </div>
                <h3 className="text-xs font-bold text-zinc-900 dark:text-zinc-100 uppercase tracking-wide">
                  4. Automated Purchase Order Generator (Excel .xlsx)
                </h3>
              </div>
              <p className="text-xs text-zinc-600 dark:text-zinc-400 leading-relaxed">
                Generates formal, supplier-ready purchase orders in 1 click. Builds structured spreadsheet workbooks 
                with customized purchase order numbers, vendor metadata, lead time terms, item quantities, and cost totals.
              </p>
              <div className="text-[11px] p-2.5 rounded bg-zinc-50 dark:bg-zinc-800/60 font-mono text-zinc-700 dark:text-zinc-300">
                SheetJS Engine ➔ binary array buffer ➔ instant .xlsx download
              </div>
            </div>
          </div>
        </section>

        {/* SECTION 4: MATHEMATICAL FORMULATIONS & EQUATIONS */}
        <section id="math-models" className="scroll-mt-20 space-y-4">
          <div className="flex items-center space-x-2 border-b border-zinc-200 dark:border-zinc-800 pb-3">
            <Code2 className="w-5 h-5 text-zinc-700 dark:text-zinc-300" />
            <h2 className="text-lg font-bold text-zinc-900 dark:text-zinc-100">
              Mathematical Formulations & Quantitative Models
            </h2>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* EOQ Card */}
            <div className="bg-white dark:bg-zinc-900/70 rounded-xl p-5 border border-zinc-200 dark:border-zinc-800 space-y-3">
              <div className="flex items-center justify-between">
                <h3 className="text-xs font-bold text-zinc-900 dark:text-zinc-100 uppercase tracking-wide">
                  Economic Order Quantity (EOQ)
                </h3>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-zinc-100 dark:bg-zinc-800 text-zinc-600 dark:text-zinc-400">
                  Ford W. Harris (1913)
                </span>
              </div>
              <p className="text-xs text-zinc-600 dark:text-zinc-400">
                Calculates the exact order batch size that minimizes the total annual cost of ordering and holding inventory:
              </p>
              <div className="p-3 bg-zinc-50 dark:bg-zinc-950/60 rounded-lg text-center font-mono text-sm font-bold text-zinc-900 dark:text-zinc-100 border border-zinc-200/70 dark:border-zinc-800">
                EOQ = √ ( (2 · D · S) / H )
              </div>
              <ul className="text-[11px] text-zinc-500 dark:text-zinc-400 space-y-1 list-disc list-inside">
                <li><strong className="text-zinc-800 dark:text-zinc-200">D</strong> = Total expected demand across planning horizon (units)</li>
                <li><strong className="text-zinc-800 dark:text-zinc-200">S</strong> = Fixed ordering fee per purchase order (INR)</li>
                <li><strong className="text-zinc-800 dark:text-zinc-200">H</strong> = Holding / carrying expense per unit per period (INR)</li>
              </ul>
            </div>

            {/* ROP Card */}
            <div className="bg-white dark:bg-zinc-900/70 rounded-xl p-5 border border-zinc-200 dark:border-zinc-800 space-y-3">
              <div className="flex items-center justify-between">
                <h3 className="text-xs font-bold text-zinc-900 dark:text-zinc-100 uppercase tracking-wide">
                  Reorder Point (ROP)
                </h3>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-zinc-100 dark:bg-zinc-800 text-zinc-600 dark:text-zinc-400">
                  Continuous Review Policy
                </span>
              </div>
              <p className="text-xs text-zinc-600 dark:text-zinc-400">
                The critical inventory threshold triggering an automatic replenishment order before stockouts occur:
              </p>
              <div className="p-3 bg-zinc-50 dark:bg-zinc-950/60 rounded-lg text-center font-mono text-sm font-bold text-zinc-900 dark:text-zinc-100 border border-zinc-200/70 dark:border-zinc-800">
                ROP = (d · L) + SS
              </div>
              <ul className="text-[11px] text-zinc-500 dark:text-zinc-400 space-y-1 list-disc list-inside">
                <li><strong className="text-zinc-800 dark:text-zinc-200">d</strong> = Average daily demand velocity (units / day)</li>
                <li><strong className="text-zinc-800 dark:text-zinc-200">L</strong> = Supplier delivery lead time in calendar days</li>
                <li><strong className="text-zinc-800 dark:text-zinc-200">SS</strong> = Safety stock buffer protecting against demand surges</li>
              </ul>
            </div>

            {/* Total Cost Card */}
            <div className="bg-white dark:bg-zinc-900/70 rounded-xl p-5 border border-zinc-200 dark:border-zinc-800 space-y-3">
              <h3 className="text-xs font-bold text-zinc-900 dark:text-zinc-100 uppercase tracking-wide">
                Total Inventory Cost Function
              </h3>
              <p className="text-xs text-zinc-600 dark:text-zinc-400">
                Sum of item purchasing costs, annual order placement costs, and annual inventory holding expenses:
              </p>
              <div className="p-3 bg-zinc-50 dark:bg-zinc-950/60 rounded-lg text-center font-mono text-sm font-bold text-zinc-900 dark:text-zinc-100 border border-zinc-200/70 dark:border-zinc-800">
                TC(Q) = (D · C) + (D / Q · S) + (Q / 2 · H)
              </div>
              <p className="text-[11px] text-zinc-500 dark:text-zinc-400">
                At <code className="font-mono text-xs">Q = EOQ</code>, ordering cost equals holding cost, achieving the global mathematical minimum cost.
              </p>
            </div>

            {/* Safety Stock Card */}
            <div className="bg-white dark:bg-zinc-900/70 rounded-xl p-5 border border-zinc-200 dark:border-zinc-800 space-y-3">
              <h3 className="text-xs font-bold text-zinc-900 dark:text-zinc-100 uppercase tracking-wide">
                Service Level Safety Stock Buffer (SS)
              </h3>
              <p className="text-xs text-zinc-600 dark:text-zinc-400">
                Statistical buffer safeguarding operations against lead-time demand standard deviation:
              </p>
              <div className="p-3 bg-zinc-50 dark:bg-zinc-950/60 rounded-lg text-center font-mono text-sm font-bold text-zinc-900 dark:text-zinc-100 border border-zinc-200/70 dark:border-zinc-800">
                SS = Z_α · σ_d · √L
              </div>
              <p className="text-[11px] text-zinc-500 dark:text-zinc-400">
                <code className="font-mono text-xs">Z_α = 1.65</code> for 95% service level cycle assurance without stockout events.
              </p>
            </div>
          </div>
        </section>

        {/* SECTION 5: SERVERLESS REST API SPECIFICATIONS */}
        <section id="api-specs" className="scroll-mt-20 space-y-4">
          <div className="flex items-center space-x-2 border-b border-zinc-200 dark:border-zinc-800 pb-3">
            <Server className="w-5 h-5 text-zinc-700 dark:text-zinc-300" />
            <h2 className="text-lg font-bold text-zinc-900 dark:text-zinc-100">
              Serverless Edge REST API Endpoints
            </h2>
          </div>

          <div className="bg-white dark:bg-zinc-900/70 rounded-xl border border-zinc-200 dark:border-zinc-800 overflow-hidden shadow-sm">
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead className="bg-zinc-50 dark:bg-zinc-800/80 text-zinc-500 dark:text-zinc-400 border-b border-zinc-200 dark:border-zinc-800 font-medium">
                  <tr>
                    <th className="p-3">Method</th>
                    <th className="p-3">Endpoint</th>
                    <th className="p-3">Description</th>
                    <th className="p-3">Payload Schema</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-zinc-100 dark:divide-zinc-800 text-zinc-700 dark:text-zinc-300">
                  <tr className="hover:bg-zinc-50 dark:hover:bg-zinc-800/40">
                    <td className="p-3">
                      <span className="px-2 py-0.5 rounded bg-blue-100 dark:bg-blue-900/50 text-blue-800 dark:text-blue-300 font-bold font-mono text-[10px]">
                        POST
                      </span>
                    </td>
                    <td className="p-3 font-mono font-medium text-zinc-900 dark:text-zinc-100">/api/forecast</td>
                    <td className="p-3">Executes time-series forecast, returns metrics, ROP, EOQ, uncertainty bounds & health status</td>
                    <td className="p-3 font-mono text-[11px] text-zinc-500">
                      &#123; days, current_stock, lead_time, ordering_cost, holding_cost, safety_stock, promo_boost_pct &#125;
                    </td>
                  </tr>

                  <tr className="hover:bg-zinc-50 dark:hover:bg-zinc-800/40">
                    <td className="p-3">
                      <span className="px-2 py-0.5 rounded bg-blue-100 dark:bg-blue-900/50 text-blue-800 dark:text-blue-300 font-bold font-mono text-[10px]">
                        POST
                      </span>
                    </td>
                    <td className="p-3 font-mono font-medium text-zinc-900 dark:text-zinc-100">/api/po</td>
                    <td className="p-3">Streams formal supplier Excel spreadsheet (.xlsx binary) with custom PO metadata</td>
                    <td className="p-3 font-mono text-[11px] text-zinc-500">
                      &#123; vendor_name, item_name, current_stock, rop, eoq, recommended_qty, unit_cost, avg_daily_demand, lead_time_days &#125;
                    </td>
                  </tr>

                  <tr className="hover:bg-zinc-50 dark:hover:bg-zinc-800/40">
                    <td className="p-3">
                      <span className="px-2 py-0.5 rounded bg-emerald-100 dark:bg-emerald-900/50 text-emerald-800 dark:text-emerald-300 font-bold font-mono text-[10px]">
                        GET
                      </span>
                    </td>
                    <td className="p-3 font-mono font-medium text-zinc-900 dark:text-zinc-100">/api/history</td>
                    <td className="p-3">Returns all 1,684 daily store sales records with promotion flags</td>
                    <td className="p-3 font-mono text-[11px] text-zinc-500">
                      None (Returns JSON Array of &#123; date, sales, onpromotion &#125;)
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </section>

        {/* SECTION 6: REPOSITORY & LINKS HUB */}
        <section id="links-hub" className="scroll-mt-20 space-y-4">
          <div className="flex items-center space-x-2 border-b border-zinc-200 dark:border-zinc-800 pb-3">
            <GitBranch className="w-5 h-5 text-zinc-700 dark:text-zinc-300" />
            <h2 className="text-lg font-bold text-zinc-900 dark:text-zinc-100">
              Project Links & Verified Resources
            </h2>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {/* GitHub Repo Card */}
            <div className="bg-white dark:bg-zinc-900/70 rounded-xl p-5 border border-zinc-200 dark:border-zinc-800 space-y-3">
              <div className="flex items-center space-x-2">
                <Github className="w-4 h-4 text-zinc-900 dark:text-zinc-100" />
                <h3 className="text-xs font-bold text-zinc-900 dark:text-zinc-100 uppercase tracking-wide">
                  GitHub Repository
                </h3>
              </div>
              <p className="text-xs text-zinc-500 dark:text-zinc-400">
                Complete open-source codebase with Next.js 14 frontend, algorithms, and documentation.
              </p>
              <div className="pt-1">
                <a
                  href="https://github.com/palakharinkhede4/AI-Powered-Inventory-Demand-Forecasting-System"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center space-x-1 text-xs font-medium text-indigo-600 dark:text-indigo-400 hover:underline"
                >
                  <span>palakharinkhede4 / AI-Inventory-System</span>
                  <ExternalLink className="w-3 h-3" />
                </a>
              </div>
            </div>

            {/* LinkedIn Profile Card */}
            <div className="bg-white dark:bg-zinc-900/70 rounded-xl p-5 border border-zinc-200 dark:border-zinc-800 space-y-3">
              <div className="flex items-center space-x-2">
                <Linkedin className="w-4 h-4 text-[#0077B5]" />
                <h3 className="text-xs font-bold text-zinc-900 dark:text-zinc-100 uppercase tracking-wide">
                  LinkedIn Network
                </h3>
              </div>
              <p className="text-xs text-zinc-500 dark:text-zinc-400">
                Connect with Palak Harinkhede for professional collaborations, systems engineering, and AI projects.
              </p>
              <div className="pt-1">
                <a
                  href="https://www.linkedin.com/in/palakharinkhede/"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center space-x-1 text-xs font-medium text-[#0077B5] hover:underline"
                >
                  <span>in / palakharinkhede</span>
                  <ExternalLink className="w-3 h-3" />
                </a>
              </div>
            </div>

            {/* Portfolio Card */}
            <div className="bg-white dark:bg-zinc-900/70 rounded-xl p-5 border border-zinc-200 dark:border-zinc-800 space-y-3">
              <div className="flex items-center space-x-2">
                <User className="w-4 h-4 text-purple-600 dark:text-purple-400" />
                <h3 className="text-xs font-bold text-zinc-900 dark:text-zinc-100 uppercase tracking-wide">
                  Developer Portfolio
                </h3>
              </div>
              <p className="text-xs text-zinc-500 dark:text-zinc-400">
                Explore Palak Harinkhede's engineering portfolio, machine learning projects, systems architecture, and live applications.
              </p>
              <div className="pt-1">
                <a
                  href="https://palakharinkhede4.github.io/"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center space-x-1 text-xs font-medium text-purple-600 dark:text-purple-400 hover:underline"
                >
                  <span>palakharinkhede4.github.io</span>
                  <ExternalLink className="w-3 h-3" />
                </a>
              </div>
            </div>
          </div>

          {/* Quick Clone Terminal Snippet */}
          <div className="p-4 rounded-xl bg-zinc-900 text-zinc-100 border border-zinc-800 space-y-2">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <Terminal className="w-4 h-4 text-zinc-400" />
                <span className="text-xs font-mono text-zinc-300">Clone & Run Locally</span>
              </div>
              <button
                onClick={() =>
                  copyToClipboard(
                    'git clone https://github.com/palakharinkhede4/AI-Powered-Inventory-Demand-Forecasting-System.git\ncd AI-Powered-Inventory-Demand-Forecasting-System\nnpm install\nnpm run dev',
                    'clone-cmd'
                  )
                }
                className="text-xs text-zinc-400 hover:text-zinc-100 flex items-center space-x-1 transition-colors"
              >
                {copiedLink === 'clone-cmd' ? (
                  <>
                    <Check className="w-3.5 h-3.5 text-emerald-400" />
                    <span className="text-emerald-400">Copied</span>
                  </>
                ) : (
                  <>
                    <Copy className="w-3.5 h-3.5" />
                    <span>Copy</span>
                  </>
                )}
              </button>
            </div>
            <pre className="font-mono text-xs text-emerald-400 bg-black/50 p-3 rounded-lg overflow-x-auto">
              git clone https://github.com/palakharinkhede4/AI-Powered-Inventory-Demand-Forecasting-System.git{'\n'}
              cd AI-Powered-Inventory-Demand-Forecasting-System{'\n'}
              npm install{'\n'}
              npm run dev
            </pre>
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className="border-t border-zinc-200 dark:border-zinc-800 py-6 mt-12 bg-white/50 dark:bg-zinc-900/50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-4 text-xs text-zinc-500 dark:text-zinc-400">
          <div className="flex items-center space-x-2">
            <span className="font-semibold text-zinc-900 dark:text-zinc-100">StockPulse AI</span>
            <span>•</span>
            <span>Developed by Palak Harinkhede</span>
          </div>

          <div className="flex items-center space-x-4">
            <Link href="/" className="hover:text-zinc-900 dark:hover:text-zinc-100 transition-colors">
              Dashboard
            </Link>
            <a
              href="https://palakharinkhede4.github.io/"
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-zinc-900 dark:hover:text-zinc-100 transition-colors"
            >
              Portfolio
            </a>
            <a
              href="https://www.linkedin.com/in/palakharinkhede/"
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-zinc-900 dark:hover:text-zinc-100 transition-colors"
            >
              LinkedIn
            </a>
            <a
              href="https://github.com/palakharinkhede4/AI-Powered-Inventory-Demand-Forecasting-System"
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-zinc-900 dark:hover:text-zinc-100 transition-colors"
            >
              GitHub Repo
            </a>
          </div>
        </div>
      </footer>
    </div>
  );
}
