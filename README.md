<div align="center">

# LeadFlow
**A Systematic Accumulation-to-Expansion Swing Strategy & 3D Market Terminal**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg?style=for-the-badge&logo=python&logoColor=white)](#)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688.svg?style=for-the-badge&logo=fastapi&logoColor=white)](#)
[![React](https://img.shields.io/badge/React-Frontend-20232A.svg?style=for-the-badge&logo=react&logoColor=61DAFB)](#)
[![TypeScript](https://img.shields.io/badge/TypeScript-Ready-blue.svg?style=for-the-badge&logo=typescript&logoColor=white)](#)
[![Pandas](https://img.shields.io/badge/Pandas-Quant_Engine-150458.svg?style=for-the-badge&logo=pandas&logoColor=white)](#)

</div>

---

##  Overview
LeadFlow is an advanced market intelligence system and quantitative research terminal built specifically for the NSE Mid/Smallcap universe (Nifty Midcap 150 + Smallcap 250). It combines a rigorous walk-forward validated backtest engine with a custom lightweight 3D isometric frontend for live forward-testing and market breadth visualization.

The core premise relies on an **accumulation-to-expansion** framework: the system algorithms buy mid/smallcap equities that have already made a strong, visible move (+20% in 10 sessions), wait for the stock to build a nested consolidation base, and enter on the breakout of that base.

##  The Trading Thesis & Geometry
Instead of chasing breakouts naively, LeadFlow detects structural accumulation:
* **The Admission Gate:** Identifies stocks with a +20% move within 10 sessions, sitting within 5% of their 52-week high[cite: 8]. *Validation:* This specific filter produces a **2.54x lift** in the rate of subsequent 30% moves (11.13% vs. 4.38% base rate)[cite: 8].
* **Nested Box Geometry:** The algorithm scans for a "big box" (15-25% range over 10-45 bars) containing a tighter "small box" (5-10% range over 5-22 bars)[cite: 8]. 
* **Dynamic Sizing & Risk:** Positions are sized dynamically on risk-per-share with a strict ATR(14) × 2.5 trailing stop and a 25-bar hard hold cap to prevent time-decaying capital[cite: 8].

## 📊 Backtest Results & Walk-Forward Performance (2022-2026)
The system was rigorously tested across multiple walk-forward folds using continuous compounded conventions and a highly realistic Indian cost model (0.585% round-trip including STT, brokerage, GST, and slippage). 

To model real-world deployment, the strategy is evaluated within a **30/50/20 Book Allocation**: 30% Strategy Sleeve, 50% Arbitrage (6.5% p.a.), and 20% Gold ETF (GOLDBEES).

### Discovery Window 1: 2015-2021
| Metric | Strategy Sleeve | Total Book (30/50/20) |
| :--- | :--- | :--- |
| **Total Return** | 90.36%| 73.76%|
| **CAGR** | 9.97% | 8.50% |
| **Max Drawdown** | -18.28% | -3.55% |
| **Sharpe Ratio** | 0.26 | 0.41 |
| **Win Rate** | 59.4% | - |
| **Profit Factor** | 1.59 | - |
| **Beta (vs Nifty 500)** | 0.219 | 0.053 |

| Metric | Strategy Sleeve | Total Book (30/50/20) |
| :--- | :--- | :--- |
| **Total Return** | 132.68% | 88.82% |
| **CAGR** | 20.46% | 15.04% |
| **Max Drawdown** | -19.80% | -5.65% |
| **Sharpe Ratio** | 0.86| 1.28 |
| **Win Rate** | 57.4% | - |
| **Profit Factor** | 1.57 | - |
| **Beta (vs Nifty 500)** | 0.418 | 0.166|

*Note: The variant (c) exit ladder deployed scales out at +12% (50%), +20% (20%), and trails the remaining 30%.*

## ✨ Key Platform Features
* **Daily Quantitative Scanner:** Automated screening for mid/smallcap accumulation and expansion setups.
* **3D Market Structure UI:** An interactive, canvas-based visual representation of sector towers, market breadth, and live candidates running locally via Vite.
* **Realistic Portfolio Simulation:** The backtester accounts for exact Indian taxation and penalizes gapped fills.
* **Multi-Timeframe Context:** Evaluates daily, weekly, and monthly breakout confluences and drawdown-conditional relative strength[cite: 8].

## 🛠️ Tech Stack Architecture
* **Backend:** Python, FastAPI, Pandas, NumPy, SQLite (Historical scans & portfolio state).
* **Frontend:** TypeScript, React, Vite, Tailwind CSS, HTML5 Canvas (Isometric rendering).

## 💻 Local Development Setup

To run LeadFlow's backend engine and 3D terminal locally on your machine:

**1. Clone the Repository**
```bash
git clone [https://github.com/kanikakataria75-ship-it/leadflow.git](https://github.com/kanikakataria75-ship-it/leadflow.git)
cd leadflow
