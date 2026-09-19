<div align="center">

# 📈 LeadFlow

**Systematic accumulation-to-expansion swing trading research platform for NSE mid/small-cap stocks.**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg?style=for-the-badge&logo=python&logoColor=white)](#)
[![TypeScript](https://img.shields.io/badge/TypeScript-Ready-blue.svg?style=for-the-badge&logo=typescript&logoColor=white)](#)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688.svg?style=for-the-badge&logo=fastapi&logoColor=white)](#)
[![React](https://img.shields.io/badge/React_Vite-Frontend-20232A.svg?style=for-the-badge&logo=react&logoColor=61DAFB)](#)

</div>

---

## 🚀 Overview

LeadFlow is an advanced, high-performance market intelligence system built for algorithmic swing trading research. Designed specifically for the Indian equity markets (NSE), it features a daily scanner, a walk-forward validated backtest engine, and portfolio simulation that accounts for realistic Indian trading costs like STT and slippage. 

*Note: This is a private research terminal and quantitative tool, not investment advice.*

## ✨ Key Features

* **Daily Market Scanner:** Automated screening for mid/small-cap accumulation and expansion setups.
* **3D Market Structure UI:** Interactive, canvas-based visual representation of sector performance and market breadth.
* **Walk-Forward Validated Backtesting:** Rigorous historical testing engine to validate trading strategies against out-of-sample data.
* **Realistic Portfolio Simulation:** Incorporates exact Indian taxation (STT, exchange transaction charges, GST, stamp duty) and simulated slippage for highly accurate equity curves.
* **Live Forward-Testing Terminal:** Dashboard for monitoring live actionable candidates and portfolio health.

## 🛠️ Tech Stack

### Backend
* **Python** (Core Logic, Backtesting Engine, API)
* **FastAPI** (High-performance async REST API)
* **Pandas / NumPy** (Quantitative data manipulation and metric calculation)
* **SQLite** (Local database for historical scans and portfolio state)

### Frontend
* **TypeScript / React** (Component-based interactive UI)
* **Vite** (Next-generation lightning-fast frontend tooling)
* **Canvas 2D / Custom Rendering** (For lightweight isometric 3D market visualization)
* **CSS / Tailwind** (Modern, sleek, dark-mode-first styling)

## 💻 Local Development Setup

Follow these steps to run LeadFlow entirely on your local machine.

### 1. Clone the Repository
```bash
git clone [https://github.com/kanikakataria75-ship-it/leadflow.git](https://github.com/kanikakataria75-ship-it/leadflow.git)
cd leadflow
