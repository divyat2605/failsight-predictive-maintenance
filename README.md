# ⚡ FailSight — Predictive Maintenance & Reliability Intelligence

<div align="center">

![License](https://img.shields.io/badge/License-MIT-blue)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![Status](https://img.shields.io/badge/Status-Active-green)

**End-to-end ML-powered predictive maintenance system for turbofan engines**

[Features](#-features) • [Quick Start](#-quick-start) • [Architecture](#-architecture) • [Dashboard](#-dashboard) • [Contributing](#-contributing)

</div>

---

## 📋 Overview

FailSight is an intelligent predictive maintenance platform built on **NASA CMAPSS** turbofan engine sensor telemetry. It combines machine learning, reliability engineering, and advanced data visualization to:

- 🎯 **Predict Remaining Useful Life (RUL)** with LightGBM models
- 🔴 **Detect Anomalies** using IsolationForest unsupervised learning
- 📊 **Analyze Reliability** using Weibull failure distributions
- 📈 **Forecast Spare Parts** demand across your fleet
- 💬 **Answer Questions** using LangGraph AI agent with RAG
- ⏰ **Automate Operations** with APScheduler daily pipelines

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Data Pipeline** | Python • Pandas • SQL (SQLite) • APScheduler |
| **ML Models** | LightGBM • Scikit-learn • IsolationForest |
| **Reliability** | Weibull • Statistical Analysis |
| **Dashboard** | Streamlit • Plotly • Pandas |
| **AI Agent** | LangChain • LangGraph • ChromaDB • OpenAI/Groq |
| **Dataset** | NASA CMAPSS (Turbofan Degradation) |

---

## ✨ Features

### 🔄 Data Pipeline
- ✅ Automated ingestion & cleaning from NASA CMAPSS dataset
- ✅ SQLite database with 160K+ sensor readings
- ✅ Robust error handling and logging

### 🏗️ Feature Engineering
- ✅ Rolling statistics (mean/std over windows [5, 10, 20])
- ✅ Temporal lag features (lags [1, 3, 5])
- ✅ Degradation index (composite sensor metric)
- ✅ Cycle ratio (lifecycle position)
- ✅ Anomaly scores & severity metrics

### 🤖 Machine Learning
- ✅ LightGBM RUL prediction model
- ✅ Train/validate/test split with proper evaluation
- ✅ IsolationForest anomaly detection (contamination=5%)
- ✅ Model persistence with joblib

### 🎯 Reliability Analysis
- ✅ Weibull distribution fitting
- ✅ MTTF (Mean Time To Failure) estimation
- ✅ Hazard rate curves
- ✅ Per-subset failure patterns

### 📊 Demand Forecasting
- ✅ Spare parts replacement prediction
- ✅ Critical/warning threshold-based demand
- ✅ Weekly forecasting windows

### 🚀 Dashboard
- 📱 **Dashboard Page** — Fleet KPIs, status breakdown, RUL distribution
- 🔍 **Data Exploration** — EDA, feature distributions, correlations, time series
- 👁️ **Unit Explorer** — Per-unit deep dive with sensor data & anomalies
- 🔴 **Anomaly Explorer** — Heatmaps, top anomalous units, anomaly-RUL correlation
- 📈 **Reliability Analysis** — Weibull plots, MTTF, hazard rates
- 💬 **AI Agent** — Natural language querying & automated reports

### ⏰ Scheduling
- ✅ Daily automated pipeline runs at midnight
- ✅ Full lifecycle: ingest → features → train → vectorstore
- ✅ Comprehensive logging with timestamps
- ✅ Error handling & failure alerts

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- 2GB+ RAM (for feature engineering)
- ~500MB disk space

### Installation

```bash
# Clone repository
git clone https://github.com/divyat2605/failsight-predictive-maintenance
cd failsight-predictive-maintenance

# Create virtual environment
python -m venv myenv
source myenv/bin/activate  # On Windows: myenv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Pipeline Execution

```bash
# Option 1: Run pipeline steps manually
python pipelines/ingest.py           # Download & process CMAPSS data
python pipelines/features.py         # Engineer features + detect anomalies
python models/train_rul.py           # Train LightGBM model
python agent/build_vectorstore.py    # Build ChromaDB vector store

# Option 2: Run automated daily scheduler
python pipelines/scheduler.py        # Runs all steps daily at midnight
```

### Launch Dashboard

```bash
streamlit run dashboard/app.py
```

Then visit `http://localhost:8501` in your browser.

---

## 📊 Dashboard Overview

### Page 1: Dashboard
- **6 KPI Cards** — Total units, critical/warning/healthy counts, avg RUL, units with anomalies
- **RUL Distribution** — Histogram with threshold markers
- **Fleet Status Pie** — Visual breakdown by status
- **Alerts Table** — Critical units with action items

### Page 2: Data Exploration
- **Dataset Overview** — Statistics, subset distribution, feature breakdown
- **Feature Distributions** — Histograms for sensors, engineered features, RUL
- **Correlation Analysis** — Heatmaps, feature-RUL rankings, sensor correlations
- **Time Series** — Select unit & visualize sensor trends + anomalies
- **Feature Engineering Impact** — Raw vs smoothed signals, rolling statistics demo
- **Anomaly Insights** — Anomaly rates, early warning indicators, anomalous sensor rankings

### Page 3: Unit Explorer
- **Per-Unit Metrics** — RUL, status, cycle count, anomaly rate
- **Sensor Time Series** — Plot any sensor with anomaly overlays
- **Degradation Index** — Composite health trend
- **Multi-Sensor Comparison** — Plot multiple sensors simultaneously

### Page 4: Anomaly Explorer
- **Heatmap** — Anomaly rate across units × subsets
- **Top 10 Table** — Most anomalous units with first anomaly cycle
- **Scatter Plot** — First anomaly cycle vs predicted RUL (early warning)

### Page 5: Reliability Analysis
- **Weibull Fitting** — Shape (β) and scale (α) parameters
- **MTTF Calculation** — Mean time to failure by subset
- **Hazard Rate Curves** — Failure probability over time
- **Per-Subset Analysis** — FD001/FD002/FD003/FD004 comparisons

### Page 6: AI Agent
- **Natural Language Chat** — Ask questions about fleet health
- **Auto Reports** — Generate failure summaries for critical units
- **Context-Aware** — RAG integration with processed data
- **Multi-Turn Dialogue** — Conversation history maintained

---

## 📁 Project Structure

```
failsight-predictive-maintenance/
├── config.py                    # Configuration & constants
├── requirements.txt             # Python dependencies
├── README.md                    # This file
│
├── pipelines/
│   ├── ingest.py               # CMAPSS download & preprocessing
│   ├── features.py             # Feature engineering & anomaly detection
│   └── scheduler.py            # Daily pipeline orchestrator (APScheduler)
│
├── models/
│   ├── train_rul.py            # LightGBM model training
│   └── saved/                  # Saved models & artifacts
│
├── analysis/
│   ├── weibull_analysis.py     # Reliability analysis
│   ├── spare_parts.py          # Demand forecasting
│   └── anomaly_detection.py    # IsolationForest anomaly detection
│
├── agent/
│   ├── failsight_agent.py      # LangGraph AI agent
│   ├── build_vectorstore.py    # ChromaDB vector store builder
│   └── vectorstore/            # Persisted embeddings
│
├── dashboard/
│   └── app.py                  # Streamlit multi-page dashboard
│
├── data/
│   ├── raw/                    # CMAPSS dataset
│   └── processed/              # Features, models, database
│
└── dags/
    └── failsight_dag.py        # Airflow DAG (reference)
```

---

## ⚙️ Configuration

Edit `config.py` to customize:

```python
# Paths
DATA_RAW_DIR = "data/raw"
DATA_PROCESSED_DIR = "data/processed"
MODELS_DIR = "models/saved"

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
MAX_RUL = 125

# Thresholds
RUL_CRITICAL_THRESHOLD = 30
RUL_WARNING_THRESHOLD = 60

# Anomaly detection
CONTAMINATION_RATE = 0.05  # 5% of cycles flagged
```

---

## 🔄 Pipeline Architecture

```
CMAPSS Download
    ↓
Ingest (ingest.py) — Clean sensors, add RUL
    ↓
Features (features.py) — Rolling stats, lags, degradation, anomalies
    ↓
Train (train_rul.py) — LightGBM model training & validation
    ↓
Vectorstore (build_vectorstore.py) — ChromaDB embeddings for AI agent
    ↓
Dashboard — Live monitoring & insights
```

**Automated by:** `scheduler.py` (runs daily at midnight)

---

## 📊 Sample Outputs

### Model Performance
```
LightGBM RUL Prediction
Training samples: 128,287
Validation samples: 32,072
RMSE: 18.5 cycles
MAE: 12.3 cycles
R² Score: 0.87
```

### Anomaly Detection
```
Total Anomalies: 8,120 (5.06% of cycles)
Units Affected: 256/260 (98%)
Average Anomaly Rate: 5.1%
Most Anomalous Subset: FD003
```

---

## 🔧 Advanced Usage

### Run Specific Pipeline Step

```bash
python pipelines/ingest.py      # Just download & clean
python pipelines/features.py    # Just engineer features
python models/train_rul.py      # Just retrain model
```

### Custom Analysis

```python
import pandas as pd
from analysis.anomaly_detection import get_anomaly_summary

# Load processed data
df = pd.read_parquet("data/processed/features.parquet")

# Get anomaly stats
anomaly_summary = get_anomaly_summary(df)
print(anomaly_summary)
```

### Schedule Custom Times

Edit `pipelines/scheduler.py`:

```python
# Run at 6 AM instead of midnight
trigger = CronTrigger(hour=6, minute=0)
```

---

## 📈 Key Metrics

| Metric | Value |
|--------|-------|
| **Dataset Size** | 160,359 sensor readings |
| **Number of Units** | 260 turbofan engines |
| **Subsets (Conditions)** | 4 (FD001-FD004) |
| **Total Features** | 150+ engineered features |
| **Model Accuracy** | 87% (R² score) |
| **Anomaly Detection Rate** | 5.1% (configurable) |
| **Dashboard Response** | <2s (cached) |

---

## 🤝 Contributing

Contributions welcome! Areas for enhancement:

- [ ] Add LSTM/GRU models for RUL prediction
- [ ] Implement XGBoost & CatBoost comparisons
- [ ] Extended Kalman Filter for RUL
- [ ] Multi-step ahead forecasting
- [ ] Alert notification system (email/Slack)
- [ ] Cloud deployment (AWS/Azure/GCP)
- [ ] REST API for model inference
- [ ] Unit tests & CI/CD pipeline

---

## 📄 License

Apache License Version 2.0
                           
---

## 📞 Contact & Support

**Repository:** https://github.com/divyat2605/failsight-predictive-maintenance

**Issues:** [GitHub Issues](https://github.com/divyat2605/failsight-predictive-maintenance/issues)

---

<div align="center">

Built with ❤️ for predictive maintenance

**FailSight** — Making engines reliable, systematically

</div>