# FailSight — Predictive Maintenance & Reliability Intelligence System

End-to-end predictive maintenance system built on NASA CMAPSS sensor telemetry. Implements ML-based Remaining Useful Life (RUL) estimation, Weibull failure distribution analysis, automated data pipelines, and a live monitoring dashboard with KPI tracking, spare parts demand forecasting, and a LangGraph-powered AI agent for natural language fleet querying and automated failure report generation.

## Stack
- **Data & Pipelines:** Python, SQL (SQLite), Pandas, Apache Airflow
- **ML:** LightGBM, Scikit-learn
- **Reliability:** Reliability (Weibull), Matplotlib
- **AI Agent:** LangChain, LangGraph, ChromaDB, OpenAI/Groq
- **Dashboard:** Streamlit
- **Dataset:** NASA CMAPSS Turbofan Engine Degradation

## Features
1. Automated ingestion + cleaning pipeline
2. Feature engineering (rolling stats, lag features, degradation index)
3. RUL prediction model (LightGBM)
4. Weibull failure distribution analysis
5. Spare parts demand forecasting
6. Live Streamlit dashboard with KPIs and alerts
7. LangGraph AI agent — RAG chatbot + auto failure report generator

## Quickstart
```bash
git clone https://github.com/yourusername/failsight-predictive-maintenance
cd failsight-predictive-maintenance
pip install -r requirements.txt
python pipelines/ingest.py
python pipelines/features.py
python models/train_rul.py
streamlit run dashboard/app.py
```