"""
FailSight Dashboard — Streamlit
Live KPIs, per-unit RUL, alerts, Weibull plots, AI agent chat
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from config import (
    DATA_PROCESSED_DIR, REPORTS_DIR,
    RUL_CRITICAL_THRESHOLD, RUL_WARNING_THRESHOLD
)
from models.train_rul import predict_rul
from analysis.spare_parts import forecast_demand, weekly_demand_curve

st.set_page_config(
    page_title="FailSight",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #1e1e2e;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #313244;
    }
    .metric-value { font-size: 2.5rem; font-weight: 700; }
    .metric-label { font-size: 0.85rem; color: #a6adc8; margin-top: 4px; }
    .critical { color: #f38ba8; }
    .warning  { color: #fab387; }
    .healthy  { color: #a6e3a1; }
    .stChatMessage { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)


# ── Data loader ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_data():
    path = os.path.join(DATA_PROCESSED_DIR, "features.parquet")
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    latest = df.sort_values("cycle").groupby("unit").last().reset_index()
    preds = predict_rul(latest)
    latest["predicted_rul"] = np.clip(preds, 0, None)

    def classify(rul):
        if rul <= RUL_CRITICAL_THRESHOLD:
            return "CRITICAL"
        elif rul <= RUL_WARNING_THRESHOLD:
            return "WARNING"
        return "HEALTHY"

    latest["status"] = latest["predicted_rul"].apply(classify)
    return df, latest


# ── Sidebar ─────────────────────────────────────────────────────────────────
st.sidebar.markdown("## ⚡ FailSight")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", ["Dashboard", "Unit Explorer", "Reliability Analysis", "AI Agent"])

subset_filter = st.sidebar.multiselect(
    "Filter by Subset",
    options=["FD001", "FD002", "FD003", "FD004"],
    default=["FD001"]
)
horizon = st.sidebar.slider("Demand Forecast Horizon (cycles)", 10, 100, 50)

# ── Load data ────────────────────────────────────────────────────────────────
result = load_data()
if result is None:
    st.error("No processed data found. Run the pipeline first: `python pipelines/ingest.py && python pipelines/features.py && python models/train_rul.py`")
    st.stop()

df_full, latest = result

if subset_filter:
    latest_f = latest[latest["subset"].isin(subset_filter)]
else:
    latest_f = latest


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: Dashboard
# ══════════════════════════════════════════════════════════════════════════════
if page == "Dashboard":
    st.title("⚡ FailSight — Fleet Health Dashboard")
    st.caption("Predictive Maintenance & Reliability Intelligence")

    # KPI Row
    n_total = len(latest_f)
    n_critical = (latest_f["status"] == "CRITICAL").sum()
    n_warning = (latest_f["status"] == "WARNING").sum()
    n_healthy = (latest_f["status"] == "HEALTHY").sum()
    avg_rul = latest_f["predicted_rul"].mean()

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.markdown(f"""<div class='metric-card'><div class='metric-value'>{n_total}</div><div class='metric-label'>Total Units</div></div>""", unsafe_allow_html=True)
    with k2:
        st.markdown(f"""<div class='metric-card'><div class='metric-value critical'>{n_critical}</div><div class='metric-label'>CRITICAL</div></div>""", unsafe_allow_html=True)
    with k3:
        st.markdown(f"""<div class='metric-card'><div class='metric-value warning'>{n_warning}</div><div class='metric-label'>WARNING</div></div>""", unsafe_allow_html=True)
    with k4:
        st.markdown(f"""<div class='metric-card'><div class='metric-value healthy'>{n_healthy}</div><div class='metric-label'>HEALTHY</div></div>""", unsafe_allow_html=True)
    with k5:
        st.markdown(f"""<div class='metric-card'><div class='metric-value'>{avg_rul:.0f}</div><div class='metric-label'>Avg Fleet RUL</div></div>""", unsafe_allow_html=True)

    st.markdown("---")

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("RUL Distribution")
        fig = px.histogram(
            latest_f, x="predicted_rul", nbins=30,
            color="status",
            color_discrete_map={"CRITICAL": "#f38ba8", "WARNING": "#fab387", "HEALTHY": "#a6e3a1"},
            title="Predicted RUL Distribution Across Fleet"
        )
        fig.add_vline(x=RUL_CRITICAL_THRESHOLD, line_dash="dash", line_color="#f38ba8", annotation_text="Critical")
        fig.add_vline(x=RUL_WARNING_THRESHOLD, line_dash="dash", line_color="#fab387", annotation_text="Warning")
        fig.update_layout(paper_bgcolor="#1e1e2e", plot_bgcolor="#1e1e2e", font_color="#cdd6f4")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Fleet Status Breakdown")
        fig2 = px.pie(
            values=[n_critical, n_warning, n_healthy],
            names=["CRITICAL", "WARNING", "HEALTHY"],
            color_discrete_sequence=["#f38ba8", "#fab387", "#a6e3a1"],
            hole=0.5
        )
        fig2.update_layout(paper_bgcolor="#1e1e2e", font_color="#cdd6f4")
        st.plotly_chart(fig2, use_container_width=True)

    # Alerts table
    st.subheader("🚨 Critical Units — Immediate Attention Required")
    critical_df = latest_f[latest_f["status"] == "CRITICAL"][["unit", "cycle", "predicted_rul", "status", "subset"]].sort_values("predicted_rul")
    if len(critical_df):
        st.dataframe(critical_df.style.applymap(
            lambda v: "color: #f38ba8" if v == "CRITICAL" else "", subset=["status"]
        ), use_container_width=True)
    else:
        st.success("No critical units in selected subsets.")

    # Spare parts demand
    st.subheader(f"📦 Spare Parts Demand (Next {horizon} cycles)")
    try:
        demand_df, summary = forecast_demand(df_full, horizon_cycles=horizon)
        weekly = weekly_demand_curve(demand_df)
        fig3 = px.bar(weekly, x="week_label", y="parts_needed",
                      color_discrete_sequence=["#89b4fa"],
                      title="Projected Part Replacements by Week")
        fig3.update_layout(paper_bgcolor="#1e1e2e", plot_bgcolor="#1e1e2e", font_color="#cdd6f4")
        st.plotly_chart(fig3, use_container_width=True)
    except Exception as e:
        st.warning(f"Spare parts forecast unavailable: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: Unit Explorer
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Unit Explorer":
    st.title("🔍 Unit Explorer")
    unit_ids = sorted(latest_f["unit"].unique().tolist())
    selected_unit = st.selectbox("Select Unit", unit_ids)

    unit_data = df_full[df_full["unit"] == selected_unit].sort_values("cycle")
    unit_latest = latest_f[latest_f["unit"] == selected_unit].iloc[0]

    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted RUL", f"{unit_latest['predicted_rul']:.0f} cycles")
    col2.metric("Status", unit_latest["status"])
    col3.metric("Current Cycle", int(unit_latest["cycle"]))

    sensor_cols = [c for c in unit_data.columns if c.startswith("sensor_") and "_roll" not in c and "_lag" not in c]
    selected_sensor = st.selectbox("Sensor to plot", sensor_cols)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=unit_data["cycle"], y=unit_data[selected_sensor],
                             mode="lines", name=selected_sensor, line=dict(color="#89b4fa")))
    fig.update_layout(
        title=f"Unit {selected_unit} — {selected_sensor} over cycles",
        paper_bgcolor="#1e1e2e", plot_bgcolor="#1e1e2e", font_color="#cdd6f4"
    )
    st.plotly_chart(fig, use_container_width=True)

    if "degradation_index" in unit_data.columns:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=unit_data["cycle"], y=unit_data["degradation_index"],
                                  fill="tozeroy", line=dict(color="#f38ba8"), name="Degradation Index"))
        fig2.update_layout(title="Degradation Index", paper_bgcolor="#1e1e2e",
                           plot_bgcolor="#1e1e2e", font_color="#cdd6f4")
        st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: Reliability Analysis
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Reliability Analysis":
    st.title("📊 Reliability Analysis — Weibull")
    subset = st.selectbox("Subset", ["FD001", "FD002", "FD003", "FD004"])

    if st.button("Run Weibull Analysis"):
        with st.spinner("Fitting Weibull distribution..."):
            from analysis.weibull_analysis import run_weibull_analysis
            results = run_weibull_analysis(subset)
            if results:
                r1, r2, r3, r4 = st.columns(4)
                r1.metric("Shape (β)", results["weibull_shape_beta"])
                r2.metric("Scale (α)", results["weibull_scale_alpha"])
                r3.metric("MTTF (cycles)", results["mttf_cycles"])
                r4.metric("Units Analyzed", results["n_units"])

                if os.path.exists(results["plot_path"]):
                    st.image(results["plot_path"])


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4: AI Agent
# ══════════════════════════════════════════════════════════════════════════════
elif page == "AI Agent":
    st.title("🤖 FailSight AI Agent")
    st.caption("Powered by LangGraph + RAG — ask anything about your fleet")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    query = st.chat_input("Ask about your fleet... (e.g. 'Which units are critical?' or 'Generate a failure report')")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing fleet..."):
                try:
                    from agent.failsight_agent import run_agent
                    response = run_agent(query)
                except Exception as e:
                    response = f"Agent unavailable (check OpenAI API key): {e}"
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})