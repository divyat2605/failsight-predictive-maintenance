"""
FailSight Dashboard — Streamlit
Live KPIs, per-unit RUL, alerts, Weibull plots, AI agent chat
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

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
from agent.failsight_agent import run_agent

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
page = st.sidebar.radio("Navigation", ["Dashboard", "Data Exploration", "Unit Explorer", "Anomaly Explorer", "Reliability Analysis", "AI Agent"])

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
    n_anomalous = df_full.groupby("unit")["is_anomaly"].any().sum()

    k1, k2, k3, k4, k5, k6 = st.columns(6)
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
    with k6:
        st.markdown(f"""<div class='metric-card'><div class='metric-value'>{n_anomalous}</div><div class='metric-label'>Units w/ Anomalies</div></div>""", unsafe_allow_html=True)

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
# PAGE 1.5: Data Exploration (EDA & Feature Engineering)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Data Exploration":
    st.title("📊 Data Exploration & Feature Analysis")
    st.caption("Understand preprocessing pipeline, feature engineering, and data distributions")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Dataset Overview",
        "📉 Feature Distributions",
        "🔗 Correlations",
        "📍 Time Series",
        "⚙️ Feature Impact",
        "🔴 Anomaly Insights"
    ])

    # ─ TAB 1: Dataset Overview ─
    with tab1:
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Units", len(df_full["unit"].unique()))
        col2.metric("Total Cycles", len(df_full))
        col3.metric("Features", len([c for c in df_full.columns if c not in ["unit", "cycle", "subset", "split", "rul"]]))
        col4.metric("Subsets", len(df_full["subset"].unique()))
        col5.metric("Memory (MB)", f"{df_full.memory_usage(deep=True).sum() / 1024 / 1024:.1f}")

        st.markdown("### Subset Distribution")
        subset_dist = df_full["subset"].value_counts()
        fig = px.pie(values=subset_dist.values, names=subset_dist.index, hole=0.4,
                     color_discrete_sequence=["#89b4fa", "#f38ba8", "#a6e3a1", "#fab387"])
        fig.update_layout(paper_bgcolor="#1e1e2e", font_color="#cdd6f4")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Feature Categories")
        sensor_cols = [c for c in df_full.columns if c.startswith("sensor_") and "_roll" not in c and "_lag" not in c]
        rolling_cols = [c for c in df_full.columns if "_roll" in c]
        lag_cols = [c for c in df_full.columns if "_lag" in c]
        anomaly_cols = [c for c in df_full.columns if c.startswith("anomaly")]
        derived_cols = [c for c in df_full.columns if c in ["degradation_index", "cycle_ratio"]]

        feat_summary = pd.DataFrame({
            "Feature Type": ["Raw Sensors", "Rolling Statistics", "Lag Features", "Anomaly Features", "Derived Features"],
            "Count": [len(sensor_cols), len(rolling_cols), len(lag_cols), len(anomaly_cols), len(derived_cols)]
        })
        st.dataframe(feat_summary, use_container_width=True, hide_index=True)

    # ─ TAB 2: Feature Distributions ─
    with tab2:
        col_type = st.radio("Select Feature Type", ["Raw Sensors", "Engineered Features", "RUL Target"])

        if col_type == "Raw Sensors":
            sensor_cols_all = [c for c in df_full.columns if c.startswith("sensor_") and "_roll" not in c and "_lag" not in c]
            selected_sensor = st.selectbox("Sensor", sensor_cols_all)
            fig = px.histogram(df_full, x=selected_sensor, nbins=50, color="subset",
                               color_discrete_map={"FD001": "#89b4fa", "FD002": "#f38ba8", "FD003": "#a6e3a1", "FD004": "#fab387"},
                               title=f"Distribution of {selected_sensor}")
            fig.update_layout(paper_bgcolor="#1e1e2e", plot_bgcolor="#1e1e2e", font_color="#cdd6f4")
            st.plotly_chart(fig, use_container_width=True)

        elif col_type == "Engineered Features":
            eng_cols = ["degradation_index", "cycle_ratio"]
            selected_eng = st.selectbox("Engineered Feature", eng_cols)
            fig = px.histogram(df_full, x=selected_eng, nbins=50, color="subset",
                               color_discrete_map={"FD001": "#89b4fa", "FD002": "#f38ba8", "FD003": "#a6e3a1", "FD004": "#fab387"},
                               title=f"Distribution of {selected_eng}")
            fig.update_layout(paper_bgcolor="#1e1e2e", plot_bgcolor="#1e1e2e", font_color="#cdd6f4")
            st.plotly_chart(fig, use_container_width=True)

        elif col_type == "RUL Target":
            fig = px.histogram(df_full, x="rul", nbins=50, color="subset",
                               color_discrete_map={"FD001": "#89b4fa", "FD002": "#f38ba8", "FD003": "#a6e3a1", "FD004": "#fab387"},
                               title="RUL Distribution Across Fleet")
            fig.add_vline(x=RUL_CRITICAL_THRESHOLD, line_dash="dash", line_color="#f38ba8", annotation_text="Critical")
            fig.add_vline(x=RUL_WARNING_THRESHOLD, line_dash="dash", line_color="#fab387", annotation_text="Warning")
            fig.update_layout(paper_bgcolor="#1e1e2e", plot_bgcolor="#1e1e2e", font_color="#cdd6f4")
            st.plotly_chart(fig, use_container_width=True)

    # ─ TAB 3: Correlations ─
    with tab3:
        corr_type = st.radio("Correlation Type", ["All Features vs RUL", "Raw Sensors Only", "Full Feature Heatmap"])

        if corr_type == "All Features vs RUL":
            sensor_cols_all = [c for c in df_full.columns if c.startswith("sensor_") and "_roll" not in c and "_lag" not in c]
            key_features = sensor_cols_all + ["degradation_index", "cycle_ratio", "anomaly_severity"]
            corr_rul = df_full[key_features + ["rul"]].corr()["rul"].drop("rul").sort_values()
            fig = px.bar(x=corr_rul.values, y=corr_rul.index, orientation="h", title="Feature Correlation with RUL",
                         color=corr_rul.values, color_continuous_scale="RdBu", color_continuous_midpoint=0)
            fig.update_layout(paper_bgcolor="#1e1e2e", plot_bgcolor="#1e1e2e", font_color="#cdd6f4")
            st.plotly_chart(fig, use_container_width=True)

        elif corr_type == "Raw Sensors Only":
            sensor_cols_all = [c for c in df_full.columns if c.startswith("sensor_") and "_roll" not in c and "_lag" not in c]
            corr_sensors = df_full[sensor_cols_all].corr()
            fig = px.imshow(corr_sensors, color_continuous_scale="RdBu", color_continuous_midpoint=0,
                            title="Sensor-to-Sensor Correlations", aspect="auto")
            fig.update_layout(paper_bgcolor="#1e1e2e", font_color="#cdd6f4")
            st.plotly_chart(fig, use_container_width=True)

        elif corr_type == "Full Feature Heatmap":
            sensor_cols_all = [c for c in df_full.columns if c.startswith("sensor_") and "_roll" not in c and "_lag" not in c]
            key_features = sensor_cols_all + ["degradation_index", "cycle_ratio", "rul"]
            corr_all = df_full[key_features].corr()
            fig = px.imshow(corr_all, color_continuous_scale="RdBu", color_continuous_midpoint=0,
                            title="Full Feature Correlation Matrix", aspect="auto")
            fig.update_layout(paper_bgcolor="#1e1e2e", font_color="#cdd6f4")
            st.plotly_chart(fig, use_container_width=True)

    # ─ TAB 4: Time Series Analysis ─
    with tab4:
        unit_ids = sorted(df_full[df_full["subset"].isin(subset_filter)]["unit"].unique().tolist())
        selected_unit = st.selectbox("Select Unit", unit_ids, key="ts_unit")
        unit_data = df_full[df_full["unit"] == selected_unit].sort_values("cycle")

        sensor_cols_all = [c for c in df_full.columns if c.startswith("sensor_") and "_roll" not in c and "_lag" not in c]
        selected_sensors = st.multiselect("Sensors to Plot (max 3)", sensor_cols_all, default=[sensor_cols_all[0], sensor_cols_all[1]])[:3]

        fig = go.Figure()
        for sensor in selected_sensors:
            fig.add_trace(go.Scatter(x=unit_data["cycle"], y=unit_data[sensor], mode="lines", name=sensor))

        anomaly_cycles = unit_data[unit_data["is_anomaly"]]["cycle"].tolist()
        if anomaly_cycles:
            fig.add_vline(x=anomaly_cycles[0], line_dash="dash", line_color="rgba(255, 0, 0, 0.3)", annotation_text="Anomalies")

        fig.update_layout(title=f"Unit {selected_unit} Sensor Time Series",
                          paper_bgcolor="#1e1e2e", plot_bgcolor="#1e1e2e", font_color="#cdd6f4",
                          hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=unit_data["cycle"], y=unit_data["rul"], mode="lines+markers",
                                  name="RUL", line=dict(color="#f38ba8", width=3)))
        fig2.add_hline(y=RUL_CRITICAL_THRESHOLD, line_dash="dash", line_color="#f38ba8", annotation_text="Critical")
        fig2.update_layout(title="Remaining Useful Life Trajectory",
                           paper_bgcolor="#1e1e2e", plot_bgcolor="#1e1e2e", font_color="#cdd6f4")
        st.plotly_chart(fig2, use_container_width=True)

    # ─ TAB 5: Feature Engineering Impact ─
    with tab5:
        st.markdown("### Rolling Statistics Impact")
        st.info("Rolling mean smooths noisy sensor readings; rolling std captures variability.")

        sensor_cols_all = [c for c in df_full.columns if c.startswith("sensor_") and "_roll" not in c and "_lag" not in c]
        demo_sensor = st.selectbox("Select Sensor for Demo", sensor_cols_all, key="demo_sensor")
        demo_unit = st.selectbox("Select Unit", sorted(df_full["unit"].unique()), key="demo_unit")

        unit_data_demo = df_full[df_full["unit"] == demo_unit].sort_values("cycle")
        rolling_5_col = f"{demo_sensor}_roll_mean_5"
        rolling_10_col = f"{demo_sensor}_roll_mean_10"

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=unit_data_demo["cycle"], y=unit_data_demo[demo_sensor],
                                 mode="lines", name="Raw Signal", line=dict(color="rgba(137, 180, 250, 0.3)", width=1)))
        if rolling_5_col in unit_data_demo.columns:
            fig.add_trace(go.Scatter(x=unit_data_demo["cycle"], y=unit_data_demo[rolling_5_col],
                                     mode="lines", name="Rolling Mean (5)", line=dict(color="#89b4fa", width=2)))
        if rolling_10_col in unit_data_demo.columns:
            fig.add_trace(go.Scatter(x=unit_data_demo["cycle"], y=unit_data_demo[rolling_10_col],
                                     mode="lines", name="Rolling Mean (10)", line=dict(color="#f38ba8", width=2)))
        fig.update_layout(title=f"Feature Engineering: Raw vs Smoothed {demo_sensor}",
                          paper_bgcolor="#1e1e2e", plot_bgcolor="#1e1e2e", font_color="#cdd6f4")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Engineered Features Summary")
        eng_info = pd.DataFrame({
            "Feature Type": ["Rolling Means", "Rolling Std", "Lag Features", "Degradation Index", "Cycle Ratio"],
            "Description": [
                "Smooths sensor noise over windows [5, 10, 20]",
                "Captures sensor variability",
                "Temporal dependencies at lags [1, 3, 5]",
                "Composite degradation from all sensors",
                "Lifecycle position (cycle / max_cycle)"
            ]
        })
        st.dataframe(eng_info, use_container_width=True, hide_index=True)

    # ─ TAB 6: Anomaly Insights ─
    with tab6:
        from analysis.anomaly_detection import get_anomaly_summary
        anomaly_summary = get_anomaly_summary(df_full)

        col1, col2, col3 = st.columns(3)
        col1.metric("Units with Anomalies", (anomaly_summary["anomaly_count"] > 0).sum())
        col2.metric("Total Anomalies Detected", anomaly_summary["anomaly_count"].sum())
        col3.metric("Avg Anomaly Rate", f"{anomaly_summary['anomaly_rate'].mean() * 100:.2f}%")

        st.markdown("### Anomaly Rate by Subset")
        subset_anomaly = anomaly_summary.groupby("subset")["anomaly_rate"].mean()
        fig = px.bar(x=subset_anomaly.index, y=subset_anomaly.values * 100,
                     labels={"x": "Subset", "y": "Anomaly Rate (%)"},
                     color=subset_anomaly.values, color_continuous_scale="Reds")
        fig.update_layout(paper_bgcolor="#1e1e2e", plot_bgcolor="#1e1e2e", font_color="#cdd6f4")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Anomaly Detection Early Warning")
        merged = anomaly_summary.merge(latest_f[["unit", "predicted_rul"]], on="unit", how="left")
        merged_valid = merged[merged["anomaly_count"] > 0]

        if len(merged_valid) > 0:
            fig = px.scatter(merged_valid, x="first_anomaly_cycle", y="predicted_rul",
                             hover_data=["unit", "anomaly_rate"], size="anomaly_count",
                             color="anomaly_rate", color_continuous_scale="Reds",
                             title="Early Anomalies Correlate with Lower RUL")
            fig.update_layout(paper_bgcolor="#1e1e2e", plot_bgcolor="#1e1e2e", font_color="#cdd6f4")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No anomalies detected in current dataset")

        st.markdown("### Top Anomalous Sensors")
        sensor_cols_all = [c for c in df_full.columns if c.startswith("sensor_") and "_roll" not in c and "_lag" not in c]
        sensor_anomaly_rate = []
        for sensor in sensor_cols_all:
            rate = df_full[df_full["is_anomaly"]][sensor].std() / df_full[sensor].std()
            sensor_anomaly_rate.append({"Sensor": sensor, "Anomaly Severity": rate})

        sensor_anomaly_df = pd.DataFrame(sensor_anomaly_rate).nlargest(10, "Anomaly Severity")
        fig = px.bar(sensor_anomaly_df, x="Sensor", y="Anomaly Severity",
                     color="Anomaly Severity", color_continuous_scale="Reds")
        fig.update_layout(paper_bgcolor="#1e1e2e", plot_bgcolor="#1e1e2e", font_color="#cdd6f4")
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: Unit Explorer
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Unit Explorer":
    st.title("🔍 Unit Explorer")
    unit_ids = sorted(latest_f["unit"].unique().tolist())
    selected_unit = st.selectbox("Select Unit", unit_ids)

    unit_data = df_full[df_full["unit"] == selected_unit].sort_values("cycle")
    unit_latest = latest_f[latest_f["unit"] == selected_unit].iloc[0]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Predicted RUL", f"{unit_latest['predicted_rul']:.0f} cycles")
    col2.metric("Status", unit_latest["status"])
    col3.metric("Current Cycle", int(unit_latest["cycle"]))
    anomaly_rate = unit_data["is_anomaly"].mean() * 100
    col4.metric("Anomaly Rate", f"{anomaly_rate:.1f}%")

    sensor_cols = [c for c in unit_data.columns if c.startswith("sensor_") and "_roll" not in c and "_lag" not in c]
    selected_sensor = st.selectbox("Sensor to plot", sensor_cols)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=unit_data["cycle"], y=unit_data[selected_sensor],
                             mode="lines", name=selected_sensor, line=dict(color="#89b4fa")))

    anomaly_cycles = unit_data[unit_data["is_anomaly"]]["cycle"]
    if not anomaly_cycles.empty:
        fig.add_trace(go.Scatter(x=anomaly_cycles, y=unit_data.loc[unit_data["is_anomaly"], selected_sensor],
                                 mode="markers", name="Anomalies", marker=dict(color="red", size=8, symbol="x")))

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
# PAGE 2.5: Anomaly Explorer
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Anomaly Explorer":
    st.title("🔍 Anomaly Explorer")

    from analysis.anomaly_detection import get_anomaly_summary
    anomaly_summary = get_anomaly_summary(df_full)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Anomaly Rate Heatmap")
        heatmap_data = anomaly_summary.pivot_table(values="anomaly_rate", index="unit", columns="subset", aggfunc="mean").fillna(0)
        fig = px.imshow(heatmap_data, aspect="auto", color_continuous_scale="Reds")
        fig.update_layout(paper_bgcolor="#1e1e2e", plot_bgcolor="#1e1e2e", font_color="#cdd6f4")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Top 10 Most Anomalous Units")
        top_anomalous = anomaly_summary.nlargest(10, "anomaly_rate")[["unit", "anomaly_rate", "first_anomaly_cycle"]]
        top_anomalous["anomaly_rate"] = (top_anomalous["anomaly_rate"] * 100).round(1).astype(str) + "%"
        st.dataframe(top_anomalous, use_container_width=True)

    st.subheader("First Anomaly Cycle vs Predicted RUL")
    merged = anomaly_summary.merge(latest_f[["unit", "predicted_rul"]], on="unit", how="left")
    fig = px.scatter(merged, x="first_anomaly_cycle", y="predicted_rul",
                     hover_data=["unit"], color="anomaly_rate",
                     color_continuous_scale="Reds")
    fig.update_layout(paper_bgcolor="#1e1e2e", plot_bgcolor="#1e1e2e", font_color="#cdd6f4")
    st.plotly_chart(fig, use_container_width=True)


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
                    history_parts = []
                    for m in st.session_state.messages[:-1]:
                        history_parts.append(m["role"].upper() + ": " + m["content"])
                    history = "\n".join(history_parts)
                    if history:
                        full_query = "Conversation so far:\n" + history + "\n\nUser: " + query
                    else:
                        full_query = query
                    response = run_agent(full_query)
                except Exception as e:
                    response = f"Agent error: {e}"
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})