"""
Anomaly Detection: IsolationForest-based outlier detection for sensor readings
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from config import MODELS_DIR


def detect_fleet_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Detect anomalies per unit using IsolationForest on sensor columns."""
    sensor_cols = [c for c in df.columns if c.startswith("sensor_") and not any(x in c for x in ["roll", "lag"])]
    anomaly_cols = ["anomaly_score", "is_anomaly", "anomaly_severity"]

    # Remove existing anomaly columns if present
    df = df.drop(columns=[c for c in anomaly_cols if c in df.columns], errors="ignore")

    # Fit per unit
    for unit in df["unit"].unique():
        unit_mask = df["unit"] == unit
        unit_data = df.loc[unit_mask, sensor_cols]

        if len(unit_data) < 10:  # Skip if too few samples
            df.loc[unit_mask, anomaly_cols] = [0.0, False, 0.0]
            continue

        # Fit IsolationForest
        iso_forest = IsolationForest(contamination=0.05, random_state=42, n_estimators=100)
        iso_forest.fit(unit_data)

        # Predict anomalies
        anomaly_scores = iso_forest.decision_function(unit_data)
        predictions = iso_forest.predict(unit_data)

        # Convert to desired format
        is_anomaly = predictions == -1  # -1 is anomaly
        anomaly_severity = (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min())
        anomaly_severity = 1 - anomaly_severity  # Invert so higher = more anomalous

        df.loc[unit_mask, "anomaly_score"] = anomaly_scores
        df.loc[unit_mask, "is_anomaly"] = is_anomaly
        df.loc[unit_mask, "anomaly_severity"] = anomaly_severity

    # Save model (using first unit's model as representative)
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "anomaly_model.pkl")
    joblib.dump(iso_forest, model_path)
    print(f"Anomaly model saved to {model_path}")

    return df


def get_anomaly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return per-unit anomaly statistics."""
    summary = []
    for unit in df["unit"].unique():
        unit_data = df[df["unit"] == unit]
        anomaly_count = unit_data["is_anomaly"].sum()
        total_cycles = len(unit_data)
        anomaly_rate = anomaly_count / total_cycles if total_cycles > 0 else 0
        first_anomaly_cycle = unit_data[unit_data["is_anomaly"]]["cycle"].min() if anomaly_count > 0 else None
        subset = unit_data["subset"].iloc[0] if not unit_data.empty else None

        summary.append({
            "unit": unit,
            "subset": subset,
            "anomaly_count": anomaly_count,
            "total_cycles": total_cycles,
            "anomaly_rate": anomaly_rate,
            "first_anomaly_cycle": first_anomaly_cycle
        })

    return pd.DataFrame(summary)