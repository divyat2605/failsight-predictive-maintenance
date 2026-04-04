"""
Pipeline 2: Feature engineering — rolling stats, lag features, degradation index
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from config import DATA_PROCESSED_DIR, WINDOW_SIZES


def get_sensor_cols(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c.startswith("sensor_")]


def add_rolling_features(df: pd.DataFrame, sensor_cols: list) -> pd.DataFrame:
    """Add rolling mean and std for each sensor per unit."""
    df = df.sort_values(["unit", "cycle"])
    for window in WINDOW_SIZES:
        for col in sensor_cols:
            grp = df.groupby("unit")[col]
            df[f"{col}_roll_mean_{window}"] = grp.transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            df[f"{col}_roll_std_{window}"] = grp.transform(
                lambda x: x.rolling(window, min_periods=1).std().fillna(0)
            )
    return df


def add_lag_features(df: pd.DataFrame, sensor_cols: list, lags: list = [1, 3, 5]) -> pd.DataFrame:
    """Add lag features per unit."""
    df = df.sort_values(["unit", "cycle"])
    for lag in lags:
        for col in sensor_cols:
            df[f"{col}_lag_{lag}"] = df.groupby("unit")[col].shift(lag).fillna(0)
    return df


def add_degradation_index(df: pd.DataFrame, sensor_cols: list) -> pd.DataFrame:
    """
    Composite degradation index: normalized sum of sensor z-scores.
    Higher = more degraded.
    """
    z_scores = df[sensor_cols].apply(lambda x: (x - x.mean()) / (x.std() + 1e-8))
    df["degradation_index"] = z_scores.mean(axis=1)
    return df


def add_cycle_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cycle ratio (how far through its life the unit is)."""
    max_cycles = df.groupby("unit")["cycle"].transform("max")
    df["cycle_ratio"] = df["cycle"] / max_cycles
    return df


def run_feature_engineering():
    input_path = os.path.join(DATA_PROCESSED_DIR, "all_train.parquet")
    if not os.path.exists(input_path):
        print(f"Processed data not found at {input_path}. Run ingest.py first.")
        return

    print("Loading processed data...")
    df = pd.read_parquet(input_path)
    sensor_cols = get_sensor_cols(df)

    print("Adding rolling features...")
    df = add_rolling_features(df, sensor_cols)

    print("Adding lag features...")
    df = add_lag_features(df, sensor_cols)

    print("Adding degradation index...")
    df = add_degradation_index(df, sensor_cols)

    print("Adding cycle ratio...")
    df = add_cycle_features(df)

    print("Detecting anomalies...")
    from analysis.anomaly_detection import detect_fleet_anomalies
    df = detect_fleet_anomalies(df)

    out_path = os.path.join(DATA_PROCESSED_DIR, "features.parquet")
    df.to_parquet(out_path, index=False)
    print(f"Feature engineering complete. Shape: {df.shape}")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    run_feature_engineering()