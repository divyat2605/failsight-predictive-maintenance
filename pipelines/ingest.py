"""
Pipeline 1: Raw CMAPSS data ingestion + cleaning → SQLite
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import sqlite3
import requests
import zipfile
import io
from config import (
    DATA_RAW_DIR, DATA_PROCESSED_DIR, COL_NAMES,
    SENSOR_COLS, SETTING_COLS, DB_PATH, CMAPSS_SUBSETS
)


def download_cmapss():
    """Download NASA CMAPSS dataset if not present."""
    os.makedirs(DATA_RAW_DIR, exist_ok=True)
    url = "https://ti.arc.nasa.gov/c/6/"
    local_zip = os.path.join(DATA_RAW_DIR, "CMAPSSData.zip")

    if not os.path.exists(local_zip):
        print("Downloading CMAPSS dataset...")
        # Note: NASA URL requires manual download in some environments
        # Place CMAPSSData.zip manually in data/raw/ if needed
        print(f"Please download from: {url}")
        print(f"And place CMAPSSData.zip in: {DATA_RAW_DIR}")
    else:
        print("CMAPSS zip found, extracting...")
        with zipfile.ZipFile(local_zip, "r") as z:
            z.extractall(DATA_RAW_DIR)
        print("Extraction complete.")


def load_subset(subset: str, split: str = "train") -> pd.DataFrame:
    """Load a CMAPSS subset (train/test)."""
    filename = f"{split}_{subset}.txt"
    filepath = os.path.join(DATA_RAW_DIR, filename)

    df = pd.read_csv(filepath, sep=r"\s+", header=None, names=COL_NAMES)
    df["subset"] = subset
    df["split"] = split
    return df


def add_rul(df: pd.DataFrame, max_rul: int = 125) -> pd.DataFrame:
    """Add piecewise linear RUL target column."""
    max_cycles = df.groupby("unit")["cycle"].max().reset_index()
    max_cycles.columns = ["unit", "max_cycle"]
    df = df.merge(max_cycles, on="unit")
    df["rul"] = df["max_cycle"] - df["cycle"]
    df["rul"] = df["rul"].clip(upper=max_rul)
    df.drop(columns=["max_cycle"], inplace=True)
    return df


def drop_constant_sensors(df: pd.DataFrame) -> pd.DataFrame:
    """Drop sensors with near-zero variance (not informative)."""
    # Sensors known to be constant in CMAPSS: 1, 5, 6, 10, 16, 18, 19
    drop_sensors = ["sensor_1", "sensor_5", "sensor_6",
                    "sensor_10", "sensor_16", "sensor_18", "sensor_19"]
    drop_cols = [c for c in drop_sensors if c in df.columns]
    return df.drop(columns=drop_cols)


def normalize_sensors(df: pd.DataFrame, sensor_cols: list) -> pd.DataFrame:
    """Min-max normalize sensor readings per subset."""
    for col in sensor_cols:
        col_min = df[col].min()
        col_max = df[col].max()
        if col_max - col_min > 0:
            df[col] = (df[col] - col_min) / (col_max - col_min)
    return df


def save_to_db(df: pd.DataFrame, table: str):
    """Save processed dataframe to SQLite."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    df.to_sql(table, conn, if_exists="replace", index=False)
    conn.close()
    print(f"Saved {len(df)} rows to table '{table}' in {DB_PATH}")


def run_ingestion():
    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
    all_train = []

    for subset in CMAPSS_SUBSETS:
        try:
            train_df = load_subset(subset, "train")
            train_df = add_rul(train_df)
            train_df = drop_constant_sensors(train_df)

            remaining_sensors = [c for c in train_df.columns if c.startswith("sensor_")]
            train_df = normalize_sensors(train_df, remaining_sensors)
            all_train.append(train_df)

            out_path = os.path.join(DATA_PROCESSED_DIR, f"{subset}_train.parquet")
            train_df.to_parquet(out_path, index=False)
            print(f"[{subset}] Processed {len(train_df)} rows")

        except FileNotFoundError:
            print(f"[{subset}] File not found — skipping. Ensure CMAPSSData is in {DATA_RAW_DIR}")

    if all_train:
        combined = pd.concat(all_train, ignore_index=True)
        save_to_db(combined, "sensor_readings")
        combined.to_parquet(os.path.join(DATA_PROCESSED_DIR, "all_train.parquet"), index=False)
        print(f"\nIngestion complete. Total rows: {len(combined)}")
    else:
        print("No data processed. Check dataset path.")


if __name__ == "__main__":
    run_ingestion()