"""
Spare Parts Demand Forecasting
Derives replacement demand from RUL predictions across the fleet.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from config import DATA_PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, RUL_CRITICAL_THRESHOLD
from models.train_rul import predict_rul


def forecast_demand(df: pd.DataFrame, horizon_cycles: int = 50) -> pd.DataFrame:
    """
    For each unit, predict RUL on latest cycle reading.
    Units with RUL <= horizon_cycles will need replacement within the window.
    Returns a demand forecast dataframe.
    """
    # Get latest cycle per unit
    latest = df.sort_values("cycle").groupby("unit").last().reset_index()

    # Predict RUL
    preds = predict_rul(latest)
    latest["predicted_rul"] = preds
    latest["predicted_rul"] = latest["predicted_rul"].clip(lower=0)

    # Classify urgency
    def classify(rul):
        if rul <= RUL_CRITICAL_THRESHOLD:
            return "CRITICAL"
        elif rul <= 60:
            return "WARNING"
        else:
            return "HEALTHY"

    latest["status"] = latest["predicted_rul"].apply(classify)

    # Demand within horizon
    latest["needs_replacement"] = latest["predicted_rul"] <= horizon_cycles

    # Aggregate demand per cycle window
    demand_df = latest[latest["needs_replacement"]].copy()
    demand_df["replacement_cycle"] = (
        latest["cycle"] + latest["predicted_rul"]
    ).astype(int)

    summary = {
        "total_units": len(latest),
        "critical_units": (latest["status"] == "CRITICAL").sum(),
        "warning_units": (latest["status"] == "WARNING").sum(),
        "healthy_units": (latest["status"] == "HEALTHY").sum(),
        "replacements_within_horizon": demand_df["needs_replacement"].sum(),
        "horizon_cycles": horizon_cycles
    }

    return latest[["unit", "cycle", "predicted_rul", "status", "needs_replacement"]], summary


def weekly_demand_curve(demand_df: pd.DataFrame, cycles_per_week: int = 7) -> pd.DataFrame:
    """Bin replacements into weekly buckets."""
    demand_df = demand_df[demand_df["needs_replacement"]].copy()
    demand_df["week"] = (demand_df["predicted_rul"] // cycles_per_week).astype(int)
    weekly = demand_df.groupby("week").size().reset_index(name="parts_needed")
    weekly["week_label"] = weekly["week"].apply(lambda w: f"Week {w+1}")
    return weekly


if __name__ == "__main__":
    import joblib
    path = os.path.join(DATA_PROCESSED_DIR, "features.parquet")
    df = pd.read_parquet(path)
    demand_df, summary = forecast_demand(df)
    print("\nDemand Forecast Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    weekly = weekly_demand_curve(demand_df)
    print("\nWeekly Demand Curve:")
    print(weekly)