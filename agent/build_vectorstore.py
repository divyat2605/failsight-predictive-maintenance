"""
Standalone script to rebuild the ChromaDB vectorstore.
Called by Airflow DAG after model retrain.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from config import DATA_PROCESSED_DIR
from agent.failsight_agent import build_vectorstore

if __name__ == "__main__":
    path = os.path.join(DATA_PROCESSED_DIR, "features.parquet")
    if not os.path.exists(path):
        print("Features not found. Run features.py first.")
        exit(1)
    df = pd.read_parquet(path)
    build_vectorstore(df)
    print("Vectorstore rebuilt successfully.")