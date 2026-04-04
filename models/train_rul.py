"""
Model: LightGBM RUL prediction
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import lightgbm as lgb
from config import DATA_PROCESSED_DIR, MODELS_DIR, RANDOM_STATE, TEST_SIZE

EXCLUDE_COLS = ["unit", "cycle", "subset", "split", "rul"]


def get_feature_cols(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in EXCLUDE_COLS]


def train():
    os.makedirs(MODELS_DIR, exist_ok=True)

    path = os.path.join(DATA_PROCESSED_DIR, "features.parquet")
    if not os.path.exists(path):
        print("Features not found. Run features.py first.")
        return

    df = pd.read_parquet(path)
    feature_cols = get_feature_cols(df)

    X = df[feature_cols].fillna(0)
    y = df["rul"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    print(f"Training on {len(X_train)} samples, validating on {len(X_val)}")

    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=7,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
    )

    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    mae = mean_absolute_error(y_val, preds)
    print(f"\nValidation RMSE: {rmse:.2f} | MAE: {mae:.2f}")

    model_path = os.path.join(MODELS_DIR, "rul_model.pkl")
    feature_path = os.path.join(MODELS_DIR, "feature_cols.pkl")
    joblib.dump(model, model_path)
    joblib.dump(feature_cols, feature_path)
    print(f"Model saved to {model_path}")

    return model, feature_cols


def predict_rul(df: pd.DataFrame) -> np.ndarray:
    """Run inference on a dataframe."""
    model_path = os.path.join(MODELS_DIR, "rul_model.pkl")
    feature_path = os.path.join(MODELS_DIR, "feature_cols.pkl")

    model = joblib.load(model_path)
    feature_cols = joblib.load(feature_path)

    X = df[feature_cols].fillna(0)
    return model.predict(X)


if __name__ == "__main__":
    train()