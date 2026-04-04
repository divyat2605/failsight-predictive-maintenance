import os
from dotenv import load_dotenv

load_dotenv()

# Paths
DATA_RAW_DIR = "data/raw"
DATA_PROCESSED_DIR = "data/processed"
MODELS_DIR = "models/saved"
REPORTS_DIR = "reports"
VECTORSTORE_DIR = "agent/vectorstore"

# CMAPSS config
CMAPSS_SUBSETS = ["FD001", "FD002", "FD003", "FD004"]
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
SETTING_COLS = ["setting_1", "setting_2", "setting_3"]
COL_NAMES = ["unit", "cycle"] + SETTING_COLS + SENSOR_COLS

# Feature engineering
WINDOW_SIZES = [5, 10, 20]
MAX_RUL = 125  # cap RUL for piecewise linear target

# Model
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Alert threshold
RUL_CRITICAL_THRESHOLD = 30
RUL_WARNING_THRESHOLD = 60

# LLM
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL = "gpt-3.5-turbo"

# DB
DB_PATH = "data/failsight.db"