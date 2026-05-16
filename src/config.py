from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"
MODELS_DIR = PROJECT_ROOT / "models"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
LOGS_DIR = PROJECT_ROOT / "logs"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

RAW_DATA_FILE = RAW_DATA_DIR / "engine_data.csv"
CLEAN_DATA_FILE = PROCESSED_DATA_DIR / "engine_data_clean.csv"
TRAIN_DATA_FILE = PROCESSED_DATA_DIR / "train.csv"
TEST_DATA_FILE = PROCESSED_DATA_DIR / "test.csv"

TARGET_COLUMN = "engine_condition"
RANDOM_STATE = 42
TEST_SIZE = 0.2

FEATURE_COLUMNS = [
    "engine_rpm",
    "lub_oil_pressure",
    "fuel_pressure",
    "coolant_pressure",
    "lub_oil_temp",
    "coolant_temp",
]

DISPLAY_NAMES = {
    "engine_rpm": "Engine RPM",
    "lub_oil_pressure": "Lub Oil Pressure",
    "fuel_pressure": "Fuel Pressure",
    "coolant_pressure": "Coolant Pressure",
    "lub_oil_temp": "Lub Oil Temperature",
    "coolant_temp": "Coolant Temperature",
    "engine_condition": "Engine Condition",
}

REPORT_TITLE = "Predictive Maintenance for Engine Health"
PROJECT_NAME = "Predictive Maintenance"


def ensure_directories() -> None:
    for path in [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
        TABLES_DIR,
        MODELS_DIR,
        EXPERIMENTS_DIR,
        LOGS_DIR,
        NOTEBOOKS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
