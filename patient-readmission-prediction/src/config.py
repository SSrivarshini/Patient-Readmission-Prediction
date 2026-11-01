from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RAW_CSV = ROOT / "data" / "raw" / "patients.csv"
DB_PATH = ROOT / "data" / "patients.db"
FEATURES_SQL = ROOT / "sql" / "features.sql"

MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
POWERBI_DIR = REPORTS_DIR / "powerbi"

TARGET = "readmitted_30d"
ID_COL = "patient_id"
RANDOM_STATE = 42
