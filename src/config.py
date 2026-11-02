from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
CLEANED_DATA = PROCESSED_DIR / "cleaned_retail.csv"
CUSTOMER_FEATURES = PROCESSED_DIR / "customer_features.csv"
MODEL_DIR = PROJECT_ROOT / "models"
REFERENCE_DATE = "2011-12-10"
