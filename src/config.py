
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

# Use the local CSV when available.
# Otherwise, use the compressed dataset for cloud deployment.
CLEANED_CSV = PROCESSED_DIR / "cleaned_retail.csv"
CLEANED_ZIP = PROCESSED_DIR / "cleaned_retail.zip"

CLEANED_DATA = (
    CLEANED_CSV if CLEANED_CSV.exists() else CLEANED_ZIP
)

CUSTOMER_FEATURES = PROCESSED_DIR / "customer_features.csv"

MODEL_DIR = PROJECT_ROOT / "models"

REFERENCE_DATE = "2011-12-10"
