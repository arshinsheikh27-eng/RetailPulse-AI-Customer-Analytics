
from pathlib import Path

import pandas as pd


# Locate the project directory and original dataset
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Online Retail.xlsx"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATA_PATH}"
        )

    # Read the original dataset without changing the Excel file
    df = pd.read_excel(DATA_PATH)

    print("\n" + "=" * 60)
    print("RETAILPULSE - RAW DATA QUALITY ASSESSMENT")
    print("=" * 60)

    print(f"\nDataset path: {DATA_PATH}")
    print(f"Total rows: {len(df):,}")
    print(f"Total columns: {len(df.columns)}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")

    print("\n1. COLUMN NAMES")
    print(list(df.columns))

    print("\n2. FIRST FIVE ROWS")
    print(df.head().to_string(index=False))

    print("\n3. DATA TYPES AND NON-NULL COUNTS")
    df.info()

    print("\n4. MISSING VALUES")
    missing = df.isnull().sum()
    missing_report = pd.DataFrame({
        "MissingCount": missing,
        "MissingPercent": (missing / len(df) * 100).round(2)
    })
    print(missing_report.to_string())

    print("\n5. DUPLICATE ROWS")
    print(f"Exact duplicate rows: {df.duplicated().sum():,}")

    print("\n6. NUMERIC COLUMN SUMMARY")
    print(df.describe().to_string())

    print("\n7. TRANSACTION DATE RANGE")
    dates = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    print(f"Earliest valid date: {dates.min()}")
    print(f"Latest valid date:   {dates.max()}")
    print(f"Invalid date values: {(dates.isna()).sum():,}")

    print("\n8. QUANTITY AND PRICE CHECKS")
    print(f"Negative quantity rows: {(df['Quantity'] < 0).sum():,}")
    print(f"Zero quantity rows: {(df['Quantity'] == 0).sum():,}")
    print(f"Negative price rows: {(df['UnitPrice'] < 0).sum():,}")
    print(f"Zero price rows: {(df['UnitPrice'] == 0).sum():,}")

    print("\n9. CANCELLATION INVOICES")
    invoice_text = df["InvoiceNo"].astype("string").str.strip()
    cancellation_mask = invoice_text.str.upper().str.startswith(
        "C", na=False
    )
    print(f"Rows with cancellation invoice numbers: {cancellation_mask.sum():,}")
    print(f"Distinct cancellation invoices: {invoice_text[cancellation_mask].nunique():,}")

    print("\n10. MISSING CUSTOMER IDs")
    print(f"Rows without CustomerID: {df['CustomerID'].isna().sum():,}")
    print(f"Distinct non-missing customers: {df['CustomerID'].nunique():,}")

    print("\n11. TOP COUNTRIES BY ROW COUNT")
    print(df["Country"].value_counts().head(10).to_string())

    print("\n" + "=" * 60)
    print("ASSESSMENT COMPLETE - ORIGINAL FILE WAS NOT MODIFIED")
    print("=" * 60)


if __name__ == "__main__":
    main()
