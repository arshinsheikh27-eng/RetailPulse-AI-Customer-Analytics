
from pathlib import Path
import json

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "Online Retail.xlsx"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_PATH = PROCESSED_DIR / "cleaned_retail.csv"
SUMMARY_PATH = PROCESSED_DIR / "cleaning_summary.json"


def main():
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Raw dataset not found: {RAW_PATH}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load the original source data.
    df = pd.read_excel(RAW_PATH)
    original_rows = len(df)

    required_columns = [
        "InvoiceNo", "StockCode", "Description", "Quantity",
        "InvoiceDate", "UnitPrice", "CustomerID", "Country"
    ]

    missing_columns = sorted(set(required_columns) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Required columns are missing: {missing_columns}")

    # 2. Standardize text without changing business meaning.
    for column in ["InvoiceNo", "StockCode", "Description", "Country"]:
        df[column] = df[column].astype("string").str.strip()

    # Treat blank text fields as missing values.
    for column in ["InvoiceNo", "StockCode", "Description", "Country"]:
        df[column] = df[column].replace("", pd.NA)

    # 3. Record exact duplicates before removing them.
    duplicate_mask = df.duplicated(keep="first")
    exact_duplicate_rows = int(duplicate_mask.sum())
    df = df.drop_duplicates(keep="first").copy()

    # 4. Standardize and validate data types.
    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"], errors="coerce"
    )
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")
    df["CustomerID"] = pd.to_numeric(
        df["CustomerID"], errors="coerce"
    )

    # Customer IDs are identifiers, not continuous measurements.
    # Validate that every non-missing ID is a whole number.
    customer_id_values = df["CustomerID"].dropna()

    invalid_customer_ids = (
        customer_id_values % 1 != 0
    )

    if invalid_customer_ids.any():
        raise ValueError(
            "CustomerID contains non-integer values. "
            "Review these identifiers before cleaning."
        )

    # Nullable integer preserves missing IDs without converting them to 0.
    df["CustomerID"] = df["CustomerID"].astype("Int64")


    # Fail clearly if essential transaction fields cannot be interpreted.
    invalid_required = df[
        ["InvoiceNo", "StockCode", "InvoiceDate", "Quantity",
         "UnitPrice", "Country"]
    ].isna().any(axis=1)

    invalid_required_rows = int(invalid_required.sum())

    if invalid_required_rows:
        raise ValueError(
            f"{invalid_required_rows} rows have missing or invalid "
            "essential transaction fields. Review these rows before "
            "deciding how to handle them."
        )

    # 5. Add transparent quality and transaction flags.
    df["IsCancellationInvoice"] = (
        df["InvoiceNo"].str.upper().str.startswith("C", na=False)
    )
    df["IsReturnOrNegativeQuantity"] = df["Quantity"] < 0
    df["IsZeroPrice"] = df["UnitPrice"] == 0
    df["IsNegativePrice"] = df["UnitPrice"] < 0
    df["HasMissingCustomerID"] = df["CustomerID"].isna()
    df["HasMissingDescription"] = df["Description"].isna()

    # Signed line value, not a claim of recognized accounting revenue.
    df["LineRevenue"] = df["Quantity"] * df["UnitPrice"]

    # Convenient time features; InvoiceDate remains the source timestamp.
    df["InvoiceYear"] = df["InvoiceDate"].dt.year.astype("int16")
    df["InvoiceMonth"] = df["InvoiceDate"].dt.month.astype("int8")
    df["InvoiceYearMonth"] = df["InvoiceDate"].dt.to_period("M").astype(str)
    df["DayOfWeek"] = df["InvoiceDate"].dt.day_name()
    df["InvoiceHour"] = df["InvoiceDate"].dt.hour.astype("int8")
    df["IsWeekend"] = df["InvoiceDate"].dt.dayofweek >= 5

    # 6. Validate the cleaned dataset.
    if df.empty:
        raise ValueError("The cleaned dataset is empty.")

    if df.duplicated().any():
        raise ValueError("Exact duplicate rows remain after cleaning.")

    if df["InvoiceDate"].isna().any():
        raise ValueError("Invalid invoice dates remain.")

    if df["LineRevenue"].isna().any():
        raise ValueError("LineRevenue contains missing values.")

    # 7. Save the processed data.
    df.to_csv(OUTPUT_PATH, index=False)

    # 8. Save a reproducible cleaning summary.
    summary = {
        "source_file": str(RAW_PATH.relative_to(PROJECT_ROOT)),
        "output_file": str(OUTPUT_PATH.relative_to(PROJECT_ROOT)),
        "original_row_count": original_rows,
        "exact_duplicate_rows_removed": exact_duplicate_rows,
        "rows_after_exact_deduplication": len(df),
        "rows_removed_for_invalid_required_fields": 0,
        "rows_with_missing_customer_id_preserved": int(
            df["CustomerID"].isna().sum()
        ),
        "rows_with_missing_description_preserved": int(
            df["Description"].isna().sum()
        ),
        "cancellation_invoice_rows_preserved": int(
            df["IsCancellationInvoice"].sum()
        ),
        "negative_quantity_rows_preserved": int(
            df["IsReturnOrNegativeQuantity"].sum()
        ),
        "zero_price_rows_preserved": int(df["IsZeroPrice"].sum()),
        "negative_price_rows_preserved": int(df["IsNegativePrice"].sum()),
        "final_row_count": len(df),
        "final_column_count": len(df.columns),
        "cleaning_policy": (
            "Remove exact duplicate rows only. Preserve cancellations, "
            "negative quantities, unusual prices, missing customer IDs, "
            "and missing descriptions. Add flags for downstream analysis."
        ),
    }

    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=4),
        encoding="utf-8"
    )

    print("\nRETAILPULSE CLEANING PIPELINE")
    print("-" * 45)
    print(f"Original rows:                 {original_rows:,}")
    print(f"Exact duplicates removed:      {exact_duplicate_rows:,}")
    print(f"Rows after deduplication:      {len(df):,}")
    print(f"Output columns:                {len(df.columns)}")
    print(f"Missing customer IDs kept:     {df['CustomerID'].isna().sum():,}")
    print(f"Missing descriptions kept:     {df['Description'].isna().sum():,}")
    print(f"Cancellation rows preserved:   {df['IsCancellationInvoice'].sum():,}")
    print(f"Negative quantity rows kept:   {df['IsReturnOrNegativeQuantity'].sum():,}")
    print(f"Output CSV: {OUTPUT_PATH}")
    print(f"Summary JSON: {SUMMARY_PATH}")
    print("\nCleaning completed and validation passed.")


if __name__ == "__main__":
    main()
