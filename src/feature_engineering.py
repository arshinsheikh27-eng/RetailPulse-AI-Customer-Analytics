
from pathlib import Path
import json

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "cleaned_retail.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "customer_features.csv"
)

SUMMARY_PATH = (
    PROJECT_ROOT / "reports" / "customer_features_summary.json"
)


def main():
    # 1. Load the cleaned transaction-level dataset.
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {INPUT_PATH}"
        )

    df = pd.read_csv(
        INPUT_PATH,
        parse_dates=["InvoiceDate"]
    )

    required_columns = {
        "InvoiceNo",
        "StockCode",
        "Quantity",
        "InvoiceDate",
        "UnitPrice",
        "CustomerID",
        "IsCancellationInvoice",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    # 2. Keep records with an identifiable customer.
    identified = df.loc[df["CustomerID"].notna()].copy()

    # 3. Select eligible purchase lines for RFM.
    # Returns, cancellation invoices, and negative-price lines
    # are excluded from these purchase metrics.
    purchases = identified.loc[
        (identified["Quantity"] > 0)
        & (~identified["IsCancellationInvoice"].astype(bool))
        & (identified["UnitPrice"] >= 0)
    ].copy()

    if purchases.empty:
        raise ValueError("No eligible purchase rows were found.")

    purchases["LineSalesValue"] = (
        purchases["Quantity"] * purchases["UnitPrice"]
    )

    # 4. Calculate invoice-level metrics first.
    # Frequency counts distinct invoices, not transaction lines.
    invoices = (
        purchases.groupby(
            ["CustomerID", "InvoiceNo"],
            as_index=False
        )
        .agg(
            InvoiceValue=("LineSalesValue", "sum"),
            InvoiceDate=("InvoiceDate", "min"),
            InvoiceUnits=("Quantity", "sum"),
        )
    )

    # 5. Aggregate eligible purchases to one row per customer.
    features = (
        purchases.groupby("CustomerID")
        .agg(
            FirstPurchaseDate=("InvoiceDate", "min"),
            LastPurchaseDate=("InvoiceDate", "max"),
            UniqueProductsPurchased=("StockCode", "nunique"),
        )
        .reset_index()
    )

    invoice_features = (
        invoices.groupby("CustomerID")
        .agg(
            Frequency=("InvoiceNo", "nunique"),
            Monetary=("InvoiceValue", "sum"),
            TotalUnitsPurchased=("InvoiceUnits", "sum"),
            AverageOrderValue=("InvoiceValue", "mean"),
        )
        .reset_index()
    )

    features = features.merge(
        invoice_features,
        on="CustomerID",
        how="left",
        validate="one_to_one",
    )

    # 6. Define the reference date consistently.
    # One day after the final date in the cleaned dataset.
    reference_date = df["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)

    features["RecencyDays"] = (
        reference_date - features["LastPurchaseDate"].dt.normalize()
    ).dt.days

    features["CustomerTenureDays"] = (
        features["LastPurchaseDate"].dt.normalize()
        - features["FirstPurchaseDate"].dt.normalize()
    ).dt.days

    features["AverageItemsPerInvoice"] = (
        features["TotalUnitsPurchased"] / features["Frequency"]
    )

    # 7. Count return/negative-quantity lines separately.
    # These counts are descriptive and are not subtracted from RFM.
    return_counts = (
        identified.loc[
            (identified["Quantity"] < 0)
            | identified["IsCancellationInvoice"].astype(bool)
        ]
        .groupby("CustomerID")
        .size()
        .rename("ReturnOrCancellationLineCount")
        .reset_index()
    )

    features = features.merge(
        return_counts,
        on="CustomerID",
        how="left",
        validate="one_to_one",
    )

    features["ReturnOrCancellationLineCount"] = (
        features["ReturnOrCancellationLineCount"]
        .fillna(0)
        .astype("int64")
    )

    # 8. Validate the customer-level output.
    if features["CustomerID"].duplicated().any():
        raise ValueError("Duplicate CustomerID values found.")

    if features["RecencyDays"].lt(0).any():
        raise ValueError("Negative recency values found.")

    if features["Frequency"].lt(1).any():
        raise ValueError("A customer has no eligible purchase invoices.")

    if features["Monetary"].lt(0).any():
        raise ValueError("Negative monetary values found.")

    # 9. Save the output and a reproducibility summary.
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)

    features = features.sort_values("CustomerID").reset_index(drop=True)
    features.to_csv(OUTPUT_PATH, index=False)

    summary = {
        "source_file": "data/processed/cleaned_retail.csv",
        "customer_feature_file": "data/processed/customer_features.csv",
        "reference_date": reference_date.strftime("%Y-%m-%d"),
        "identified_customers_in_source": int(
            identified["CustomerID"].nunique()
        ),
        "customers_with_eligible_purchases": int(
            features["CustomerID"].nunique()
        ),
        "customer_feature_rows": int(len(features)),
        "customer_feature_columns": int(len(features.columns)),
        "feature_columns": features.columns.tolist(),
        "purchase_filter": (
            "CustomerID present, Quantity > 0, "
            "not a cancellation invoice, UnitPrice >= 0"
        ),
        "recency_definition": (
            "Days from last eligible purchase date to reference date"
        ),
        "frequency_definition": (
            "Count of distinct eligible purchase invoices per customer"
        ),
        "monetary_definition": (
            "Sum of Quantity * UnitPrice for eligible purchase lines"
        ),
        "limitations": [
            "RFM describes historical behavior and is not a churn prediction.",
            "Returns and cancellation lines are counted separately, "
            "not subtracted from RFM monetary value.",
            "The reference date is based on the dataset's final date.",
        ],
    }

    with open(SUMMARY_PATH, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print("\nRETAILPULSE CUSTOMER FEATURE SUMMARY")
    print("---------------------------------------------")
    print(f"Source rows: {len(df):,}")
    print(f"Identified customers in source: {identified['CustomerID'].nunique():,}")
    print(f"Customers with eligible purchases: {len(features):,}")
    print(f"Reference date: {reference_date.date()}")
    print(f"Feature columns: {len(features.columns)}")
    print(f"Customer features saved to: {OUTPUT_PATH}")
    print(f"Summary saved to: {SUMMARY_PATH}")

    print("\nRFM FEATURE PREVIEW")
    print(
        features[
            [
                "CustomerID",
                "RecencyDays",
                "Frequency",
                "Monetary",
                "AverageOrderValue",
                "UniqueProductsPurchased",
            ]
        ].head(5).to_string(index=False)
    )


if __name__ == "__main__":
    main()
