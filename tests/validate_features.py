
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURES_PATH = (
    PROJECT_ROOT / "data" / "processed" / "customer_features.csv"
)

CLEANED_PATH = (
    PROJECT_ROOT / "data" / "processed" / "cleaned_retail.csv"
)


def main():
    print("RETAILPULSE CUSTOMER FEATURE VALIDATION")
    print("-" * 45)

    features = pd.read_csv(FEATURES_PATH)
    cleaned = pd.read_csv(
        CLEANED_PATH,
        parse_dates=["InvoiceDate"],
    )

    required = [
        "CustomerID",
        "FirstPurchaseDate",
        "LastPurchaseDate",
        "UniqueProductsPurchased",
        "Frequency",
        "Monetary",
        "TotalUnitsPurchased",
        "AverageOrderValue",
        "RecencyDays",
        "CustomerTenureDays",
        "AverageItemsPerInvoice",
        "ReturnOrCancellationLineCount",
    ]

    # Check required columns.
    assert set(required).issubset(features.columns), (
        "Some required feature columns are missing."
    )
    print("[PASS] All required feature columns exist.")

    # Check customer identity and row uniqueness.
    assert features["CustomerID"].notna().all(), (
        "Missing CustomerID values found."
    )
    assert features["CustomerID"].is_unique, (
        "Duplicate customer records found."
    )
    print("[PASS] Each customer has exactly one feature row.")

    # Check that all eligible customers are included.
    cleaned["CustomerID"] = pd.to_numeric(
        cleaned["CustomerID"], errors="coerce"
    )

    eligible_purchases = cleaned.loc[
        cleaned["CustomerID"].notna()
        & (cleaned["Quantity"] > 0)
        & (~cleaned["IsCancellationInvoice"].astype(bool))
        & (cleaned["UnitPrice"] >= 0)
    ]

    expected_customers = set(
        eligible_purchases["CustomerID"].unique()
    )

    actual_customers = set(features["CustomerID"].unique())

    assert actual_customers == expected_customers, (
        "Customer coverage mismatch: "
        f"{len(expected_customers - actual_customers)} eligible customers missing, "
        f"{len(actual_customers - expected_customers)} unexpected customers found."
    )

    print("[PASS] All eligible customers are included in the feature table.")

    # Check required numerical values.
    numeric_columns = [
        "Frequency",
        "Monetary",
        "AverageOrderValue",
        "RecencyDays",
        "CustomerTenureDays",
        "UniqueProductsPurchased",
        "TotalUnitsPurchased",
        "AverageItemsPerInvoice",
        "ReturnOrCancellationLineCount",
    ]

    assert features[numeric_columns].notna().all().all(), (
        "Missing values found in required numerical features."
    )
    assert np.isfinite(
        features[numeric_columns].to_numpy(dtype=float)
    ).all(), "Infinite numerical values found."
    print("[PASS] Numerical features contain no missing or infinite values.")

    # Check valid ranges.
    assert features["Frequency"].ge(1).all()
    assert features["Monetary"].ge(0).all()
    assert features["RecencyDays"].ge(0).all()
    assert features["CustomerTenureDays"].ge(0).all()
    assert features["UniqueProductsPurchased"].ge(1).all()
    assert features["TotalUnitsPurchased"].gt(0).all()
    assert features["ReturnOrCancellationLineCount"].ge(0).all()
    print("[PASS] Feature values satisfy basic range checks.")

    # Check average order value calculation.
    expected_aov = (
        features["Monetary"] / features["Frequency"]
    )

    assert np.allclose(
        features["AverageOrderValue"],
        expected_aov,
        rtol=1e-6,
        atol=1e-6,
    ), "AverageOrderValue does not match Monetary / Frequency."
    print("[PASS] Average order value calculations are consistent.")

    # Check average units per invoice.
    expected_items = (
        features["TotalUnitsPurchased"] / features["Frequency"]
    )

    assert np.allclose(
        features["AverageItemsPerInvoice"],
        expected_items,
        rtol=1e-6,
        atol=1e-6,
    ), "AverageItemsPerInvoice calculation is inconsistent."
    print("[PASS] Average items per invoice calculations are consistent.")

    # Check the reference date and recency.
    cleaned["CustomerID"] = pd.to_numeric(
        cleaned["CustomerID"], errors="coerce"
    )

    reference_date = (
        cleaned["InvoiceDate"].max().normalize()
        + pd.Timedelta(days=1)
    )

    last_purchase = pd.to_datetime(features["LastPurchaseDate"])
    expected_recency = (
        reference_date - last_purchase.dt.normalize()
    ).dt.days

    assert np.array_equal(
        features["RecencyDays"].to_numpy(),
        expected_recency.to_numpy(),
    ), "Recency values do not match the documented reference date."
    print("[PASS] Recency matches the dataset reference date.")

    # Confirm that the feature table has one row per eligible customer.
    assert len(features) == features["CustomerID"].nunique()
    print(f"\nValidated customers: {len(features):,}")
    print(f"Feature columns: {len(features.columns)}")
    print("RESULT: All validation checks passed.")


if __name__ == "__main__":
    main()
