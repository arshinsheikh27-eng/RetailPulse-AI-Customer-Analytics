
"""Demand-based inventory recommendations for RetailPulse.

The source dataset does not contain actual stock levels, supplier lead times,
or purchase costs. Recommendations are estimates based on historical sales.
"""

import pandas as pd

from src.data_loader import positive_sales


def inventory_recommendations(
    df: pd.DataFrame,
    lead_time_days: int = 14,
    safety_days: int = 7,
    top_n: int = 30,
) -> pd.DataFrame:
    """Calculate demand-based inventory planning estimates."""

    if not isinstance(lead_time_days, int) or isinstance(lead_time_days, bool):
        raise ValueError("lead_time_days must be an integer.")

    if not isinstance(safety_days, int) or isinstance(safety_days, bool):
        raise ValueError("safety_days must be an integer.")

    if not isinstance(top_n, int) or isinstance(top_n, bool):
        raise ValueError("top_n must be an integer.")

    if lead_time_days < 1:
        raise ValueError("lead_time_days must be at least 1.")

    if safety_days < 0:
        raise ValueError("safety_days cannot be negative.")

    if top_n < 1:
        raise ValueError("top_n must be at least 1.")

    required_columns = {"Quantity", "InvoiceDate"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Transaction data is missing required columns: {sorted(missing)}"
        )

    sales = positive_sales(df).copy()

    if sales.empty:
        raise ValueError("No positive sales records are available.")

    sales["InvoiceDate"] = pd.to_datetime(
        sales["InvoiceDate"], errors="coerce"
    )
    sales["Quantity"] = pd.to_numeric(
        sales["Quantity"], errors="coerce"
    )

    sales = sales.dropna(subset=["InvoiceDate", "Quantity"])
    sales = sales[sales["Quantity"] > 0].copy()

    if sales.empty:
        raise ValueError("No valid positive quantities or sales dates were found.")

    if "StockCode" not in sales.columns:
        sales["StockCode"] = "Unknown"

    if "Description" not in sales.columns:
        sales["Description"] = "Unknown product"

    sales["StockCode"] = sales["StockCode"].fillna("Unknown").astype(str)
    sales["Description"] = (
        sales["Description"].fillna("Unknown product").astype(str)
    )

    # Calculate line revenue consistently if the column is unavailable.
    if "LineRevenue" in sales.columns:
        sales["LineRevenue"] = pd.to_numeric(
            sales["LineRevenue"], errors="coerce"
        )
    elif "UnitPrice" in sales.columns:
        sales["LineRevenue"] = (
            pd.to_numeric(sales["UnitPrice"], errors="coerce")
            * sales["Quantity"]
        )
    else:
        sales["LineRevenue"] = 0.0

    sales["LineRevenue"] = sales["LineRevenue"].fillna(0.0)

    # Count distinct days on which each product was sold.
    by_product = (
        sales.groupby(["StockCode", "Description"], as_index=False)
        .agg(
            UnitsSold=("Quantity", "sum"),
            ActiveDays=("InvoiceDate", lambda dates: dates.dt.normalize().nunique()),
            SalesValue=("LineRevenue", "sum"),
        )
    )

    start_date = sales["InvoiceDate"].min().normalize()
    end_date = sales["InvoiceDate"].max().normalize()
    total_days = max(1, (end_date - start_date).days + 1)

    # Use the complete observed dataset period, including days with no sales.
    by_product["AverageDailyDemand"] = (
        by_product["UnitsSold"] / total_days
    )

    by_product["EstimatedLeadTimeDemand"] = (
        by_product["AverageDailyDemand"] * lead_time_days
    )

    by_product["SafetyStockUnits"] = (
        by_product["AverageDailyDemand"] * safety_days
    )

    by_product["ReorderPointUnits"] = (
        by_product["EstimatedLeadTimeDemand"]
        + by_product["SafetyStockUnits"]
    )

    # Round up to whole units for practical planning.
    by_product["EstimatedLeadTimeDemand"] = (
        by_product["EstimatedLeadTimeDemand"].apply(lambda value: int(-(-value // 1)))
    )
    by_product["SafetyStockUnits"] = (
        by_product["SafetyStockUnits"].apply(lambda value: int(-(-value // 1)))
    )
    by_product["ReorderPointUnits"] = (
        by_product["ReorderPointUnits"].apply(lambda value: int(-(-value // 1)))
    )

    by_product["SuggestedAction"] = "Review demand"

    if not by_product.empty:
        high_demand_cutoff = by_product["AverageDailyDemand"].quantile(0.75)

        by_product.loc[
            by_product["AverageDailyDemand"] >= high_demand_cutoff,
            "SuggestedAction",
        ] = "Prioritize replenishment"

    by_product = by_product.sort_values(
        ["UnitsSold", "SalesValue"],
        ascending=[False, False],
    )

    return by_product.head(top_n).reset_index(drop=True)
