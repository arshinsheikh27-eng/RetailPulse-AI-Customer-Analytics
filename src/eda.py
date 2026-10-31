
from pathlib import Path
import json

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_retail.csv"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
SUMMARY_PATH = PROJECT_ROOT / "reports" / "eda_summary.json"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {DATA_PATH}. "
            "Run src/clean_data.py first."
        )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH, parse_dates=["InvoiceDate"])

    required = {
        "InvoiceNo", "StockCode", "Description", "Quantity",
        "InvoiceDate", "UnitPrice", "CustomerID", "Country",
        "IsCancellationInvoice", "LineRevenue"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Define an explicit population for positive-sales analysis.
    sales = df.loc[
        (df["Quantity"] > 0)
        & (~df["IsCancellationInvoice"].astype(bool))
        & (df["UnitPrice"] >= 0)
    ].copy()

    sales["SalesValue"] = sales["Quantity"] * sales["UnitPrice"]
    sales["InvoiceYearMonth"] = (
        sales["InvoiceDate"].dt.to_period("M").astype(str)
    )

    if sales.empty:
        raise ValueError("No records meet the positive-sales criteria.")

    # Invoice-level metrics: one record per invoice.
    invoice_totals = (
        sales.groupby("InvoiceNo", as_index=False)
        .agg(
            InvoiceValue=("SalesValue", "sum"),
            InvoiceDate=("InvoiceDate", "min"),
            CustomerID=("CustomerID", "first"),
        )
    )

    # Monthly sales trend.
    monthly = (
        sales.groupby("InvoiceYearMonth", as_index=False)
        .agg(
            SalesValue=("SalesValue", "sum"),
            UnitsSold=("Quantity", "sum"),
            InvoiceLines=("InvoiceNo", "size"),
        )
    )

    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=monthly, x="InvoiceYearMonth", y="SalesValue", marker="o"
    )
    plt.title("RetailPulse: Monthly Positive-Sales Value")
    plt.xlabel("Year-Month")
    plt.ylabel("Sales value (source currency)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "monthly_sales_trend.png", dpi=160)
    plt.close()

    # Top products by positive-sales value.
    product_sales = (
        sales.groupby("Description", dropna=False, as_index=False)
        .agg(
            SalesValue=("SalesValue", "sum"),
            UnitsSold=("Quantity", "sum"),
        )
        .sort_values("SalesValue", ascending=False)
        .head(10)
    )

    product_plot = product_sales.copy()
    product_plot["Description"] = (
        product_plot["Description"].fillna("(Missing description)")
    )

    plt.figure(figsize=(12, 7))
    sns.barplot(
        data=product_plot.sort_values("SalesValue"),
        x="SalesValue",
        y="Description",
    )
    plt.title("Top 10 Products by Positive-Sales Value")
    plt.xlabel("Sales value (source currency)")
    plt.ylabel("Product description")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "top_products_by_sales.png", dpi=160)
    plt.close()

    # Top countries by positive-sales value.
    country_sales = (
        sales.groupby("Country", as_index=False)
        .agg(
            SalesValue=("SalesValue", "sum"),
            UnitsSold=("Quantity", "sum"),
            InvoiceCount=("InvoiceNo", "nunique"),
        )
        .sort_values("SalesValue", ascending=False)
        .head(10)
    )

    plt.figure(figsize=(11, 6))
    sns.barplot(
        data=country_sales.sort_values("SalesValue"),
        x="SalesValue",
        y="Country",
    )
    plt.title("Top 10 Countries by Positive-Sales Value")
    plt.xlabel("Sales value (source currency)")
    plt.ylabel("Country")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "top_countries_by_sales.png", dpi=160)
    plt.close()

    # Quantity distribution, limited to the 99th percentile for readability.
    quantity_limit = sales["Quantity"].quantile(0.99)
    quantity_for_plot = sales.loc[
        sales["Quantity"] <= quantity_limit, "Quantity"
    ]

    plt.figure(figsize=(10, 6))
    sns.histplot(quantity_for_plot, bins=50)
    plt.title("Distribution of Positive Quantities (Up to 99th Percentile)")
    plt.xlabel("Quantity per invoice line")
    plt.ylabel("Number of invoice lines")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "quantity_distribution.png", dpi=160)
    plt.close()

    # Key metrics, with definitions reflected in their names.
    summary = {
        "source_file": str(DATA_PATH.relative_to(PROJECT_ROOT)),
        "source_rows": int(len(df)),
        "positive_sales_rows": int(len(sales)),
        "positive_sales_value": round(float(sales["SalesValue"].sum()), 2),
        "positive_units_sold": int(sales["Quantity"].sum()),
        "distinct_invoices_in_positive_sales": int(
            sales["InvoiceNo"].nunique()
        ),
        "distinct_identifiable_customers_in_positive_sales": int(
            sales["CustomerID"].nunique()
        ),
        "average_positive_sales_invoice_value": round(
            float(invoice_totals["InvoiceValue"].mean()), 2
        ),
        "monthly_periods": int(monthly["InvoiceYearMonth"].nunique()),
        "sales_definition": (
            "Rows with Quantity > 0, non-cancellation invoice numbers, "
            "and UnitPrice >= 0. SalesValue = Quantity * UnitPrice. "
            "This is an analytical positive-sales measure, not audited "
            "accounting revenue."
        ),
        "charts": [
            "monthly_sales_trend.png",
            "top_products_by_sales.png",
            "top_countries_by_sales.png",
            "quantity_distribution.png",
        ],
    }

    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=4),
        encoding="utf-8",
    )

    print("\nRETAILPULSE EDA SUMMARY")
    print("-" * 45)
    print(f"Cleaned source rows: {len(df):,}")
    print(f"Positive-sales rows: {len(sales):,}")
    print(f"Positive-sales value: {summary['positive_sales_value']:,.2f}")
    print(f"Positive units sold: {summary['positive_units_sold']:,}")
    print(
        "Distinct invoices: "
        f"{summary['distinct_invoices_in_positive_sales']:,}"
    )
    print(
        "Identifiable customers: "
        f"{summary['distinct_identifiable_customers_in_positive_sales']:,}"
    )
    print("\nTop 10 countries by positive-sales value:")
    print(country_sales.to_string(index=False))
    print(f"\nCharts saved in: {FIGURES_DIR}")
    print(f"Summary saved to: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
