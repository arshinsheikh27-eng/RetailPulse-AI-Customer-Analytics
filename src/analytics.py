import pandas as pd
from src.data_loader import positive_sales

def sales_summary(df: pd.DataFrame) -> dict:
    sales = positive_sales(df)
    sales_value = sales["Quantity"] * sales["UnitPrice"]
    return {
        "positive_sales_value": float(sales_value.sum()),
        "positive_units": int(sales["Quantity"].sum()),
        "invoices": int(sales["InvoiceNo"].nunique()),
        "customers": int(sales["CustomerID"].nunique()) if "CustomerID" in sales else 0,
        "average_invoice_value": float(sales.assign(_value=sales_value).groupby("InvoiceNo")["_value"].sum().mean()),
        "transaction_rows": int(len(df)),
    }

def monthly_sales(df: pd.DataFrame) -> pd.DataFrame:
    sales = positive_sales(df)
    sales["SalesValue"] = sales["Quantity"] * sales["UnitPrice"]
    sales["Month"] = sales["InvoiceDate"].dt.to_period("M").dt.to_timestamp()
    return sales.groupby("Month", as_index=False)["SalesValue"].sum().sort_values("Month")

def top_products(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    sales = positive_sales(df)
    sales["SalesValue"] = sales["Quantity"] * sales["UnitPrice"]
    return (sales.groupby("Description", dropna=False)["SalesValue"].sum()
            .nlargest(n).reset_index().sort_values("SalesValue"))

def top_countries(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    sales = positive_sales(df)
    sales["SalesValue"] = sales["Quantity"] * sales["UnitPrice"]
    return (sales.groupby("Country")["SalesValue"].sum()
            .nlargest(n).reset_index().sort_values("SalesValue"))

def weekday_sales(df: pd.DataFrame) -> pd.DataFrame:
    sales = positive_sales(df)
    sales["SalesValue"] = sales["Quantity"] * sales["UnitPrice"]
    return sales.groupby("DayOfWeek", as_index=False)["SalesValue"].sum()
