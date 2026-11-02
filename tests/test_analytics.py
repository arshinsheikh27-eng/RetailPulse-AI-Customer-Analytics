import pandas as pd
from src.analytics import sales_summary, monthly_sales
from src.data_loader import positive_sales

def sample_data():
    return pd.DataFrame({
        "InvoiceNo": ["1", "1", "C2"],
        "Quantity": [2, 1, -1],
        "UnitPrice": [5.0, 3.0, 5.0],
        "CustomerID": [10, 10, 10],
        "InvoiceDate": pd.to_datetime(["2011-01-01", "2011-01-01", "2011-01-02"]),
        "IsCancellationInvoice": [False, False, True],
        "LineRevenue": [10, 3, -5],
    })

def test_positive_sales_excludes_cancellation_and_returns():
    df = positive_sales(sample_data())
    assert len(df) == 2
    assert (df["Quantity"] > 0).all()

def test_sales_summary():
    result = sales_summary(sample_data())
    assert result["positive_sales_value"] == 13
    assert result["positive_units"] == 3
    assert result["invoices"] == 1

def test_monthly_sales():
    result = monthly_sales(sample_data())
    assert len(result) == 1
    assert result.iloc[0]["SalesValue"] == 13
