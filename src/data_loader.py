
import pandas as pd
from src.config import CLEANED_DATA, CUSTOMER_FEATURES


def load_transactions(path=None) -> pd.DataFrame:
    if path is None:
        path = CLEANED_DATA

    if not path.exists() if hasattr(path, "exists") else False:
        raise FileNotFoundError(f"Transaction dataset not found: {path}")

    df = pd.read_csv(path, low_memory=False, compression="infer")

    if "InvoiceDate" in df:
        df["InvoiceDate"] = pd.to_datetime(
            df["InvoiceDate"], errors="coerce"
        )

    for col in ("Quantity", "UnitPrice", "LineRevenue", "CustomerID"):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def load_customer_features(path: str = str(CUSTOMER_FEATURES)) -> pd.DataFrame:
    df = pd.read_csv(path)

    for col in ("FirstPurchaseDate", "LastPurchaseDate"):
        if col in df:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    return df


def positive_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Return purchase lines used for analytical sales, excluding cancellations/returns."""
    mask = (df["Quantity"] > 0) & (df["UnitPrice"] >= 0)

    if "IsCancellationInvoice" in df:
        mask &= ~df["IsCancellationInvoice"].fillna(False).astype(bool)
    elif "InvoiceNo" in df:
        mask &= ~df["InvoiceNo"].astype(str).str.startswith("C", na=False)

    return df.loc[mask].copy()
