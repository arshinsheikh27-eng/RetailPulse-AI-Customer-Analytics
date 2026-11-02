
"""Churn-risk analysis for RetailPulse.

The dataset does not contain verified churn labels. This module therefore
uses a recency-based proxy for demonstration purposes. Model metrics must not
be presented as validated real-world churn performance.
"""

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline


FEATURES = [
    "Frequency",
    "Monetary",
    "UniqueProductsPurchased",
    "TotalUnitsPurchased",
    "AverageOrderValue",
    "CustomerTenureDays",
    "AverageItemsPerInvoice",
    "ReturnOrCancellationLineCount",
]


def add_proxy_risk(
    customers: pd.DataFrame,
    threshold_days: int = 90,
) -> pd.DataFrame:
    """Assign Low, Medium, or High risk using days since the last purchase."""
    if "RecencyDays" not in customers.columns:
        raise ValueError("Customer data must contain RecencyDays.")

    if not 1 <= threshold_days < 180:
        raise ValueError("threshold_days must be between 1 and 179.")

    out = customers.copy()
    recency = pd.to_numeric(out["RecencyDays"], errors="coerce")

    # Missing recency remains missing instead of being treated as low risk.
    risk = pd.Series(pd.NA, index=out.index, dtype="object")
    risk.loc[recency.notna() & (recency <= threshold_days)] = "Low"
    risk.loc[
        recency.notna()
        & (recency > threshold_days)
        & (recency <= 180)
    ] = "Medium"
    risk.loc[recency.notna() & (recency > 180)] = "High"

    out["ChurnRisk"] = risk
    return out


def train_proxy_churn_model(
    customers: pd.DataFrame,
    threshold_days: int = 90,
):
    """Train a demonstration classifier using a recency-derived proxy target.

    Returns:
        model: fitted sklearn pipeline
        metrics: holdout metrics and proxy-target definition
        features: feature names used by the model
    """
    if "RecencyDays" not in customers.columns:
        raise ValueError("Customer data must contain RecencyDays.")

    if threshold_days < 1:
        raise ValueError("threshold_days must be at least 1.")

    data = customers.copy()
    data["RecencyDays"] = pd.to_numeric(
        data["RecencyDays"], errors="coerce"
    )
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=["RecencyDays"])

    if len(data) < 20:
        raise ValueError(
            "At least 20 customers with valid recency values are required."
        )

    # This is a proxy label, not a verified churn outcome.
    data["ProxyChurn"] = (
        data["RecencyDays"] > threshold_days
    ).astype(int)

    features = [column for column in FEATURES if column in data.columns]
    if not features:
        raise ValueError(
            "No supported model features were found in the customer dataset."
        )

    X = data[features].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    y = data["ProxyChurn"]

    if y.nunique() < 2:
        raise ValueError(
            "The proxy target contains only one class. Try a different threshold."
        )

    if y.value_counts().min() < 2:
        raise ValueError(
            "Each proxy class needs at least two customers for a stratified split."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    model = make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=150,
            random_state=42,
            class_weight="balanced",
        ),
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, predictions, zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "target_definition": (
            f"Proxy churn = RecencyDays > {threshold_days}; "
            "not a verified churn label."
        ),
        "test_customers": int(len(y_test)),
        "training_customers": int(len(y_train)),
    }

    return model, metrics, features


def predict_proxy_churn(
    customers: pd.DataFrame,
    threshold_days: int = 90,
) -> pd.DataFrame:
    """Add transparent recency-based risk and proxy-target columns."""
    out = add_proxy_risk(customers, threshold_days)
    recency = pd.to_numeric(out["RecencyDays"], errors="coerce")

    out["ProxyChurn"] = pd.Series(
        pd.NA, index=out.index, dtype="Int64"
    )
    valid = recency.notna()
    out.loc[valid, "ProxyChurn"] = (
        recency.loc[valid] > threshold_days
    ).astype(int)

    return out
