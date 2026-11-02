
"""Demand forecasting for RetailPulse.

Uses Prophet when available. If Prophet is not installed, falls back to
a linear trend model. Forecasts are estimates and should be validated
before being used for business decisions.
"""

import numpy as np
import pandas as pd

from src.analytics import monthly_sales


def build_forecast(
    df: pd.DataFrame,
    periods: int = 3,
) -> tuple[pd.DataFrame, str]:
    """Build a monthly sales forecast.

    Args:
        df: Transaction data accepted by monthly_sales().
        periods: Number of future months to forecast.

    Returns:
        A DataFrame containing actual sales and forecast values,
        plus the name of the forecasting engine.
    """
    if not isinstance(periods, int) or isinstance(periods, bool):
        raise ValueError("Forecast periods must be an integer.")

    if not 1 <= periods <= 36:
        raise ValueError("Forecast periods must be between 1 and 36 months.")

    history = monthly_sales(df).rename(
        columns={"Month": "ds", "SalesValue": "y"}
    )

    if history.empty:
        raise ValueError("No usable monthly sales data is available.")

    history["ds"] = pd.to_datetime(history["ds"], errors="coerce")
    history["y"] = pd.to_numeric(history["y"], errors="coerce")
    history = history.replace([np.inf, -np.inf], np.nan)
    history = history.dropna(subset=["ds", "y"])

    # Ensure one row per month, in chronological order.
    history["ds"] = history["ds"].dt.to_period("M").dt.to_timestamp()
    history = (
        history.groupby("ds", as_index=False)["y"]
        .sum()
        .sort_values("ds")
        .reset_index(drop=True)
    )

    if len(history) < 3:
        raise ValueError(
            "At least three months of sales data are required for forecasting."
        )

    if not np.isfinite(history["y"]).all():
        raise ValueError("Sales history contains invalid numeric values.")

    # Complete missing calendar months with zero sales so the timeline
    # is continuous. Confirm this assumption is suitable for your dataset.
    all_months = pd.date_range(
        start=history["ds"].min(),
        end=history["ds"].max(),
        freq="MS",
    )
    history = (
        history.set_index("ds")
        .reindex(all_months, fill_value=0)
        .rename_axis("ds")
        .reset_index()
    )
    history["y"] = pd.to_numeric(history["y"], errors="coerce").fillna(0)

    try:
        from prophet import Prophet
    except ImportError:
        return _linear_trend_forecast(history, periods)

    try:
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
        )
        model.fit(history[["ds", "y"]])

        future_dates = pd.date_range(
            start=history["ds"].max() + pd.offsets.MonthBegin(1),
            periods=periods,
            freq="MS",
        )
        future = pd.DataFrame({"ds": future_dates})

        predicted = model.predict(future)[
            ["ds", "yhat", "yhat_lower", "yhat_upper"]
        ]

        predicted["yhat"] = predicted["yhat"].clip(lower=0)
        predicted["yhat_lower"] = predicted["yhat_lower"].clip(lower=0)
        predicted["yhat_upper"] = predicted["yhat_upper"].clip(lower=0)

        actual = history.copy()
        actual["yhat"] = actual["y"]
        actual["yhat_lower"] = np.nan
        actual["yhat_upper"] = np.nan
        actual["Type"] = "Actual"

        predicted["y"] = np.nan
        predicted["Type"] = "Forecast"

        result = pd.concat(
            [
                actual[
                    ["ds", "y", "yhat", "yhat_lower", "yhat_upper", "Type"]
                ],
                predicted[
                    ["ds", "y", "yhat", "yhat_lower", "yhat_upper", "Type"]
                ],
            ],
            ignore_index=True,
        )

        return result.sort_values("ds").reset_index(drop=True), "Prophet"

    except Exception as exc:
        # Do not silently disguise Prophet fitting errors as a successful
        # Prophet forecast. Use the documented fallback and expose the reason.
        fallback, engine = _linear_trend_forecast(history, periods)
        return fallback, f"{engine} (Prophet failed: {type(exc).__name__})"


def _linear_trend_forecast(
    history: pd.DataFrame,
    periods: int,
) -> tuple[pd.DataFrame, str]:
    """Generate a simple linear-trend forecast when Prophet is unavailable."""
    hist = history[["ds", "y"]].copy().reset_index(drop=True)

    x = np.arange(len(hist), dtype=float)
    y = hist["y"].to_numpy(dtype=float)

    if len(y) < 2 or not np.isfinite(y).all():
        raise ValueError("Not enough valid sales history for a trend forecast.")

    slope, intercept = np.polyfit(x, y, 1)

    future_x = np.arange(len(hist), len(hist) + periods, dtype=float)
    future_y = np.maximum(0, intercept + slope * future_x)

    actual = hist.copy()
    actual["yhat"] = actual["y"]
    actual["yhat_lower"] = np.nan
    actual["yhat_upper"] = np.nan
    actual["Type"] = "Actual"

    future_dates = pd.date_range(
        start=hist["ds"].max() + pd.offsets.MonthBegin(1),
        periods=periods,
        freq="MS",
    )

    future = pd.DataFrame(
        {
            "ds": future_dates,
            "y": np.nan,
            "yhat": future_y,
            "yhat_lower": np.nan,
            "yhat_upper": np.nan,
            "Type": "Forecast",
        }
    )

    result = pd.concat([actual, future], ignore_index=True)
    return (
        result.sort_values("ds").reset_index(drop=True),
        "Linear trend fallback",
    )
