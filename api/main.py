from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI, HTTPException
from src.data_loader import load_transactions, load_customer_features
from src.analytics import sales_summary, monthly_sales
from src.segmentation import segment_customers
from src.churn import add_proxy_risk
from src.forecasting import build_forecast

app = FastAPI(title="RetailPulse API", version="1.0.0",
              description="Local API for retail analytics and demonstration ML endpoints.")

@app.get("/health")
def health():
    return {"status": "ok", "service": "RetailPulse API"}

@app.get("/summary")
def summary():
    try:
        return sales_summary(load_transactions())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.get("/sales/monthly")
def monthly():
    try:
        df = monthly_sales(load_transactions())
        df["Month"] = df["Month"].astype(str)
        return df.to_dict(orient="records")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.get("/customers/segments")
def segments(clusters: int = 4, limit: int = 100):
    try:
        data = segment_customers(load_customer_features(), clusters).head(limit)
        return data.fillna("").to_dict(orient="records")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.get("/customers/churn-risk")
def churn_risk(threshold_days: int = 90, limit: int = 100):
    try:
        data = add_proxy_risk(load_customer_features(), threshold_days)
        return data.sort_values("RecencyDays", ascending=False).head(limit).fillna("").to_dict(orient="records")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.get("/forecast")
def forecast(periods: int = 3):
    try:
        data, engine = build_forecast(load_transactions(), periods)
        data["ds"] = data["ds"].astype(str)
        return {"engine": engine, "forecast": data.tail(periods).fillna("").to_dict(orient="records")}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
