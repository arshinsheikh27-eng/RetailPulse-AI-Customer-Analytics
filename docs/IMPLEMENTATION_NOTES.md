# Implementation notes

## Data contract
- `cleaned_retail.csv`: cleaned transaction-level rows and derived flags.
- `customer_features.csv`: one row per identifiable customer with RFM-style features.

## Modules
- `src/analytics.py`: positive sales metrics and group-by aggregations.
- `src/segmentation.py`: K-Means on log-transformed Recency, Frequency and Monetary values.
- `src/churn.py`: recency threshold risk bands and a proxy-label model helper.
- `src/forecasting.py`: Prophet forecasting or an explicitly labelled linear trend fallback.
- `src/inventory.py`: average daily product demand and assumed reorder-point estimates.
- `src/monitoring.py`: basic PSI-style numeric drift report helper.
- `api/main.py`: local FastAPI serving layer.
- `app.py`: Streamlit dashboard.

## Next hardening tasks
1. Validate forecasting with a time-based holdout and report MAE/RMSE/MAPE where meaningful.
2. Replace proxy churn with a future-window label when additional customer history is available.
3. Add MLflow tracking around model training and metrics.
4. Add data drift baselines and scheduled monitoring.
5. Add real inventory and supplier data before operational use.
6. Add CI workflow, Dockerfile and compose configuration after the local workflow runs successfully.
