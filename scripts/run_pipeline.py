
import mlflow

from src.data_loader import load_transactions, load_customer_features
from src.analytics import sales_summary
from src.segmentation import segment_customers
from src.churn import add_proxy_risk
from src.forecasting import build_forecast
from src.inventory import inventory_recommendations
from src.config import PROCESSED_DIR


def main():
    mlflow.set_experiment("RetailPulse-Pipeline")

    with mlflow.start_run(run_name="retailpulse-pipeline"):
        tx = load_transactions()
        customers = load_customer_features()

        summary = sales_summary(tx)
        print("Sales summary:", summary)

        segments = segment_customers(customers)
        risk = add_proxy_risk(customers)
        forecast, engine = build_forecast(tx, periods=3)
        inventory = inventory_recommendations(tx)

        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

        outputs = {
            "customer_segments.csv": segments,
            "customer_churn_risk.csv": risk,
            "demand_forecast.csv": forecast,
            "inventory_recommendations.csv": inventory,
        }

        for filename, dataframe in outputs.items():
            output_path = PROCESSED_DIR / filename
            dataframe.to_csv(output_path, index=False)
            mlflow.log_artifact(str(output_path))

        # Record pipeline configuration and dataset sizes.
        mlflow.log_param("forecast_engine", engine)
        mlflow.log_param("forecast_periods", 3)
        mlflow.log_param("segmentation_algorithm", "KMeans")
        mlflow.log_metric("transaction_rows", len(tx))
        mlflow.log_metric("customer_rows", len(customers))
        mlflow.log_metric("segment_rows", len(segments))
        mlflow.log_metric("churn_risk_rows", len(risk))
        mlflow.log_metric("forecast_rows", len(forecast))
        mlflow.log_metric("inventory_rows", len(inventory))

        print(f"Pipeline completed. Forecast engine: {engine}")
        print("Outputs saved under data/processed/")
        print("MLflow tracking enabled.")


if __name__ == "__main__":
    main()
