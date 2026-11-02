import json
from pathlib import Path
from joblib import dump
from src.data_loader import load_customer_features
from src.churn import train_proxy_churn_model
from src.config import MODEL_DIR

def main():
    customers = load_customer_features()
    model, metrics, features = train_proxy_churn_model(customers)
    MODEL_DIR.mkdir(exist_ok=True)
    dump(model, MODEL_DIR / "proxy_churn_model.joblib")
    with open(MODEL_DIR / "proxy_churn_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "features": features}, f, indent=2)
    print(json.dumps({"metrics": metrics, "features": features}, indent=2))
    print("NOTE: target is a recency-based proxy, not verified churn.")
if __name__ == "__main__":
    main()
