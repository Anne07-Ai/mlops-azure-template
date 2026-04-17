import os
import logging
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = os.getenv("MODEL_NAME", "churn-classifier")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "mlruns")


def generate_sample_data(n_samples: int = 1000) -> pd.DataFrame:
    """
    Generate synthetic customer churn dataset.

    Args:
        n_samples: Number of samples to generate

    Returns:
        DataFrame with features and target
    """
    np.random.seed(42)
    df = pd.DataFrame({
        "tenure": np.random.randint(1, 72, n_samples),
        "monthly_charges": np.random.uniform(20, 120, n_samples),
        "total_charges": np.random.uniform(100, 8000, n_samples),
        "num_products": np.random.randint(1, 5, n_samples),
        "support_calls": np.random.randint(0, 10, n_samples),
        "contract_length": np.random.choice([1, 12, 24], n_samples),
        "age": np.random.randint(18, 80, n_samples),
        "satisfaction_score": np.random.uniform(1, 10, n_samples),
    })

    churn_prob = (
        0.3
        - 0.004 * df["tenure"]
        + 0.002 * df["monthly_charges"]
        - 0.001 * df["contract_length"]
        + 0.02 * df["support_calls"]
        - 0.01 * df["satisfaction_score"]
    )
    df["churn"] = (churn_prob + np.random.normal(0, 0.1, n_samples) > 0.3).astype(int)
    return df


def train_model(df: pd.DataFrame) -> dict:
    """
    Train a RandomForest churn classifier with MLflow tracking.

    Args:
        df: Training DataFrame with features and target

    Returns:
        dict with model, scaler, and metrics
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MODEL_NAME)

    feature_cols = [
        "tenure", "monthly_charges", "total_charges",
        "num_products", "support_calls", "contract_length",
        "age", "satisfaction_score"
    ]

    X = df[feature_cols]
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    params = {
        "n_estimators": 100,
        "max_depth": 6,
        "min_samples_split": 5,
        "random_state": 42
    }

    with mlflow.start_run():
        mlflow.log_params(params)

        model = RandomForestClassifier(**params)
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]

        metrics = {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "f1_score": round(f1_score(y_test, y_pred), 4),
            "roc_auc": round(roc_auc_score(y_test, y_prob), 4)
        }

        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, MODEL_NAME)

        logger.info(f"Training complete — metrics: {metrics}")

    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(model, "artifacts/model.joblib")
    joblib.dump(scaler, "artifacts/scaler.joblib")

    return {
        "model": model,
        "scaler": scaler,
        "metrics": metrics,
        "feature_cols": feature_cols
    }


if __name__ == "__main__":
    logger.info("Generating training data...")
    df = generate_sample_data(1000)
    logger.info("Training model...")
    result = train_model(df)
    logger.info(f"Done! Metrics: {result['metrics']}")