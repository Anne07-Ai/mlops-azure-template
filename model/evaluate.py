import logging
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from model.train import generate_sample_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "tenure", "monthly_charges", "total_charges",
    "num_products", "support_calls", "contract_length",
    "age", "satisfaction_score"
]


def load_artifacts() -> tuple:
    """
    Load saved model and scaler from artifacts folder.

    Returns:
        Tuple of (model, scaler)
    """
    try:
        model = joblib.load("artifacts/model.joblib")
        scaler = joblib.load("artifacts/scaler.joblib")
        logger.info("Artifacts loaded successfully")
        return model, scaler
    except FileNotFoundError:
        logger.error("Artifacts not found. Run train.py first.")
        raise


def evaluate_model(df: pd.DataFrame = None) -> dict:
    """
    Evaluate model on a dataset.

    Args:
        df: Optional DataFrame. Generates sample data if not provided.

    Returns:
        dict with full evaluation metrics
    """
    model, scaler = load_artifacts()

    if df is None:
        df = generate_sample_data(500)

    X = df[FEATURE_COLS]
    y = df["churn"]

    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    y_prob = model.predict_proba(X_scaled)[:, 1]

    cm = confusion_matrix(y, y_pred)
    report = classification_report(y, y_pred, output_dict=True)

    metrics = {
        "accuracy": round(accuracy_score(y, y_pred), 4),
        "f1_score": round(f1_score(y, y_pred), 4),
        "roc_auc": round(roc_auc_score(y, y_prob), 4),
        "confusion_matrix": cm.tolist(),
        "classification_report": report
    }

    logger.info(f"Evaluation complete — accuracy: {metrics['accuracy']}")
    return metrics


def check_model_quality(metrics: dict, thresholds: dict = None) -> dict:
    """
    Check if model meets quality thresholds for deployment.

    Args:
        metrics: Evaluation metrics dict
        thresholds: Min thresholds for each metric

    Returns:
        dict with pass/fail status per metric
    """
    if thresholds is None:
        thresholds = {
            "accuracy": 0.75,
            "f1_score": 0.70,
            "roc_auc": 0.80
        }

    results = {}
    all_passed = True

    for metric, threshold in thresholds.items():
        passed = metrics[metric] >= threshold
        results[metric] = {
            "value": metrics[metric],
            "threshold": threshold,
            "passed": passed
        }
        if not passed:
            all_passed = False
            logger.warning(
                f"Quality check FAILED: {metric} = "
                f"{metrics[metric]} < {threshold}"
            )
        else:
            logger.info(
                f"Quality check PASSED: {metric} = "
                f"{metrics[metric]} >= {threshold}"
            )

    results["overall_passed"] = all_passed
    return results


if __name__ == "__main__":
    metrics = evaluate_model()
    quality = check_model_quality(metrics)
    print(f"Overall passed: {quality['overall_passed']}")
    