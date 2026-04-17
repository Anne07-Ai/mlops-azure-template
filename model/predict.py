import logging
import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "tenure", "monthly_charges", "total_charges",
    "num_products", "support_calls", "contract_length",
    "age", "satisfaction_score"
]


class CustomerFeatures(BaseModel):
    """Input schema for churn prediction."""
    tenure: int
    monthly_charges: float
    total_charges: float
    num_products: int
    support_calls: int
    contract_length: int
    age: int
    satisfaction_score: float


class PredictionResult(BaseModel):
    """Output schema for churn prediction."""
    churn_prediction: int
    churn_probability: float
    risk_level: str
    recommendation: str


def load_artifacts():
    """Load model and scaler from artifacts folder."""
    try:
        model = joblib.load("artifacts/model.joblib")
        scaler = joblib.load("artifacts/scaler.joblib")
        return model, scaler
    except FileNotFoundError:
        logger.error("Artifacts not found. Run train.py first.")
        raise


def classify_risk(probability: float) -> tuple[str, str]:
    """
    Classify churn risk level and recommendation.

    Args:
        probability: Churn probability score

    Returns:
        Tuple of (risk_level, recommendation)
    """
    if probability >= 0.7:
        return "high", "Immediate intervention required — offer retention package"
    elif probability >= 0.4:
        return "medium", "Monitor closely — consider proactive outreach"
    else:
        return "low", "Customer is stable — maintain current engagement"


def predict(features: CustomerFeatures) -> PredictionResult:
    """
    Predict churn for a single customer.

    Args:
        features: CustomerFeatures input

    Returns:
        PredictionResult with prediction and risk assessment
    """
    model, scaler = load_artifacts()

    df = pd.DataFrame([features.model_dump()])
    X = df[FEATURE_COLS]
    X_scaled = scaler.transform(X)

    prediction = int(model.predict(X_scaled)[0])
    probability = round(float(model.predict_proba(X_scaled)[0][1]), 4)
    risk_level, recommendation = classify_risk(probability)

    logger.info(
        f"Prediction: {prediction} "
        f"Probability: {probability} "
        f"Risk: {risk_level}"
    )

    return PredictionResult(
        churn_prediction=prediction,
        churn_probability=probability,
        risk_level=risk_level,
        recommendation=recommendation
    )


def batch_predict(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run batch predictions on a DataFrame.

    Args:
        df: DataFrame with feature columns

    Returns:
        DataFrame with predictions added
    """
    model, scaler = load_artifacts()
    X = df[FEATURE_COLS]
    X_scaled = scaler.transform(X)

    df["churn_prediction"] = model.predict(X_scaled)
    df["churn_probability"] = model.predict_proba(X_scaled)[:, 1].round(4)
    df["risk_level"] = df["churn_probability"].apply(
        lambda p: classify_risk(p)[0]
    )
    return df