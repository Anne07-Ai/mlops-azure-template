import pytest
import numpy as np
import pandas as pd
from model.train import generate_sample_data, train_model
from model.evaluate import evaluate_model, check_model_quality
from model.predict import predict, batch_predict, CustomerFeatures
from monitoring.drift_detector import DriftDetector


@pytest.fixture
def sample_df():
    return generate_sample_data(200)


@pytest.fixture
def trained_model(sample_df):
    return train_model(sample_df)


def test_generate_data():
    df = generate_sample_data(100)
    assert len(df) == 100
    assert "churn" in df.columns
    assert df["churn"].isin([0, 1]).all()


def test_train_model(trained_model):
    assert "model" in trained_model
    assert "metrics" in trained_model
    assert "accuracy" in trained_model["metrics"]
    assert trained_model["metrics"]["accuracy"] > 0.5


def test_model_metrics(trained_model):
    metrics = trained_model["metrics"]
    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["f1_score"] <= 1
    assert 0 <= metrics["roc_auc"] <= 1


def test_evaluate_model(sample_df, trained_model):
    metrics = evaluate_model(sample_df)
    assert "accuracy" in metrics
    assert "f1_score" in metrics
    assert "roc_auc" in metrics
    assert "confusion_matrix" in metrics


def test_quality_check_pass(trained_model):
    metrics = trained_model["metrics"]
    result = check_model_quality(
        metrics,
        thresholds={"accuracy": 0.5, "f1_score": 0.3, "roc_auc": 0.5}
    )
    assert result["overall_passed"] is True


def test_quality_check_fail():
    metrics = {"accuracy": 0.4, "f1_score": 0.3, "roc_auc": 0.5}
    result = check_model_quality(
        metrics,
        thresholds={"accuracy": 0.9, "f1_score": 0.9, "roc_auc": 0.9}
    )
    assert result["overall_passed"] is False


def test_predict_single(trained_model):
    features = CustomerFeatures(
        tenure=12,
        monthly_charges=65.0,
        total_charges=780.0,
        num_products=2,
        support_calls=3,
        contract_length=12,
        age=35,
        satisfaction_score=6.5
    )
    result = predict(features)
    assert result.churn_prediction in [0, 1]
    assert 0 <= result.churn_probability <= 1
    assert result.risk_level in ["low", "medium", "high"]
    assert len(result.recommendation) > 0


def test_batch_predict(sample_df, trained_model):
    result = batch_predict(sample_df.head(10))
    assert "churn_prediction" in result.columns
    assert "churn_probability" in result.columns
    assert "risk_level" in result.columns


def test_drift_detector_no_drift():
    detector = DriftDetector(drift_threshold=0.15)
    ref_df = generate_sample_data(500)
    cur_df = generate_sample_data(500)
    detector.set_reference(ref_df)
    result = detector.detect_drift(cur_df)
    assert "overall_drift_detected" in result
    assert "feature_results" in result


def test_drift_detector_with_drift():
    detector = DriftDetector(drift_threshold=0.05)
    ref_df = generate_sample_data(500)
    cur_df = generate_sample_data(500)
    cur_df["monthly_charges"] = cur_df["monthly_charges"] * 3
    cur_df["tenure"] = cur_df["tenure"] * 0.1
    detector.set_reference(ref_df)
    result = detector.detect_drift(cur_df)
    assert result["overall_drift_detected"] is True


def test_drift_history():
    detector = DriftDetector()
    df = generate_sample_data(200)
    detector.set_reference(df)
    detector.detect_drift(df)
    detector.detect_drift(df)
    assert len(detector.get_drift_history()) == 2