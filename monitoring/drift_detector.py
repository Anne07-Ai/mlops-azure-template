import logging
import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "tenure", "monthly_charges", "total_charges",
    "num_products", "support_calls", "contract_length",
    "age", "satisfaction_score"
]


class DriftDetector:
    """
    Statistical drift detector for ML model monitoring.
    Uses KS test and PSI to detect data and concept drift.
    """

    def __init__(self, drift_threshold: float = 0.15):
        self.drift_threshold = drift_threshold
        self.reference_data = None
        self.drift_history = []

    def set_reference(self, df: pd.DataFrame):
        """
        Set reference dataset for drift comparison.

        Args:
            df: Reference DataFrame
        """
        self.reference_data = df[FEATURE_COLS].copy()
        logger.info(
            f"Reference data set with {len(df)} samples"
        )

    def compute_ks_test(
        self,
        reference: pd.Series,
        current: pd.Series
    ) -> dict:
        """
        Run KS test between reference and current distributions.

        Args:
            reference: Reference data series
            current: Current data series

        Returns:
            dict with statistic and p_value
        """
        statistic, p_value = stats.ks_2samp(reference, current)
        return {
            "statistic": round(float(statistic), 4),
            "p_value": round(float(p_value), 4),
            "drift_detected": p_value < 0.05
        }

    def compute_psi(
        self,
        reference: pd.Series,
        current: pd.Series,
        bins: int = 10
    ) -> float:
        """
        Compute Population Stability Index (PSI).

        Args:
            reference: Reference data series
            current: Current data series
            bins: Number of bins for PSI calculation

        Returns:
            PSI score
        """
        ref_min = reference.min()
        ref_max = reference.max()
        bin_edges = np.linspace(ref_min, ref_max, bins + 1)

        ref_counts = np.histogram(reference, bins=bin_edges)[0]
        cur_counts = np.histogram(current, bins=bin_edges)[0]

        ref_pct = (ref_counts + 1e-6) / len(reference)
        cur_pct = (cur_counts + 1e-6) / len(current)

        psi = np.sum(
            (cur_pct - ref_pct) * np.log(cur_pct / ref_pct)
        )
        return round(float(psi), 4)

    def detect_drift(self, current_df: pd.DataFrame) -> dict:
        """
        Detect drift between reference and current data.

        Args:
            current_df: Current production DataFrame

        Returns:
            dict with drift results per feature
        """
        if self.reference_data is None:
            raise ValueError(
                "Reference data not set. Call set_reference() first."
            )

        results = {}
        drift_detected = False

        for col in FEATURE_COLS:
            ks_result = self.compute_ks_test(
                self.reference_data[col],
                current_df[col]
            )
            psi_score = self.compute_psi(
                self.reference_data[col],
                current_df[col]
            )

            feature_drift = (
                ks_result["drift_detected"] or
                psi_score > self.drift_threshold
            )

            if feature_drift:
                drift_detected = True
                logger.warning(
                    f"Drift detected in {col} — "
                    f"PSI: {psi_score} KS p-value: "
                    f"{ks_result['p_value']}"
                )

            results[col] = {
                "ks_statistic": ks_result["statistic"],
                "ks_p_value": ks_result["p_value"],
                "psi_score": psi_score,
                "drift_detected": feature_drift
            }

        summary = {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_drift_detected": drift_detected,
            "features_with_drift": [
                col for col, r in results.items()
                if r["drift_detected"]
            ],
            "feature_results": results
        }

        self.drift_history.append(summary)
        logger.info(
            f"Drift check complete — "
            f"drift detected: {drift_detected}"
        )
        return summary

    def get_drift_history(self) -> list:
        """Get history of all drift checks."""
        return self.drift_history