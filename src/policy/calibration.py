"""Probability calibration and score normalization for Phase 3 matchers."""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression


class ScoreCalibrator:
    """Calibrates model output scores into well-behaved posterior probabilities."""

    def __init__(self, method: str = "none"):
        self.method = method.lower()
        self.calibrator = None

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> "ScoreCalibrator":
        """Fit calibration mapping on development data."""
        if self.method == "none":
            return self

        scores_clean = np.clip(np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)

        if self.method == "platt":
            self.calibrator = LogisticRegression(C=1.0, solver="lbfgs", max_iter=200)
            X = scores_clean.reshape(-1, 1)
            self.calibrator.fit(X, labels)
        elif self.method == "isotonic":
            self.calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            self.calibrator.fit(scores_clean, labels)

        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        """Apply calibration mapping to raw scores."""
        scores_clean = np.clip(np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)

        if self.method == "none" or self.calibrator is None:
            return scores_clean

        if self.method == "platt":
            X = scores_clean.reshape(-1, 1)
            return self.calibrator.predict_proba(X)[:, 1].astype(np.float32)
        elif self.method == "isotonic":
            return self.calibrator.transform(scores_clean).astype(np.float32)

        return scores_clean


def normalize_ranker_scores(scores: np.ndarray) -> np.ndarray:
    """Normalize raw uncalibrated LambdaMART ranking scores into [0, 1].

    Applies sigmoid transformation for smooth monotonic probability mapping.
    """
    scores_arr = np.nan_to_num(scores, nan=0.0, posinf=10.0, neginf=-10.0)
    # Sigmoid mapping centered at 0
    return (1.0 / (1.0 + np.exp(-scores_arr))).astype(np.float32)
