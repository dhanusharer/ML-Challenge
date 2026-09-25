"""Model ensemble blender for LightGBM Classifier and LambdaMART Ranker."""

from typing import List, Tuple
import numpy as np


class ScoreEnsemble:
    """Blends calibrated classifier scores and normalized ranker scores."""

    def __init__(self, weight_classifier: float = 0.50):
        self.weight_classifier = float(weight_classifier)
        self.weight_ranker = 1.0 - self.weight_classifier

    def blend(self, scores_classifier: np.ndarray, scores_ranker: np.ndarray) -> np.ndarray:
        """Compute weighted linear combination of two score arrays."""
        s1 = np.clip(np.nan_to_num(scores_classifier, nan=0.0), 0.0, 1.0)
        s2 = np.clip(np.nan_to_num(scores_ranker, nan=0.0), 0.0, 1.0)
        return (self.weight_classifier * s1 + self.weight_ranker * s2).astype(np.float32)
