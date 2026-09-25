"""Phase 3: Entity-Level F0.5 Optimization & Decision Policy Lab.

Provides calibrated decision policies, score calibration, ensembles,
adaptive retrieval rescue passes, and official entity-level evaluation.
"""

from src.policy.decision import (
    BaseDecisionPolicy,
    GlobalThresholdPolicy,
    TopKThresholdPolicy,
    RelativeMarginPolicy,
    AdaptiveRatioPolicy,
)
from src.policy.calibration import ScoreCalibrator
from src.policy.ensemble import ScoreEnsemble
from src.policy.rescue import AdaptiveRescueCoordinator

__all__ = [
    "BaseDecisionPolicy",
    "GlobalThresholdPolicy",
    "TopKThresholdPolicy",
    "RelativeMarginPolicy",
    "AdaptiveRatioPolicy",
    "ScoreCalibrator",
    "ScoreEnsemble",
    "AdaptiveRescueCoordinator",
]
