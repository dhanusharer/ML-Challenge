"""Unit tests for Phase 3 calibration, ensemble, and adaptive rescue coordinator."""

import pytest
import numpy as np

from src.policy.calibration import ScoreCalibrator, normalize_ranker_scores
from src.policy.ensemble import ScoreEnsemble
from src.policy.rescue import AdaptiveRescueCoordinator


def test_score_calibrator_platt_and_isotonic():
    scores = np.array([0.1, 0.4, 0.6, 0.9, 0.95], dtype=np.float32)
    labels = np.array([0, 0, 1, 1, 1], dtype=np.int32)

    platt = ScoreCalibrator(method="platt").fit(scores, labels)
    cal_scores = platt.transform(scores)
    assert len(cal_scores) == len(scores)
    assert np.all(cal_scores >= 0.0) and np.all(cal_scores <= 1.0)
    assert cal_scores[0] < cal_scores[-1]

    iso = ScoreCalibrator(method="isotonic").fit(scores, labels)
    iso_scores = iso.transform(scores)
    assert len(iso_scores) == len(scores)
    assert np.all(iso_scores >= 0.0) and np.all(iso_scores <= 1.0)


def test_normalize_ranker_scores():
    raw_ranks = np.array([-2.5, 0.0, 3.2], dtype=np.float32)
    norm = normalize_ranker_scores(raw_ranks)
    assert np.all(norm >= 0.0) and np.all(norm <= 1.0)
    assert norm[0] < norm[1] < norm[2]
    assert np.isclose(norm[1], 0.5, atol=1e-2)


def test_score_ensemble():
    ens = ScoreEnsemble(weight_classifier=0.70)
    s1 = np.array([0.80, 0.50], dtype=np.float32)
    s2 = np.array([0.90, 0.40], dtype=np.float32)
    blended = ens.blend(s1, s2)
    expected_0 = 0.70 * 0.80 + 0.30 * 0.90  # 0.56 + 0.27 = 0.83
    assert np.isclose(blended[0], expected_0, atol=1e-4)


def test_adaptive_rescue_coordinator():
    coordinator = AdaptiveRescueCoordinator(rescue_threshold=0.70, rescue_margin=0.05)

    # Empty candidate list -> should rescue
    assert coordinator.should_rescue_query([]) is True

    # High confidence top score -> should NOT rescue
    high_conf = [{"target_id": "S2-1", "score": 0.95}]
    assert coordinator.should_rescue_query(high_conf) is False

    # Low top score -> should rescue
    low_conf = [{"target_id": "S2-1", "score": 0.55}]
    assert coordinator.should_rescue_query(low_conf) is True

    # Ambiguous top score with small margin -> should rescue
    ambiguous = [
        {"target_id": "S2-1", "score": 0.78},
        {"target_id": "S3-2", "score": 0.76},  # diff = 0.02 <= 0.05
    ]
    assert coordinator.should_rescue_query(ambiguous) is True
