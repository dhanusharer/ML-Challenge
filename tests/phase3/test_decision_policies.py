"""Unit tests for Phase 3 decision policies and constraint adherence."""

import pytest
import numpy as np

from src.policy.decision import (
    GlobalThresholdPolicy,
    TopKThresholdPolicy,
    RelativeMarginPolicy,
    AdaptiveRatioPolicy,
)


def test_rule_5_s2_s3_only_and_deduplication():
    """Verify that predictions contain ONLY S2/S3 IDs and deduplicate duplicate targets."""
    policy = GlobalThresholdPolicy(threshold=0.50)
    cands = [
        {"target_id": "S1-999", "score": 0.99},  # S1 ID - MUST BE REJECTED
        {"target_id": "S2-100", "score": 0.85},
        {"target_id": "S2-100", "score": 0.80},  # Duplicate - keep highest score
        {"target_id": "S3-200", "score": 0.75},
        {"target_id": "OTHER-55", "score": 0.90}, # Invalid prefix - MUST BE REJECTED
    ]
    preds = policy.predict_matches("S1-1", cands)
    assert "S1-999" not in preds
    assert "OTHER-55" not in preds
    assert preds == ["S2-100", "S3-200"]


def test_deterministic_tie_breaking_and_nan_handling():
    """Verify deterministic tie-breaking by target_id and NaN/inf score handling."""
    policy = GlobalThresholdPolicy(threshold=0.50)
    cands = [
        {"target_id": "S2-B", "score": 0.80},
        {"target_id": "S2-A", "score": 0.80},  # Exact score tie: S2-A should precede S2-B
        {"target_id": "S2-C", "score": float("nan")},  # NaN score
        {"target_id": "S3-D", "score": float("inf")},  # Inf score
    ]
    preds = policy.predict_matches("S1-2", cands)
    assert preds[0] == "S2-A"
    assert preds[1] == "S2-B"
    assert "S2-C" not in preds


def test_singleton_protection_relative_margin_policy():
    """Verify RelativeMarginPolicy predicts empty set when top score is below floor."""
    policy = RelativeMarginPolicy(threshold_floor=0.75, multi_threshold=0.80, max_margin=0.08, max_k=3)

    # Case A: True singleton / weak candidate pool -> must predict []
    weak_cands = [
        {"target_id": "S2-10", "score": 0.65},
        {"target_id": "S3-20", "score": 0.60},
    ]
    assert policy.predict_matches("S1-Singleton", weak_cands) == []

    # Case B: Clear single match (dominant top score) -> must predict only top match
    dominant_cands = [
        {"target_id": "S2-10", "score": 0.92},
        {"target_id": "S3-20", "score": 0.70},  # Below multi_threshold and margin > 0.08
    ]
    assert policy.predict_matches("S1-OneMatch", dominant_cands) == ["S2-10"]

    # Case C: Genuine multi-match (closely clustered high scores) -> predict both
    multi_cands = [
        {"target_id": "S2-10", "score": 0.92},
        {"target_id": "S3-20", "score": 0.89},  # score >= 0.80 and margin (0.03) <= 0.08
    ]
    assert policy.predict_matches("S1-MultiMatch", multi_cands) == ["S2-10", "S3-20"]


def test_topk_and_adaptive_ratio_policies():
    topk_policy = TopKThresholdPolicy(threshold=0.60, max_k=2)
    cands = [
        {"target_id": "S2-1", "score": 0.90},
        {"target_id": "S2-2", "score": 0.85},
        {"target_id": "S2-3", "score": 0.80},
    ]
    preds = topk_policy.predict_matches("S1-X", cands)
    assert len(preds) == 2
    assert preds == ["S2-1", "S2-2"]

    ratio_policy = AdaptiveRatioPolicy(threshold_floor=0.70, min_ratio=0.90, max_k=3)
    # 0.90 * 0.90 = 0.81. S2-2 (0.85) >= 0.81 (keep). S2-3 (0.75) < 0.81 (drop).
    ratio_preds = ratio_policy.predict_matches("S1-Y", cands)
    assert ratio_preds == ["S2-1", "S2-2"]
