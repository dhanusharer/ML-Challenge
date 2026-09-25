"""Unit tests for Phase 2.1 Stability Gate and Evaluator consistency."""

import pytest
import numpy as np

from src.matching.evaluator import MatcherEvaluator


def test_evaluator_singleton_and_multimatch_metrics():
    gt = {
        "S1-1": ["S2-10", "S3-11"],  # multi-match
        "S1-2": ["S2-20"],           # 1-match
        "S1-3": [],                  # singleton
    }
    evaluator = MatcherEvaluator(ground_truth=gt)

    scored_cohort = [
        {
            "query_record": {"entity_id": "S1-1", "business_name": "A", "business_address": "Addr A"},
            "ground_truth_targets": {"S2-10", "S3-11"},
            "candidates": [
                {"target_record": {"entity_id": "S2-10"}, "score": 0.85, "label": 1},
                {"target_record": {"entity_id": "S3-11"}, "score": 0.75, "label": 1},
                {"target_record": {"entity_id": "S2-99"}, "score": 0.15, "label": 0},
            ],
        },
        {
            "query_record": {"entity_id": "S1-2", "business_name": "B", "business_address": "Addr B"},
            "ground_truth_targets": {"S2-20"},
            "candidates": [
                {"target_record": {"entity_id": "S2-20"}, "score": 0.90, "label": 1},
                {"target_record": {"entity_id": "S3-88"}, "score": 0.20, "label": 0},
            ],
        },
        {
            "query_record": {"entity_id": "S1-3", "business_name": "C", "business_address": "Addr C"},
            "ground_truth_targets": set(),
            "candidates": [
                {"target_record": {"entity_id": "S2-77"}, "score": 0.10, "label": 0},
            ],
        },
    ]

    sweeps = evaluator.evaluate_end_to_end_sweep(scored_cohort, thresholds=(0.3, 0.5, 0.7))
    assert len(sweeps) == 3
    for s in sweeps:
        assert 0.0 <= s["macro_f05"] <= 1.0
        assert 0.0 <= s["end_to_end_link_recall"] <= 1.0
