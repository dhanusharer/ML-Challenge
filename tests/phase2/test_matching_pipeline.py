"""Integration test for Phase 2 matching pipeline."""

import pytest
import numpy as np

from src.matching.pipeline import MatchingPipelineRunner
from src.matching.candidate_loader import CandidateArmLoader
from src.matching.negative_sampler import StratifiedNegativeSampler
from src.models.deterministic import DeterministicWeightedMatcher
from src.models.tree_matcher import LightGBMMatcher


def test_pipeline_runner_initialization():
    runner = MatchingPipelineRunner(seed=42)
    assert len(runner.train_s1_ids) > 0
    assert len(runner.val_s1_ids) > 0
    # Strict validation partition isolation
    assert len(set(runner.train_s1_ids).intersection(set(runner.val_s1_ids))) == 0


def test_small_end_to_end_matching_flow():
    runner = MatchingPipelineRunner(seed=42)
    sample_s1_ids = runner.train_s1_ids[:20]

    # Required target IDs
    req_targets = set()
    for sid in sample_s1_ids:
        for mid in runner.ground_truth.get(sid, []):
            req_targets.add(mid)

    # Load 20 dev queries and minimal targets (true links + 500 distractors)
    q_recs, t_recs = runner.load_entity_records(sample_s1_ids, target_ids=req_targets, max_distractors_per_source=250)
    assert len(q_recs) == 20
    assert len(t_recs) >= len(req_targets)

    # Arm B loader
    arm_b_loader = CandidateArmLoader(target_records=t_recs, arm="ARM_B")
    sampler = StratifiedNegativeSampler(negatives_per_positive=3, seed=42)

    X, y, groups, meta = runner.prepare_training_pairs(arm_b_loader, q_recs, sampler)
    assert X.shape[0] == len(y)
    assert X.shape[1] == len(runner.feature_names)
    assert meta["positives"] > 0
    assert meta["negatives"] > 0

    # Fit LightGBM
    matcher = LightGBMMatcher(n_estimators=10)
    matcher.fit(X, y)

    # Evaluate on the cohort
    res = runner.evaluate_model_on_cohort(matcher, arm_b_loader, q_recs)
    assert "blocking_recall" in res
    assert "candidate_conditioned" in res
    assert "best_diagnostic_macro_f05" in res
    assert res["best_diagnostic_macro_f05"] >= 0.0
