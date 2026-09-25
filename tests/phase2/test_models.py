"""Unit tests for StratifiedNegativeSampler and model families."""

import pytest
import numpy as np

from src.matching.negative_sampler import StratifiedNegativeSampler
from src.models.deterministic import DeterministicWeightedMatcher
from src.models.logistic import LogisticRegressionMatcher
from src.models.tree_matcher import LightGBMMatcher
from src.models.ltr_matcher import LightGBMRankerMatcher


def test_stratified_negative_sampler_proportions():
    sampler = StratifiedNegativeSampler(negatives_per_positive=4, seed=42)
    s1_id = "S1-999"
    gt_targets = {"S2-1", "S3-1"}

    candidates = [
        {"target_record": {"entity_id": "S2-1"}, "provenance": {"tfidf_rank": 1, "support_count": 3}},
        {"target_record": {"entity_id": "S3-1"}, "provenance": {"tfidf_rank": 2, "support_count": 2}},
        # Negatives with different strata
        {"target_record": {"entity_id": "S2-N1"}, "provenance": {"tfidf_rank": 3, "tfidf_score": 0.8, "support_count": 1}},
        {"target_record": {"entity_id": "S2-N2"}, "provenance": {"tfidf_rank": 4, "tfidf_score": 0.7, "support_count": 1}},
        {"target_record": {"entity_id": "S2-N3"}, "provenance": {"support_count": 3}},
        {"target_record": {"entity_id": "S3-N4"}, "provenance": {"exact_name": True, "support_count": 1}},
        {"target_record": {"entity_id": "S3-N5"}, "provenance": {"name_numeric": True, "support_count": 1}},
        {"target_record": {"entity_id": "S2-N6"}, "provenance": {"support_count": 1}},
        {"target_record": {"entity_id": "S3-N7"}, "provenance": {"support_count": 1}},
        {"target_record": {"entity_id": "S2-N8"}, "provenance": {"support_count": 1}},
        {"target_record": {"entity_id": "S3-N9"}, "provenance": {"support_count": 1}},
        {"target_record": {"entity_id": "S2-N10"}, "provenance": {"support_count": 1}},
    ]

    sampled_negs = sampler.sample_query_negatives(s1_id, candidates, gt_targets)
    # Expected target negatives = 2 positives * 4 = 8 negatives
    assert len(sampled_negs) == 8
    # Ensure no true positive was included in sampled negatives
    for n in sampled_negs:
        assert n["target_record"]["entity_id"] not in gt_targets


def test_model_families_fit_and_predict():
    # Synthetic small dataset: 100 samples, 10 features
    rng = np.random.RandomState(42)
    X = rng.rand(100, 10).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 1.0).astype(np.int32)
    feature_names = [f"feat_{i}" for i in range(10)]
    feature_names[0] = "name_token_jaccard"
    feature_names[1] = "addr_token_jaccard"

    # 1. Deterministic baseline
    det = DeterministicWeightedMatcher()
    det.fit(X, y, feature_names=feature_names)
    s_det = det.predict_proba(X)
    assert s_det.shape == (100,)
    assert (s_det >= 0.0).all() and (s_det <= 1.0).all()

    # 2. Logistic Regression
    lr = LogisticRegressionMatcher()
    lr.fit(X, y)
    s_lr = lr.predict_proba(X)
    assert s_lr.shape == (100,)
    assert (s_lr >= 0.0).all() and (s_lr <= 1.0).all()

    # 3. LightGBM Classifier
    lgb = LightGBMMatcher(n_estimators=10)
    lgb.fit(X, y)
    s_lgb = lgb.predict_proba(X)
    assert s_lgb.shape == (100,)
    assert (s_lgb >= 0.0).all() and (s_lgb <= 1.0).all()

    # 4. LightGBM Ranker
    groups = [25, 25, 25, 25]  # 4 query groups of 25 pairs
    ltr = LightGBMRankerMatcher(n_estimators=10)
    ltr.fit(X, y, query_groups=groups)
    s_ltr = ltr.predict_proba(X)
    assert s_ltr.shape == (100,)
    assert (s_ltr >= 0.0).all() and (s_ltr <= 1.0).all()
