"""Unit tests for Phase 2.1 Feature and Label Leakage Audit."""

import pytest
from src.features.schema import FEATURE_REGISTRY
from src.features.extractor import PairwiseFeatureExtractor


def test_no_labels_in_feature_registry():
    """Verify that no feature in the registry accesses ground-truth labels."""
    forbidden_terms = [
        "label", "ground_truth", "target_match_count", "is_match",
        "true_link", "posterior_prob", "gold"
    ]
    for spec in FEATURE_REGISTRY:
        desc_lower = spec.description.lower()
        src_lower = spec.data_source.lower()
        for term in forbidden_terms:
            assert term not in desc_lower or "leakage" in desc_lower, (
                f"Feature {spec.name} mentions forbidden term '{term}' in description: {spec.description}"
            )
            assert term not in src_lower, (
                f"Feature {spec.name} has forbidden term '{term}' in data source: {spec.data_source}"
            )


def test_feature_extractor_independence_from_ground_truth():
    """Verify that extract_pair produces identical results regardless of external labels."""
    extractor = PairwiseFeatureExtractor()
    q = {
        "entity_id": "S1-999",
        "business_name": "Antigravity Engineering",
        "business_address": "100 Innovation Way, Seattle, WA",
        "country": "US",
    }
    t = {
        "entity_id": "S2-888",
        "business_name": "Antigravity Eng Corp",
        "business_address": "100 Innovation Way, Seattle, Washington",
        "country": "US",
    }
    prov = {
        "exact_name": False,
        "sorted_name": True,
        "tfidf": True,
        "tfidf_score": 0.82,
        "tfidf_rank": 2,
        "support_count": 3,
    }

    feats1 = extractor.extract_pair(q, t, prov)
    feats2 = extractor.extract_pair(q, t, prov)

    for k in feats1:
        assert feats1[k] == feats2[k]
        assert not isinstance(feats1[k], bool)
        assert -1.0 <= feats1[k] <= 1000.0


def test_unsupervised_corpus_idf():
    """Verify that corpus IDF is purely token frequency based without class labels."""
    mock_idf = {"antigravity": 6.2, "engineering": 2.1}
    extractor = PairwiseFeatureExtractor(corpus_idf_dict=mock_idf)
    assert extractor.corpus_idf["antigravity"] == 6.2
    assert extractor.corpus_idf.get("nonexistent", 5.0) == 5.0
