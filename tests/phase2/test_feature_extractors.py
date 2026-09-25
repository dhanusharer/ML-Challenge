"""Unit tests for Phase 2 feature extractors and schema."""

import pytest
import numpy as np

from src.features.schema import FEATURE_REGISTRY, get_feature_names
from src.features.extractor import PairwiseFeatureExtractor, compute_levenshtein_ratio, compute_ngram_cosine


def test_feature_registry_completeness():
    names = get_feature_names()
    assert len(names) >= 35
    assert "name_exact_clean" in names
    assert "addr_numeric_exact_match" in names
    assert "country_exact_match" in names
    assert "retrieval_support_count" in names
    assert "inter_name_x_addr_jaccard" in names


def test_levenshtein_ratio():
    assert compute_levenshtein_ratio("Apex", "Apex") == 1.0
    assert compute_levenshtein_ratio("Apex", "Ape") > 0.70
    assert compute_levenshtein_ratio("Apex", "Zyzz") < 0.30
    assert compute_levenshtein_ratio("", "") == 1.0


def test_ngram_cosine():
    assert compute_ngram_cosine("Reliance Industries", "Reliance Industries") == 1.0
    assert compute_ngram_cosine("Reliance", "Reliance Ltd") > 0.60
    assert compute_ngram_cosine("Apple", "Zebra") == 0.0


def test_feature_extractor_single_pair():
    extractor = PairwiseFeatureExtractor()
    q = {
        "entity_id": "S1-100",
        "business_name": "Apex Solutions Private Limited",
        "business_address": "123 Main Street, Suite 400, Chicago, IL 60601",
        "country": "US",
    }
    t = {
        "entity_id": "S2-200",
        "business_name": "Apex Solutions Corp",
        "business_address": "123 Main St, Chicago, IL",
        "country": "US",
    }
    prov = {
        "exact_name": True,
        "sorted_name": True,
        "name_numeric": True,
        "tfidf": True,
        "tfidf_score": 0.85,
        "tfidf_rank": 1,
        "support_count": 4,
    }

    feats = extractor.extract_pair(q, t, prov)
    assert len(feats) == len(extractor.feature_names)
    assert feats["name_exact_clean"] == 1.0
    assert feats["country_exact_match"] == 1.0
    assert feats["source_is_s2"] == 1.0
    assert feats["retrieval_support_count"] == 4.0
    assert feats["tfidf_similarity_score"] == 0.85
    assert not np.isnan(list(feats.values())).any()


def test_feature_extractor_matrix_shape_and_no_nan():
    extractor = PairwiseFeatureExtractor()
    pairs = [
        ({"business_name": "Acme Inc", "business_address": "100 1st Ave", "country": "US", "entity_id": "S1-1"},
         {"business_name": "Acme", "business_address": "100 1st Avenue", "country": "US", "entity_id": "S2-1"},
         {"support_count": 2}),
        ({"business_name": "Beta LLC", "business_address": None, "country": "FRANCE", "entity_id": "S1-2"},
         {"business_name": "Beta SARL", "business_address": "Paris", "country": "FRANCE", "entity_id": "S3-2"},
         {"support_count": 1}),
    ]
    X = extractor.extract_matrix(pairs)
    assert X.shape == (2, len(extractor.feature_names))
    assert not np.isnan(X).any()
    assert not np.isinf(X).any()
