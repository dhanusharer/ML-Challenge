"""Unit tests for blocking strategies and candidate generators."""

import pytest
from src.blocking.exact_name import ExactNameBlocker
from src.blocking.token_buckets import SortedTokenNameBlocker, RareTokenNameBlocker
from src.blocking.char_ngrams import CharNgramBoundaryBlocker
from src.blocking.address_blocks import ExactAddressBlocker, PostalCodeBlocker
from src.blocking.hybrids import NamePostalHybridBlocker
from src.blocking.sparse_tfidf import SparseTfidfBlocker


@pytest.fixture
def sample_dataset():
    targets = [
        {"entity_id": "S2-001", "business_name": "Orelee Barbershop Inc", "business_address": "1795 Westchester Dr 27262", "country": "US"},
        {"entity_id": "S2-002", "business_name": "Tata Motors Private Limited", "business_address": "Bombay House 400001", "country": "India"},
        {"entity_id": "S3-001", "business_name": "Barbershop Orelee", "business_address": "Westchester Dr 27262", "country": "US"},
        {"entity_id": "S3-002", "business_name": "Boulangerie Paris SARL", "business_address": "63 Rue De Dieppe 59000", "country": "France"},
        {"entity_id": "S3-003", "business_name": "Unknown Corp", "business_address": "", "country": "US"},  # missing address
    ]
    queries = [
        {"entity_id": "S1-101", "business_name": "Orelee's Barbershop", "business_address": "1795 Westchester Drive, High Point 27262", "country": "US"},
        {"entity_id": "S1-102", "business_name": "Tata Motors Ltd", "business_address": "Mumbai 400001", "country": "India"},
        {"entity_id": "S1-103", "business_name": "Boulangerie Paris", "business_address": "Lille 59000", "country": "France"},
        {"entity_id": "S1-104", "business_name": "Singleton Shop", "business_address": "100 Main St", "country": "US"},
    ]
    gt = {
        "S1-101": ["S2-001", "S3-001"],
        "S1-102": ["S2-002"],
        "S1-103": ["S3-002"],
        "S1-104": [],  # true singleton
    }
    return targets, queries, gt


def test_exact_name_blocker(sample_dataset):
    targets, queries, gt = sample_dataset
    blocker = ExactNameBlocker(strip_legal=True)
    blocker.fit(targets)
    metrics, cands = blocker.evaluate(queries, gt, total_target_records=len(targets))

    assert cands["S1-102"] == {"S2-002"}
    assert cands["S1-103"] == {"S3-002"}
    assert cands["S1-104"] == set()
    assert metrics.captured_true_links >= 2
    assert metrics.blocking_recall > 0.0


def test_sorted_token_blocker(sample_dataset):
    targets, queries, gt = sample_dataset
    blocker = SortedTokenNameBlocker(strip_legal=True)
    blocker.fit(targets)
    metrics, cands = blocker.evaluate(queries, gt, total_target_records=len(targets))

    # S1-101 ("Orelee's Barbershop") matches both S2-001 ("Orelee Barbershop") and S3-001 ("Barbershop Orelee")
    assert "S3-001" in cands["S1-101"]


def test_char_ngram_boundary_blocker(sample_dataset):
    targets, queries, gt = sample_dataset
    blocker = CharNgramBoundaryBlocker(n=3)
    blocker.fit(targets)
    metrics, cands = blocker.evaluate(queries, gt, total_target_records=len(targets))
    assert metrics.total_candidates > 0


def test_postal_code_blocker(sample_dataset):
    targets, queries, gt = sample_dataset
    blocker = PostalCodeBlocker()
    blocker.fit(targets)
    metrics, cands = blocker.evaluate(queries, gt, total_target_records=len(targets))

    # 27262 should retrieve S2-001 and S3-001
    assert "S2-001" in cands["S1-101"]
    assert "S3-001" in cands["S1-101"]
    # Missing address record S3-003 causes no crash
    assert "S3-003" not in cands["S1-104"]


def test_sparse_tfidf_blocker(sample_dataset):
    targets, queries, gt = sample_dataset
    blocker = SparseTfidfBlocker(top_k=2, min_df=1)
    blocker.fit(targets)
    metrics, cands = blocker.evaluate(queries, gt, total_target_records=len(targets))

    for s1_id, cand_set in cands.items():
        assert len(cand_set) <= 2
    assert metrics.total_candidates <= len(queries) * 2
