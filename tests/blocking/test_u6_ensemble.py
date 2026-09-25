"""Tests for integrated U6 ensemble blocker."""

import pytest
from src.blocking.u6_ensemble import U6EnsembleBlocker


def test_u6_ensemble_deterministic_retrieval():
    targets = [
        {"entity_id": "S2-1", "business_name": "Apex Solutions Inc", "business_address": "100 Main St, Austin, TX 78701", "country": "US"},
        {"entity_id": "S3-1", "business_name": "Solutions Apex", "business_address": "100 Main Street, Austin, Texas", "country": "US"},
        {"entity_id": "S2-2", "business_name": "Reliance Retail Ltd", "business_address": "Plot 45, Bandra Kurla Complex, Mumbai", "country": "India"},
        {"entity_id": "S3-2", "business_name": "SARL Boulangerie Moderne", "business_address": "12 Rue de la Paix, Paris 75001", "country": "France"},
        {"entity_id": "S2-3", "business_name": "Distractor Corp", "business_address": "999 Random Blvd", "country": "US"},
    ]

    queries = [
        {"entity_id": "S1-1", "business_name": "Apex Solutions", "business_address": "100 Main St, Austin, TX", "country": "US"},
        {"entity_id": "S1-2", "business_name": "Reliance Retail Pvt Ltd", "business_address": "Plot 45, BKC, Mumbai", "country": "India"},
        {"entity_id": "S1-3", "business_name": "Boulangerie Moderne", "business_address": "12 Rue de la Paix, Paris", "country": "France"},
    ]

    gt = {
        "S1-1": ["S2-1", "S3-1"],
        "S1-2": ["S2-2"],
        "S1-3": ["S3-2"],
    }

    blocker = U6EnsembleBlocker(tfidf_top_k=2)
    blocker.fit(targets)
    metrics = blocker.evaluate_cohort(queries, gt, batch_size=2)

    assert metrics["blocking_recall"] == 1.0
    assert metrics["s2_recall"] == 1.0
    assert metrics["s3_recall"] == 1.0
    assert metrics["mean_candidates"] <= 4.0
    assert metrics["total_queries"] == 3


def test_u6_ensemble_handles_empty_records():
    targets = [
        {"entity_id": "S2-10", "business_name": "", "business_address": "", "country": ""},
        {"entity_id": "S3-10", "business_name": "Alpha Corp", "business_address": "500 Lake Ave", "country": "US"},
    ]
    queries = [
        {"entity_id": "S1-10", "business_name": "", "business_address": "", "country": ""},
        {"entity_id": "S1-11", "business_name": "Alpha", "business_address": "500 Lake Ave", "country": "US"},
    ]
    gt = {
        "S1-10": ["S2-10"],
        "S1-11": ["S3-10"],
    }

    blocker = U6EnsembleBlocker(tfidf_top_k=2)
    blocker.fit(targets)
    metrics = blocker.evaluate_cohort(queries, gt)
    assert metrics["total_queries"] == 2
