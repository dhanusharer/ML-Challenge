"""Unit tests for block union and complementarity analyzer."""

import pytest
from src.blocking.union import analyze_complementarity, evaluate_union, merge_candidate_dicts


def test_merge_candidate_dicts():
    d1 = {"S1-1": {"S2-1"}, "S1-2": set()}
    d2 = {"S1-1": {"S3-1"}, "S1-2": {"S2-2"}}
    merged = merge_candidate_dicts([d1, d2])
    assert merged["S1-1"] == {"S2-1", "S3-1"}
    assert merged["S1-2"] == {"S2-2"}


def test_analyze_complementarity():
    cands_a = {"S1-1": {"S2-1"}, "S1-2": {"S2-2"}}
    cands_b = {"S1-1": {"S2-1", "S3-1"}, "S1-2": set()}
    gt = {
        "S1-1": ["S2-1", "S3-1"],
        "S1-2": ["S2-2"],
    }
    # Total true links = 3: (S1-1->S2-1, S1-1->S3-1, S1-2->S2-2)
    # A finds: S1-1->S2-1, S1-2->S2-2 (2 links)
    # B finds: S1-1->S2-1, S1-1->S3-1 (2 links)
    # A only: S1-2->S2-2 (1 link)
    # B only: S1-1->S3-1 (1 link)
    # Both: S1-1->S2-1 (1 link)
    # Union finds: all 3 links
    comp = analyze_complementarity("BlockA", cands_a, "BlockB", cands_b, gt)

    assert comp["total_true_links"] == 3
    assert comp["a_only_true_links"] == 1
    assert comp["b_only_true_links"] == 1
    assert comp["intersection_true_links"] == 1
    assert comp["union_true_links"] == 3
    assert comp["missed_by_both"] == 0
    assert comp["union_recall"] == 1.0


def test_evaluate_union():
    queries = [{"entity_id": "S1-1"}, {"entity_id": "S1-2"}]
    cands_a = {"S1-1": {"S2-1"}, "S1-2": set()}
    cands_b = {"S1-1": {"S3-1"}, "S1-2": {"S2-2"}}
    gt = {"S1-1": ["S2-1", "S3-1"], "S1-2": ["S2-2"]}

    res = evaluate_union("Union-AB", [cands_a, cands_b], queries, gt, total_target_records=100)
    assert res["blocking_recall"] == 1.0
    assert res["total_candidates"] == 3
    assert res["reduction_ratio"] > 0.95
