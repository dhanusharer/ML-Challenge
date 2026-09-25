"""Unit tests for full corpus feasibility gate components."""

import pytest
from collections import defaultdict
from src.representations.normalizer import standard_clean
from src.representations.name import strip_legal_suffixes, get_sorted_tokens_name, get_name_tokens
from src.representations.address import extract_numeric_tokens


def test_query_side_indexing_logic():
    queries = [
        {"entity_id": "S1-1", "business_name": "Acme Widgets Corp", "business_address": "123 Main St", "country": "US"},
        {"entity_id": "S1-2", "business_name": "Reliance Retail Ltd", "business_address": "Plot 50 Bandra", "country": "India"},
    ]

    targets = [
        {"entity_id": "S2-1", "business_name": "Acme Widgets", "business_address": "123 Main St", "country": "US"},
        {"entity_id": "S3-1", "business_name": "Widgets Acme", "business_address": "123 Main St", "country": "US"},
        {"entity_id": "S2-2", "business_name": "Reliance Retail", "business_address": "Plot 50 Bandra", "country": "India"},
        {"entity_id": "S3-3", "business_name": "Random Corp", "business_address": "999 Other St", "country": "US"},
    ]

    gt = {
        "S1-1": ["S2-1", "S3-1"],
        "S1-2": ["S2-2"],
    }

    # Query side index
    query_exact = defaultdict(list)
    query_sorted = defaultdict(list)
    query_numeric = defaultdict(list)

    for q_idx, q in enumerate(queries):
        name_clean = strip_legal_suffixes(q["business_name"])
        query_exact[name_clean].append(q_idx)
        sorted_name = get_sorted_tokens_name(q["business_name"], strip_legal=True)
        query_sorted[sorted_name].append(q_idx)
        tokens = get_name_tokens(name_clean)
        if tokens:
            nums = extract_numeric_tokens(q.get("business_address", ""))
            for num in nums:
                query_numeric[f"{tokens[0]}::{num}"].append(q_idx)

    cands = defaultdict(set)
    for t in targets:
        eid = t["entity_id"]
        t_clean = strip_legal_suffixes(t["business_name"])
        if t_clean in query_exact:
            for q_idx in query_exact[t_clean]:
                cands[queries[q_idx]["entity_id"]].add(eid)
        t_sorted = get_sorted_tokens_name(t["business_name"], strip_legal=True)
        if t_sorted in query_sorted:
            for q_idx in query_sorted[t_sorted]:
                cands[queries[q_idx]["entity_id"]].add(eid)
        t_tokens = get_name_tokens(t_clean)
        if t_tokens:
            nums = extract_numeric_tokens(t.get("business_address", ""))
            for num in nums:
                k = f"{t_tokens[0]}::{num}"
                if k in query_numeric:
                    for q_idx in query_numeric[k]:
                        cands[queries[q_idx]["entity_id"]].add(eid)

    assert "S2-1" in cands["S1-1"]
    assert "S3-1" in cands["S1-1"]
    assert "S2-2" in cands["S1-2"]
    assert "S3-3" not in cands["S1-1"]
    assert "S3-3" not in cands["S1-2"]
