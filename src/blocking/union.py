"""Block Complementarity and Progressive Union Analysis.

Analyzes marginal utility and trade-offs of combining blocking mechanisms:
1. Pairwise Complementarity:
   - A only (true links retrieved strictly by A)
   - B only (true links retrieved strictly by B)
   - A ∩ B (true links retrieved by both)
   - A ∪ B (combined true links)
   - Missed by both
2. Progressive Union Experiments:
   - Total recall
   - Incremental recall
   - Candidate volume
   - Incremental candidate cost
   - Candidate size distributions
"""

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import numpy as np


def merge_candidate_dicts(
    candidate_dicts: Sequence[Dict[str, Set[str]]],
) -> Dict[str, Set[str]]:
    """Merge multiple candidate sets per entity via set union."""
    merged: Dict[str, Set[str]] = {}
    if not candidate_dicts:
        return merged

    all_keys = set().union(*[c.keys() for c in candidate_dicts])
    for k in all_keys:
        union_set: Set[str] = set()
        for cd in candidate_dicts:
            if k in cd:
                union_set.update(cd[k])
        merged[k] = union_set
    return merged


def analyze_complementarity(
    block_a_name: str,
    cands_a: Dict[str, Set[str]],
    block_b_name: str,
    cands_b: Dict[str, Set[str]],
    ground_truth: Dict[str, Sequence[str]],
) -> Dict[str, Any]:
    """Analyze overlap and unique contributions between two candidate sets."""
    total_true_links = 0
    a_only_links = 0
    b_only_links = 0
    both_links = 0
    missed_both = 0

    for s1_id, true_list in ground_truth.items():
        true_set = set(true_list)
        if not true_set:
            continue

        a_set = cands_a.get(s1_id, set())
        b_set = cands_b.get(s1_id, set())

        for mid in true_set:
            total_true_links += 1
            in_a = mid in a_set
            in_b = mid in b_set

            if in_a and in_b:
                both_links += 1
            elif in_a and not in_b:
                a_only_links += 1
            elif in_b and not in_a:
                b_only_links += 1
            else:
                missed_both += 1

    union_cands = merge_candidate_dicts([cands_a, cands_b])
    total_cands_a = sum(len(v) for v in cands_a.values())
    total_cands_b = sum(len(v) for v in cands_b.values())
    total_cands_union = sum(len(v) for v in union_cands.values())

    return {
        "block_a": block_a_name,
        "block_b": block_b_name,
        "total_true_links": total_true_links,
        "a_only_true_links": a_only_links,
        "a_only_recall": a_only_links / total_true_links if total_true_links > 0 else 0.0,
        "b_only_true_links": b_only_links,
        "b_only_recall": b_only_links / total_true_links if total_true_links > 0 else 0.0,
        "intersection_true_links": both_links,
        "intersection_recall": both_links / total_true_links if total_true_links > 0 else 0.0,
        "union_true_links": both_links + a_only_links + b_only_links,
        "union_recall": (both_links + a_only_links + b_only_links) / total_true_links if total_true_links > 0 else 0.0,
        "missed_by_both": missed_both,
        "total_candidates_a": total_cands_a,
        "total_candidates_b": total_cands_b,
        "total_candidates_union": total_cands_union,
        "candidate_overlap_percentage": round(100.0 * (total_cands_a + total_cands_b - total_cands_union) / total_cands_union, 2) if total_cands_union > 0 else 0.0,
    }


def evaluate_union(
    union_name: str,
    candidate_dicts: Sequence[Dict[str, Set[str]]],
    queries: Sequence[Dict[str, str]],
    ground_truth: Dict[str, Sequence[str]],
    total_target_records: int,
) -> Dict[str, Any]:
    """Evaluate a merged candidate set across all queries at true link level."""
    merged = merge_candidate_dicts(candidate_dicts)

    total_true = 0
    captured_true = 0
    s2_true = 0
    s2_captured = 0
    s3_true = 0
    s3_captured = 0

    candidate_counts: List[int] = []

    for q in queries:
        s1_id = q["entity_id"]
        true_mids = set(ground_truth.get(s1_id, []))
        pred_cands = merged.get(s1_id, set())

        candidate_counts.append(len(pred_cands))

        for mid in true_mids:
            total_true += 1
            is_captured = mid in pred_cands
            if is_captured:
                captured_true += 1

            if mid.startswith("S2-"):
                s2_true += 1
                if is_captured:
                    s2_captured += 1
            elif mid.startswith("S3-"):
                s3_true += 1
                if is_captured:
                    s3_captured += 1

    total_s1 = len(queries)
    counts_arr = np.array(candidate_counts) if candidate_counts else np.array([0])
    total_cands = int(np.sum(counts_arr))
    cartesian_space = total_s1 * total_target_records if total_target_records > 0 else 1

    return {
        "union_name": union_name,
        "total_s1_evaluated": total_s1,
        "total_true_links": total_true,
        "captured_true_links": captured_true,
        "blocking_recall": round(captured_true / total_true, 6) if total_true > 0 else 0.0,
        "s2_recall": round(s2_captured / s2_true, 6) if s2_true > 0 else 0.0,
        "s3_recall": round(s3_captured / s3_true, 6) if s3_true > 0 else 0.0,
        "total_candidates": total_cands,
        "mean_candidates_per_s1": round(float(np.mean(counts_arr)), 2),
        "median_candidates_per_s1": float(np.median(counts_arr)),
        "p90_candidates_per_s1": float(np.percentile(counts_arr, 90)),
        "p95_candidates_per_s1": float(np.percentile(counts_arr, 95)),
        "p99_candidates_per_s1": float(np.percentile(counts_arr, 99)),
        "max_candidates_per_s1": int(np.max(counts_arr)),
        "reduction_ratio": round(1.0 - (total_cands / cartesian_space), 8) if cartesian_space > 0 else 1.0,
    }
