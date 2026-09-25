"""Unit tests for Phase 4.1 Forensic Revalidation Gate."""

from pathlib import Path
import pytest
from src.utils.env import PROJECT_ROOT
from src.policy.decision import RelativeMarginPolicy
from src.forensics.alignment_audit import audit_candidate_alignment
from src.forensics.policy_audit import audit_entity_decision_policy


def test_relative_margin_policy_cases_1_to_6():
    """Verify RelativeMarginPolicy conforms to all 6 forensic specification cases."""
    results = audit_entity_decision_policy()
    for r in results:
        assert r["status"] == "PASSED", f"Case {r['case_id']} failed: {r}"


def test_alignment_and_id_mapping_invariants():
    """Verify S1 query isolation and ID round-trip mapping."""
    res = audit_candidate_alignment()
    assert res["isolation_test_passed"] is True
    assert res["roundtrip_tested_ids"] == 10000
    assert res["roundtrip_mismatches"] == 0
    assert res["prefix_violations"] == 0
    assert res["summary_verdict"] == "PASSED"


def test_output_tsv_pathological_collapse_metrics():
    """Verify that matching_results.tsv exhibits the measured 99.88% empty collapse."""
    tsv_path = PROJECT_ROOT / "matching_results.tsv"
    assert tsv_path.is_file(), "matching_results.tsv must exist"

    total = 0
    empty = 0
    with open(tsv_path, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\r\n").split("\t")
        assert header == ["source1_entity_id", "matched_entity_ids"]
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if not parts:
                continue
            total += 1
            if len(parts) < 2 or not parts[1].strip():
                empty += 1

    assert total == 1732544, f"Expected 1,732,544 rows, found {total}"
    empty_pct = empty / total
    assert empty_pct > 0.998, f"Expected >99.8% empty rate, found {empty_pct:.4f}"


def test_singleton_rate_explains_0_057_score():
    """Verify the mathematical proof: singleton rate (~5.7%) equals 0.057 leaderboard score."""
    from src.data.parser import parse_ground_truth_file
    from src.utils.env import resolve_train_dir

    train_dir = resolve_train_dir()
    gt = parse_ground_truth_file(train_dir / "train_ground_truth.tsv")

    # In train ground truth, singletons are S1s with 0 links (len == 0)
    total = len(gt)
    singletons = sum(1 for m in gt.values() if len(m) == 0)
    singleton_rate = singletons / total

    # Expected singleton rate is ~5.58% (0.0558 ~ 0.057)
    assert 0.050 <= singleton_rate <= 0.065, f"Singleton rate {singleton_rate:.4f} outside [0.05, 0.065]"
    # Macro F0.5 of an all-empty prediction = singleton_rate * 1.0 + (1 - singleton_rate) * 0.0 = singleton_rate
    theoretical_f05 = singleton_rate
    assert abs(theoretical_f05 - 0.057) < 0.005, f"Theoretical score {theoretical_f05:.4f} differs from 0.057"
