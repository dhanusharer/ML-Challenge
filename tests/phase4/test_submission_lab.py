"""Unit tests for Phase 4 Leaderboard Optimization & Submission Lab."""

from pathlib import Path
import tempfile
import pytest

from src.submission.analyzer import SubmissionDistributionAnalyzer
from src.submission.registry import SubmissionRegistryManager
from src.policy.decision import RelativeMarginPolicy


def test_submission_distribution_analyzer():
    """Verify analyzer correctly parses matching TSV and computes metrics."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tsv_path = Path(tmpdir) / "test_matching.tsv"
        with open(tsv_path, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tmatched_entity_ids\n")
            f.write("S1-1\tS2-100,S3-200\n")  # 2 matches
            f.write("S1-2\tS2-300\n")          # 1 match
            f.write("S1-3\t\n")                # 0 matches (empty)

        stats = SubmissionDistributionAnalyzer.analyze_submission_file(tsv_path)
        assert stats["total_s1_entities"] == 3
        assert stats["empty_prediction_count"] == 1
        assert stats["one_match_count"] == 1
        assert stats["two_match_count"] == 1
        assert stats["total_predicted_matches"] == 3
        assert stats["s2_target_count"] == 2
        assert stats["s3_target_count"] == 1
        assert stats["invalid_prefix_count"] == 0
        assert "NORMAL" in stats["distribution_shift_verdict"]


def test_submission_registry_manager():
    """Verify registry manager writes valid CSV rows and computes SHA256."""
    with tempfile.TemporaryDirectory() as tmpdir:
        reg_mgr = SubmissionRegistryManager(registry_dir=Path(tmpdir))

        dummy_file = Path(tmpdir) / "sample.tsv"
        dummy_file.write_text("dummy content", encoding="utf-8")

        sha256 = reg_mgr.compute_sha256(dummy_file)
        assert len(sha256) == 64

        reg_mgr.register_submission({
            "submission_id": "test_v00",
            "timestamp": "2026-09-25 12:00:00 UTC",
            "day": 1,
            "hypothesis_id": "H0",
            "candidate_arm": "ARM_B",
            "matcher": "LightGBM",
            "threshold_floor": 0.75,
            "multi_threshold": 0.80,
            "margin": 0.08,
            "max_k": 3,
            "rescue_trigger": "test",
            "variant_description": "Baseline",
            "local_dev_f05": 0.9422,
            "local_holdout_f05": 0.9122,
            "public_score": "UNKNOWN",
            "file_sha256": sha256,
            "validator_status": "PASSED",
            "decision": "KEEP",
            "notes": "Test note",
        })

        assert reg_mgr.registry_csv.is_file()
        content = reg_mgr.registry_csv.read_text(encoding="utf-8")
        assert "test_v00" in content
        assert "0.9422" in content


def test_rule_5_target_id_invariance():
    """Verify set selection enforces Rule 5 (only S2/S3 IDs) and deduplication."""
    policy = RelativeMarginPolicy(threshold_floor=0.70, multi_threshold=0.75, max_margin=0.10, max_k=3)
    cands = [
        {"target_id": "S1-999", "score": 0.99},   # S1 ID (MUST BE REJECTED)
        {"target_id": "S2-100", "score": 0.95},   # Valid S2
        {"target_id": "S2-100", "score": 0.92},   # Duplicate S2
        {"target_id": "S3-200", "score": 0.90},   # Valid S3
        {"target_id": "UNKNOWN-5", "score": 0.85},# Invalid prefix
    ]
    raw_preds = policy.predict_matches("dummy_s1", cands)
    # Apply filtering logic from SubmissionGenerator
    clean_preds = []
    seen = set()
    for pid in raw_preds:
        if (pid.startswith("S2-") or pid.startswith("S3-")) and pid not in seen:
            clean_preds.append(pid)
            seen.add(pid)

    assert "S1-999" not in clean_preds
    assert "UNKNOWN-5" not in clean_preds
    assert clean_preds == ["S2-100", "S3-200"]
    assert len(clean_preds) == len(set(clean_preds))
