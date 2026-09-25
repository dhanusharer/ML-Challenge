"""Unit tests for deterministic ground truth parser."""

import pytest
from pathlib import Path

from src.data.parser import (
    parse_ground_truth_row,
    parse_ground_truth_file,
    GroundTruthParseError,
)


def test_parse_ground_truth_row_singleton():
    res = parse_ground_truth_row("S1-001", "")
    assert res == {
        "source1_entity_id": "S1-001",
        "matched_entity_ids": [],
    }


def test_parse_ground_truth_row_whitespace_singleton():
    res = parse_ground_truth_row("S1-001", "   ")
    assert res == {
        "source1_entity_id": "S1-001",
        "matched_entity_ids": [],
    }


def test_parse_ground_truth_row_multi_matches():
    res = parse_ground_truth_row("S1-001", "S2-001, S3-002 , S2-003")
    assert res["source1_entity_id"] == "S1-001"
    assert res["matched_entity_ids"] == ["S2-001", "S3-002", "S2-003"]


def test_parse_ground_truth_row_rejects_empty_token():
    with pytest.raises(GroundTruthParseError, match="Malformed empty ID token"):
        parse_ground_truth_row("S1-001", "S2-001,,S3-002")


def test_parse_ground_truth_row_rejects_self_match():
    with pytest.raises(GroundTruthParseError, match="Forbidden self-match"):
        parse_ground_truth_row("S1-001", "S1-002,S2-001")


def test_parse_ground_truth_row_rejects_invalid_prefix():
    with pytest.raises(GroundTruthParseError, match="Invalid target prefix"):
        parse_ground_truth_row("S1-001", "S4-001")


def test_parse_ground_truth_row_rejects_intra_list_duplicate():
    with pytest.raises(GroundTruthParseError, match="Duplicate matched ID"):
        parse_ground_truth_row("S1-001", "S2-001,S3-002,S2-001")


def test_parse_ground_truth_row_rejects_invalid_s1():
    with pytest.raises(GroundTruthParseError, match="does not start with expected prefix 'S1-'"):
        parse_ground_truth_row("X1-001", "S2-001")


def test_parse_ground_truth_file_full(tmp_path: Path):
    f = tmp_path / "ground_truth.tsv"
    content = (
        "source1_entity_id\tmatched_entity_ids\n"
        "S1-1\tS2-1,S3-1\n"
        "S1-2\t\n"
        "S1-3\tS2-2\n"
    )
    f.write_text(content, encoding="utf-8")
    gt_map = parse_ground_truth_file(f)
    assert len(gt_map) == 3
    assert gt_map["S1-1"] == ["S2-1", "S3-1"]
    assert gt_map["S1-2"] == []
    assert gt_map["S1-3"] == ["S2-2"]


def test_parse_ground_truth_file_duplicate_s1(tmp_path: Path):
    f = tmp_path / "dup_gt.tsv"
    content = (
        "source1_entity_id\tmatched_entity_ids\n"
        "S1-1\tS2-1\n"
        "S1-1\tS2-2\n"
    )
    f.write_text(content, encoding="utf-8")
    with pytest.raises(GroundTruthParseError, match="Duplicate source1_entity_id"):
        parse_ground_truth_file(f)
