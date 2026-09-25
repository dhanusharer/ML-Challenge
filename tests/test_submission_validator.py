"""Unit tests for submission file formatting and schema validator."""

import pytest
from pathlib import Path

from src.evaluation.validator import validate_results_tsv
from src.utils.env import EXPECTED_SUBMISSION_COLUMNS


def test_validate_results_tsv_valid(tmp_path: Path):
    f = tmp_path / "matching_results.tsv"
    content = (
        "source1_entity_id\tmatched_entity_ids\n"
        "S1-1\tS2-1,S3-1\n"
        "S1-2\t\n"
        "S1-3\tS2-2\n"
    )
    f.write_text(content, encoding="utf-8")
    mapping, errors, warnings = validate_results_tsv(
        f,
        expected_header=EXPECTED_SUBMISSION_COLUMNS,
        col_label="matched_entity_ids",
        required_s1_ids={"S1-1", "S1-2", "S1-3"},
    )
    assert len(errors) == 0
    assert len(mapping) == 3
    assert mapping["S1-1"] == {"S2-1", "S3-1"}
    assert mapping["S1-2"] == set()


def test_validate_results_tsv_comma_separated(tmp_path: Path):
    f = tmp_path / "matching_results.tsv"
    content = (
        "source1_entity_id,matched_entity_ids\n"
        "S1-1,S2-1\n"
    )
    f.write_text(content, encoding="utf-8")
    _, errors, _ = validate_results_tsv(
        f,
        expected_header=EXPECTED_SUBMISSION_COLUMNS,
        col_label="matched_entity_ids",
    )
    assert any("Header has no TAB but contains commas" in e for e in errors)


def test_validate_results_tsv_duplicate_s1(tmp_path: Path):
    f = tmp_path / "matching_results.tsv"
    content = (
        "source1_entity_id\tmatched_entity_ids\n"
        "S1-1\tS2-1\n"
        "S1-1\tS2-2\n"
    )
    f.write_text(content, encoding="utf-8")
    _, errors, _ = validate_results_tsv(
        f,
        expected_header=EXPECTED_SUBMISSION_COLUMNS,
        col_label="matched_entity_ids",
    )
    assert any("Duplicate source1_entity_id rows" in e for e in errors)


def test_validate_results_tsv_repeated_id_in_list(tmp_path: Path):
    f = tmp_path / "matching_results.tsv"
    content = (
        "source1_entity_id\tmatched_entity_ids\n"
        "S1-1\tS2-1,S3-1,S2-1\n"
    )
    f.write_text(content, encoding="utf-8")
    _, errors, _ = validate_results_tsv(
        f,
        expected_header=EXPECTED_SUBMISSION_COLUMNS,
        col_label="matched_entity_ids",
    )
    assert any("Repeated IDs inside a matched_entity_ids list" in e for e in errors)


def test_validate_results_tsv_self_matches(tmp_path: Path):
    f = tmp_path / "matching_results.tsv"
    content = (
        "source1_entity_id\tmatched_entity_ids\n"
        "S1-1\tS1-2,S2-1\n"
    )
    f.write_text(content, encoding="utf-8")
    _, errors, _ = validate_results_tsv(
        f,
        expected_header=EXPECTED_SUBMISSION_COLUMNS,
        col_label="matched_entity_ids",
    )
    assert any("Self-matches" in e for e in errors)


def test_validate_results_tsv_missing_required_s1(tmp_path: Path):
    f = tmp_path / "matching_results.tsv"
    content = (
        "source1_entity_id\tmatched_entity_ids\n"
        "S1-1\tS2-1\n"
    )
    f.write_text(content, encoding="utf-8")
    _, errors, _ = validate_results_tsv(
        f,
        expected_header=EXPECTED_SUBMISSION_COLUMNS,
        col_label="matched_entity_ids",
        required_s1_ids={"S1-1", "S1-2"},
    )
    assert any("Required S1 entities missing" in e for e in errors)
