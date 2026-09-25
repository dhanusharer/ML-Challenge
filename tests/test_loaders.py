"""Unit tests for TSV data loaders and schema validator."""

import pytest
import pandas as pd
from pathlib import Path

from src.data.loader import (
    load_source_tsv,
    load_ground_truth_tsv,
    validate_tsv_header,
    iter_source_tsv_chunks,
    TSVSchemaValidationError,
)
from src.utils.env import EXPECTED_SOURCE_COLUMNS, EXPECTED_GROUND_TRUTH_COLUMNS


def test_validate_tsv_header_valid(tmp_path: Path):
    tsv = tmp_path / "valid.tsv"
    tsv.write_text("entity_id\tbusiness_name\tbusiness_address\tcountry\n", encoding="utf-8")
    cols = validate_tsv_header(tsv, EXPECTED_SOURCE_COLUMNS)
    assert cols == EXPECTED_SOURCE_COLUMNS


def test_validate_tsv_header_csv_error(tmp_path: Path):
    csv = tmp_path / "accident.csv"
    csv.write_text("entity_id,business_name,business_address,country\n", encoding="utf-8")
    with pytest.raises(TSVSchemaValidationError, match="Header contains commas but NO tab delimiter"):
        validate_tsv_header(csv, EXPECTED_SOURCE_COLUMNS)


def test_validate_tsv_header_single_column_error(tmp_path: Path):
    bad = tmp_path / "single_col.tsv"
    bad.write_text("just_one_header\n", encoding="utf-8")
    with pytest.raises(TSVSchemaValidationError, match="Loaded as a single column"):
        validate_tsv_header(bad, EXPECTED_SOURCE_COLUMNS)


def test_validate_tsv_header_column_mismatch(tmp_path: Path):
    bad = tmp_path / "wrong_cols.tsv"
    bad.write_text("id\tname\taddress\tcountry\n", encoding="utf-8")
    with pytest.raises(TSVSchemaValidationError, match="Schema mismatch"):
        validate_tsv_header(bad, EXPECTED_SOURCE_COLUMNS)


def test_load_source_tsv_valid(tmp_path: Path):
    f = tmp_path / "source1.tsv"
    content = (
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S1-001\tAcme Corp\t123 Main St\tUS\n"
        "S1-002\tGlobex Inc\t456 Market St\tIndia\n"
    )
    f.write_text(content, encoding="utf-8")
    df = load_source_tsv(f, expected_prefix="S1-")
    assert len(df) == 2
    assert list(df["entity_id"]) == ["S1-001", "S1-002"]


def test_load_source_tsv_empty_entity_id(tmp_path: Path):
    f = tmp_path / "empty_id.tsv"
    content = (
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "   \tAcme Corp\t123 Main St\tUS\n"
    )
    f.write_text(content, encoding="utf-8")
    with pytest.raises(TSVSchemaValidationError, match="empty or whitespace-only entity_id"):
        load_source_tsv(f, expected_prefix="S1-")


def test_load_source_tsv_duplicate_entity_id(tmp_path: Path):
    f = tmp_path / "dup_id.tsv"
    content = (
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S1-001\tAcme Corp\t123 Main St\tUS\n"
        "S1-001\tAcme Second\t456 Main St\tUS\n"
    )
    f.write_text(content, encoding="utf-8")
    with pytest.raises(TSVSchemaValidationError, match="duplicate entity_id rows"):
        load_source_tsv(f, expected_prefix="S1-")


def test_load_source_tsv_wrong_prefix(tmp_path: Path):
    f = tmp_path / "wrong_prefix.tsv"
    content = (
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S2-001\tAcme Corp\t123 Main St\tUS\n"
    )
    f.write_text(content, encoding="utf-8")
    with pytest.raises(TSVSchemaValidationError, match="not starting with 'S1-'"):
        load_source_tsv(f, expected_prefix="S1-")


def test_iter_source_tsv_chunks(tmp_path: Path):
    f = tmp_path / "chunks.tsv"
    lines = ["entity_id\tbusiness_name\tbusiness_address\tcountry\n"]
    for i in range(10):
        lines.append(f"S1-{i:03d}\tBiz {i}\tAddress {i}\tUS\n")
    f.write_text("".join(lines), encoding="utf-8")

    chunks = list(iter_source_tsv_chunks(f, chunksize=3, expected_prefix="S1-"))
    assert len(chunks) == 4
    total_loaded = sum(len(c) for c in chunks)
    assert total_loaded == 10


def test_load_ground_truth_tsv_valid(tmp_path: Path):
    f = tmp_path / "train_ground_truth.tsv"
    content = (
        "source1_entity_id\tmatched_entity_ids\n"
        "S1-001\tS2-001,S3-001\n"
        "S1-002\t\n"
    )
    f.write_text(content, encoding="utf-8")
    df = load_ground_truth_tsv(f)
    assert len(df) == 2
    assert df.loc[0, "source1_entity_id"] == "S1-001"
    assert df.loc[1, "matched_entity_ids"] == ""
