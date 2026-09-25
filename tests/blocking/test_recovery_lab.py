"""Unit tests for Phase 1.7 RecoveryLab components."""

import pytest
from src.blocking.recovery_lab import (
    get_informative_name_tokens,
    extract_address_numeric_compounds,
    RecoveryLabRunner,
)


def test_get_informative_name_tokens():
    tokens = get_informative_name_tokens("Apex Enterprises Private Limited")
    assert "apex" in tokens
    assert "enterprises" not in tokens

    tokens2 = get_informative_name_tokens("Global Steel Solutions")
    assert "steel" in tokens2
    assert "solutions" not in tokens2


def test_extract_address_numeric_compounds():
    compounds = extract_address_numeric_compounds("Plot No 70C/4G, Chhota Baghara, Allahabad")
    assert any("70c/4g" in c or "70c" in c for c in compounds)

    compounds2 = extract_address_numeric_compounds("Flat At Dcm Building, 16, Barakhamba Road")
    assert "16" in compounds2


def test_recovery_lab_runner_initialization():
    runner = RecoveryLabRunner(query_cohort_size=10, seed=42)
    assert runner.query_cohort_size == 10
    assert runner.seed == 42
