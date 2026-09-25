"""Unit tests for entity-level validation split and leakage verification."""

import pytest
from src.validation.split import (
    create_entity_validation_split,
    partition_ground_truth,
    ValidationSplit,
)
from src.validation.leakage import verify_no_leakage, LeakageVerificationResult


def test_validation_split_disjointness_and_ratios():
    s1_ids = [f"S1-{i:04d}" for i in range(100)]
    split = create_entity_validation_split(s1_ids, val_fraction=0.20, seed=42)

    assert split.total_entities == 100
    assert split.train_entities == 80
    assert split.val_entities == 20

    train_set = set(split.train_s1_ids)
    val_set = set(split.val_s1_ids)

    # Assert ZERO entity overlap
    assert len(train_set & val_set) == 0
    # Assert exact union
    assert train_set | val_set == set(s1_ids)


def test_validation_split_deterministic_reproducibility():
    s1_ids = [f"S1-{i:04d}" for i in range(250)]

    split1 = create_entity_validation_split(s1_ids, val_fraction=0.20, seed=42)
    split2 = create_entity_validation_split(s1_ids, val_fraction=0.20, seed=42)

    assert split1.train_s1_ids == split2.train_s1_ids
    assert split1.val_s1_ids == split2.val_s1_ids
    assert split1.train_ids_hash == split2.train_ids_hash
    assert split1.val_ids_hash == split2.val_ids_hash


def test_validation_split_seed_variation():
    s1_ids = [f"S1-{i:04d}" for i in range(250)]

    split42 = create_entity_validation_split(s1_ids, val_fraction=0.20, seed=42)
    split99 = create_entity_validation_split(s1_ids, val_fraction=0.20, seed=99)

    assert split42.train_ids_hash != split99.train_ids_hash


def test_validation_split_rejects_duplicate_input():
    s1_ids = ["S1-001", "S1-001", "S1-002"]
    with pytest.raises(ValueError, match="duplicate IDs"):
        create_entity_validation_split(s1_ids)


def test_partition_ground_truth_and_leakage_check():
    s1_ids = [f"S1-{i}" for i in range(10)]
    gt_map = {f"S1-{i}": [f"S2-{i}"] if i % 2 == 0 else [] for i in range(10)}

    split = create_entity_validation_split(s1_ids, val_fraction=0.20, seed=42)
    train_gt, val_gt = partition_ground_truth(gt_map, split)

    assert len(train_gt) == 8
    assert len(val_gt) == 2

    # Verification passes
    res = verify_no_leakage(split.train_s1_ids, split.val_s1_ids, train_gt, val_gt)
    assert res.is_leak_free is True
    assert res.s1_entity_overlap_count == 0


def test_leakage_detection_raises_assertion_error():
    train_ids = ["S1-1", "S1-2", "S1-3"]
    val_ids = ["S1-3", "S1-4"]  # S1-3 is leaked!
    train_gt = {"S1-1": [], "S1-2": [], "S1-3": []}
    val_gt = {"S1-3": [], "S1-4": []}

    with pytest.raises(AssertionError, match="LEAKAGE DETECTED: 1 S1 entities appear in BOTH"):
        verify_no_leakage(train_ids, val_ids, train_gt, val_gt)
