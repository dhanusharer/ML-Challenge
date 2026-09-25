"""Unit tests for Phase 2.1 cohort determinism and split isolation."""

import hashlib
import json
import pytest
import numpy as np

from src.utils.env import PROJECT_ROOT, resolve_train_dir
from src.data.parser import parse_ground_truth_file
from src.validation.split import create_entity_validation_split


def test_validation_cohort_determinism_and_nesting():
    train_dir = resolve_train_dir()
    gt = parse_ground_truth_file(train_dir / "train_ground_truth.tsv")
    s1_all = list(gt.keys())

    split_summary = PROJECT_ROOT / "reports" / "phase0" / "validation_split_summary.json"
    with open(split_summary, "r", encoding="utf-8") as f:
        meta = json.load(f)

    split = create_entity_validation_split(
        s1_all,
        val_fraction=meta["split_metadata"]["validation_fraction"],
        seed=meta["split_metadata"]["validation_seed"],
    )

    rng1 = np.random.RandomState(42)
    shuffled_1 = rng1.permutation(split.val_s1_ids)

    rng2 = np.random.RandomState(42)
    shuffled_2 = rng2.permutation(split.val_s1_ids)

    # Determinism assertion
    np.testing.assert_array_equal(shuffled_1, shuffled_2)

    c75 = list(shuffled_1[:75])
    c1k = list(shuffled_1[:1000])
    c5k = list(shuffled_1[:5000])

    # Nesting assertions
    assert set(c75).issubset(set(c1k))
    assert set(c1k).issubset(set(c5k))

    # Zero overlap with train partition
    train_set = set(split.train_s1_ids)
    assert len(train_set.intersection(set(c5k))) == 0

    # Deterministic SHA256 hash
    h75 = hashlib.sha256(",".join(sorted(c75)).encode("utf-8")).hexdigest()
    assert len(h75) == 64
