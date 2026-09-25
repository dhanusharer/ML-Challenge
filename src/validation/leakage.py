"""Leakage safeguards and boundary verification.

ARCHITECTURAL BOUNDARIES FOR LEAKAGE PREVENTION:

1. ENTITY ISOLATION:
   - Source 1 entities allocated to the validation split MUST NEVER have their labels
     accessed during model training, candidate generation tuning, or feature engineering.
   - Validation ground-truth links are strictly sequestered for offline evaluation only.

2. FEATURE CALCULATION POLICIES:
   - Supervised Statistics: MUST be computed strictly on the training partition
     (e.g., target encoding, match priors, likelihood ratios, classifier training).
   - Validation & Test Inference: Must transform features using fitted parameters
     from the training pipeline without peeking at validation labels.
   - Global / Static Transforms: Fixed string normalizers (lowercase, punctuation removal,
     tokenization) are label-independent and may be applied deterministically across partitions.

3. LEAKAGE AUDIT FUNCTION:
   - verify_no_leakage() asserts zero entity overlap and zero label bleeding between partitions.
"""

from dataclasses import dataclass
from typing import Dict, List, Sequence, Set, Tuple


@dataclass(frozen=True)
class LeakageVerificationResult:
    """Diagnostic outcome of leakage verification."""
    is_leak_free: bool
    s1_entity_overlap_count: int
    train_entity_count: int
    val_entity_count: int
    validation_labels_in_train: int
    status_message: str


def verify_no_leakage(
    train_s1_ids: Sequence[str],
    val_s1_ids: Sequence[str],
    train_gt: Dict[str, List[str]],
    val_gt: Dict[str, List[str]],
) -> LeakageVerificationResult:
    """Perform strict assertion checks ensuring zero partition leakage.

    Args:
        train_s1_ids: Source 1 entity IDs in training set.
        val_s1_ids: Source 1 entity IDs in validation set.
        train_gt: Ground truth dictionary for training set.
        val_gt: Ground truth dictionary for validation set.

    Returns:
        LeakageVerificationResult

    Raises:
        AssertionError: If any leakage or cross-contamination is detected.
    """
    train_set = set(train_s1_ids)
    val_set = set(val_s1_ids)

    # 1. Entity overlap
    overlap = train_set & val_set
    if overlap:
        sample = sorted(list(overlap))[:5]
        msg = f"LEAKAGE DETECTED: {len(overlap)} S1 entities appear in BOTH train and validation sets! Samples: {sample}"
        raise AssertionError(msg)

    # 2. Train GT keys must strictly match train_s1_ids
    train_gt_keys = set(train_gt.keys())
    if train_gt_keys != train_set:
        diff_train = (train_gt_keys - train_set) | (train_set - train_gt_keys)
        sample = sorted(list(diff_train))[:5]
        msg = f"LEAKAGE / MISMATCH: train_gt keys do not match train_s1_ids exactly ({len(diff_train)} diff). Samples: {sample}"
        raise AssertionError(msg)

    # 3. Val GT keys must strictly match val_s1_ids
    val_gt_keys = set(val_gt.keys())
    if val_gt_keys != val_set:
        diff_val = (val_gt_keys - val_set) | (val_set - val_gt_keys)
        sample = sorted(list(diff_val))[:5]
        msg = f"LEAKAGE / MISMATCH: val_gt keys do not match val_s1_ids exactly ({len(diff_val)} diff). Samples: {sample}"
        raise AssertionError(msg)

    # 4. Check that no validation S1 keys exist inside train_gt
    leaked_val_keys = val_set & train_gt_keys
    if leaked_val_keys:
        sample = sorted(list(leaked_val_keys))[:5]
        msg = f"CRITICAL LEAKAGE: {len(leaked_val_keys)} validation S1 keys found inside train_gt! Samples: {sample}"
        raise AssertionError(msg)

    return LeakageVerificationResult(
        is_leak_free=True,
        s1_entity_overlap_count=0,
        train_entity_count=len(train_set),
        val_entity_count=len(val_set),
        validation_labels_in_train=0,
        status_message="PASS: 0 S1 entity overlap, strict boundary verified between train and validation partitions.",
    )
