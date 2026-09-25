"""Deterministic Entity-Level Validation Split.

CRITICAL DESIGN PRINCIPLE:
Source 1 entity resolution evaluation is macro-averaged across Source-1 entities.
Therefore, splitting MUST be performed strictly at the Source-1 ENTITY level, NEVER at the pair level.
A Source-1 entity must NEVER appear partially in training and partially in validation.

PROJECT DESIGN CHOICE:
The 80/20 train/validation split with random_state=42 is an internal project design choice,
NOT an Amazon challenge requirement.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple
import hashlib
import json
import numpy as np

from src.utils.env import DEFAULT_VALIDATION_FRACTION, DEFAULT_VALIDATION_SEED


@dataclass(frozen=True)
class ValidationSplit:
    """Metadata and entity ID sets for a deterministic validation split."""
    train_s1_ids: List[str]
    val_s1_ids: List[str]
    total_entities: int
    train_entities: int
    val_entities: int
    validation_fraction: float
    validation_seed: int
    split_policy: str  # Always "PROJECT DESIGN CHOICE"
    train_ids_hash: str
    val_ids_hash: str

    def to_metadata_dict(self) -> Dict:
        return {
            "total_entities": self.total_entities,
            "train_entities": self.train_entities,
            "val_entities": self.val_entities,
            "validation_fraction": self.validation_fraction,
            "validation_seed": self.validation_seed,
            "split_policy": self.split_policy,
            "train_ids_hash": self.train_ids_hash,
            "val_ids_hash": self.val_ids_hash,
        }


def _compute_id_list_hash(id_list: Sequence[str]) -> str:
    """Compute sha256 checksum of sorted IDs for reproducibility verification."""
    hasher = hashlib.sha256()
    for entity_id in id_list:
        hasher.update(entity_id.encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def create_entity_validation_split(
    s1_entity_ids: Sequence[str],
    val_fraction: float = DEFAULT_VALIDATION_FRACTION,
    seed: int = DEFAULT_VALIDATION_SEED,
) -> ValidationSplit:
    """Create a deterministic entity-level split on Source 1 IDs.

    Args:
        s1_entity_ids: Sequence of all unique Source 1 entity IDs.
        val_fraction: Fraction allocated to validation (default: 0.20, PROJECT DESIGN CHOICE).
        seed: Random seed for shuffling (default: 42, PROJECT DESIGN CHOICE).

    Returns:
        ValidationSplit containing the disjoint ID lists and cryptographic hashes.

    Raises:
        ValueError: If s1_entity_ids contains duplicates, is empty, or val_fraction is invalid.
    """
    if not s1_entity_ids:
        raise ValueError("s1_entity_ids sequence is empty.")

    if not (0.0 < val_fraction < 1.0):
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")

    unique_ids = sorted(list(set(s1_entity_ids)))
    if len(unique_ids) != len(s1_entity_ids):
        raise ValueError(
            f"Input s1_entity_ids contains {len(s1_entity_ids) - len(unique_ids)} duplicate IDs."
        )

    # Deterministic permutation using NumPy RandomState on canonically sorted IDs
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(unique_ids))

    num_val = int(round(len(unique_ids) * val_fraction))
    num_train = len(unique_ids) - num_val

    val_indices = indices[:num_val]
    train_indices = indices[num_val:]

    # Sort the resulting partitions so their order is strictly canonical
    val_s1_ids = sorted([unique_ids[i] for i in val_indices])
    train_s1_ids = sorted([unique_ids[i] for i in train_indices])

    # Assert 0 leakage
    train_set = set(train_s1_ids)
    val_set = set(val_s1_ids)
    overlap = train_set & val_set
    if overlap:
        raise RuntimeError(f"FATAL: Detected {len(overlap)} overlapping entities between train and validation!")

    if len(train_set) + len(val_set) != len(unique_ids):
        raise RuntimeError("FATAL: Partition size sum does not match total entity count.")

    return ValidationSplit(
        train_s1_ids=train_s1_ids,
        val_s1_ids=val_s1_ids,
        total_entities=len(unique_ids),
        train_entities=len(train_s1_ids),
        val_entities=len(val_s1_ids),
        validation_fraction=val_fraction,
        validation_seed=seed,
        split_policy="PROJECT DESIGN CHOICE (Entity-level disjoint S1 split)",
        train_ids_hash=_compute_id_list_hash(train_s1_ids),
        val_ids_hash=_compute_id_list_hash(val_s1_ids),
    )


def partition_ground_truth(
    ground_truth: Dict[str, List[str]],
    split: ValidationSplit,
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """Partition ground truth into train and validation ground truth mappings.

    Ensures validation ground truth is strictly sequestered from training.

    Args:
        ground_truth: Full {s1_id: [matched_ids]} mapping.
        split: ValidationSplit instance.

    Returns:
        (train_ground_truth, val_ground_truth)
    """
    train_gt: Dict[str, List[str]] = {}
    val_gt: Dict[str, List[str]] = {}

    train_set = set(split.train_s1_ids)
    val_set = set(split.val_s1_ids)

    for s1_id, matches in ground_truth.items():
        if s1_id in train_set:
            train_gt[s1_id] = matches
        elif s1_id in val_set:
            val_gt[s1_id] = matches
        else:
            raise KeyError(f"Entity '{s1_id}' in ground truth is not present in the split definition.")

    return train_gt, val_gt


def main():
    import argparse
    from src.utils.env import PROJECT_ROOT, resolve_train_dir
    from src.validation.leakage import verify_no_leakage
    from src.experiments.registry import register_experiment
    from src.data.parser import parse_ground_truth_file

    parser = argparse.ArgumentParser(description="Create deterministic entity-level validation split.")
    parser.add_argument("--train-dir", default=None, help="Directory containing train TSVs")
    parser.add_argument("--val-fraction", type=float, default=DEFAULT_VALIDATION_FRACTION, help="Validation fraction")
    parser.add_argument("--seed", type=int, default=DEFAULT_VALIDATION_SEED, help="Random seed")
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "reports" / "phase0" / "validation_split_summary.json"),
        help="Path to output summary JSON",
    )
    args = parser.parse_args()

    train_dir = resolve_train_dir(args.train_dir)
    gt_file = train_dir / "train_ground_truth.tsv"

    print(f"Loading ground truth from: {gt_file}")
    gt_map = parse_ground_truth_file(gt_file)
    s1_ids = list(gt_map.keys())

    print(f"Creating deterministic entity-level split for {len(s1_ids):,} entities...")
    split = create_entity_validation_split(
        s1_entity_ids=s1_ids,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )

    print("Partitioning ground truth into train and validation partitions...")
    train_gt, val_gt = partition_ground_truth(gt_map, split)

    print("Verifying partition leakage...")
    leakage_res = verify_no_leakage(split.train_s1_ids, split.val_s1_ids, train_gt, val_gt)
    print(f"Leakage status: {leakage_res.status_message}")

    # Register Experiment EXP-P0-001
    print("Registering Phase 0 foundation experiment EXP-P0-001...")
    exp_record = register_experiment(
        experiment_id="EXP-P0-001",
        notes="Phase 0: Experiment and Evaluation Foundation established. Deterministic 80/20 entity split.",
        validation_seed=args.seed,
        validation_fraction=args.val_fraction,
        code_version="0.1.0",
        data_version="amazon-ml-challenge-2026-v1",
        metrics={
            "train_entities": split.train_entities,
            "val_entities": split.val_entities,
            "total_entities": split.total_entities,
        },
        extra_metadata=split.to_metadata_dict(),
    )

    summary = {
        "experiment_id": exp_record.experiment_id,
        "split_metadata": split.to_metadata_dict(),
        "leakage_verification": {
            "is_leak_free": leakage_res.is_leak_free,
            "overlap_count": leakage_res.s1_entity_overlap_count,
            "message": leakage_res.status_message,
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Validation split complete and verified! Written to: {out_path}")


if __name__ == "__main__":
    main()

