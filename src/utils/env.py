"""Environment configuration, filesystem paths, and authoritative constants.

Authoritative constraints referenced from Amazon ML Challenge 2026 problem statement.
Any parameters not explicitly dictated by Amazon are marked as:
ASSUMPTION — UNVERIFIED or PROJECT DESIGN CHOICE.
"""

from pathlib import Path
from typing import Dict, List, Optional
import os

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Candidate data paths (in priority order)
# 1. Official student_resource location provided with challenge package
# 2. Preferred modular data/raw and data/test locations
DATA_PATHS_TRAIN = [
    PROJECT_ROOT / "student_resource" / "dataset" / "train",
    PROJECT_ROOT / "data" / "raw",
]

DATA_PATHS_TEST = [
    PROJECT_ROOT / "student_resource" / "dataset" / "test",
    PROJECT_ROOT / "data" / "test",
]


def resolve_train_dir(override: Optional[str] = None) -> Path:
    """Resolve the training data directory.

    Raises:
        FileNotFoundError: If no valid train directory with the official files exists.
    """
    if override:
        p = Path(override)
        if p.is_dir():
            return p
        raise FileNotFoundError(f"Specified train directory does not exist: {override}")

    for p in DATA_PATHS_TRAIN:
        if p.is_dir() and (p / "train_source1.tsv").is_file():
            return p

    raise FileNotFoundError(
        "MANUAL ACTION REQUIRED: Place/provide the dataset at student_resource/dataset/train "
        "or data/raw OR specify the existing path via --train-dir."
    )


def resolve_test_dir(override: Optional[str] = None) -> Path:
    """Resolve the test data directory.

    Raises:
        FileNotFoundError: If no valid test directory exists.
    """
    if override:
        p = Path(override)
        if p.is_dir():
            return p
        raise FileNotFoundError(f"Specified test directory does not exist: {override}")

    for p in DATA_PATHS_TEST:
        if p.is_dir() and (p / "test_source1.tsv").is_file():
            return p

    raise FileNotFoundError(
        "MANUAL ACTION REQUIRED: Place/provide the dataset at student_resource/dataset/test "
        "or data/test OR specify the existing path via --test-dir."
    )


# Authoritative File Schema Constants
DELIMITER = "\t"
EXPECTED_SOURCE_COLUMNS: List[str] = [
    "entity_id",
    "business_name",
    "business_address",
    "country",
]

EXPECTED_GROUND_TRUTH_COLUMNS: List[str] = [
    "source1_entity_id",
    "matched_entity_ids",
]

EXPECTED_SUBMISSION_COLUMNS: List[str] = [
    "source1_entity_id",
    "matched_entity_ids",
]

EXPECTED_CANDIDATE_COLUMNS: List[str] = [
    "source1_entity_id",
    "candidate_entity_ids",
]

# Source ID prefixes dictated by challenge specification
SOURCE_PREFIXES: Dict[str, str] = {
    "source1": "S1-",
    "source2": "S2-",
    "source3": "S3-",
}

VALID_MATCH_PREFIXES = ("S2-", "S3-")

# Validation split parameters (PROJECT DESIGN CHOICE)
# Note: 80/20 train/validation split is a PROJECT DESIGN CHOICE, not an Amazon requirement.
DEFAULT_VALIDATION_FRACTION: float = 0.20  # PROJECT DESIGN CHOICE
DEFAULT_VALIDATION_SEED: int = 42          # PROJECT DESIGN CHOICE

# Authoritative F0.5 Formula Constant
F_BETA: float = 0.5
BETA_SQUARED: float = F_BETA ** 2  # 0.25
FACTOR_NUMERATOR: float = 1.0 + BETA_SQUARED  # 1.25
