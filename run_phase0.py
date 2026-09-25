#!/usr/bin/env python3
"""Main Phase 0 Reproduction Runner.

Executes all Phase 0 foundation verification tasks:
1. Automated unit test suite (evaluator, parser, loaders, validation split, submission validator)
2. Data integrity audit across raw train and test TSV files
3. Deterministic entity-level validation split generation with leakage verification
4. Registration of experiment EXP-P0-001

Usage:
    python run_phase0.py --all
    python run_phase0.py --test
    python run_phase0.py --audit
    python run_phase0.py --split
"""

import argparse
import sys
from pathlib import Path

# Ensure UTF-8 output encoding on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def run_tests() -> int:
    """Run pytest suite."""
    print("=" * 60)
    print("STEP 1: RUNNING AUTOMATED UNIT TEST SUITE")
    print("=" * 60)
    import pytest
    return pytest.main(["tests", "-v"])


def run_audit(train_dir=None, test_dir=None) -> int:
    """Run data integrity audit."""
    print("=" * 60)
    print("STEP 2: RUNNING DATA INTEGRITY AUDIT")
    print("=" * 60)
    from src.data.integrity import DataIntegrityChecker
    from src.utils.env import PROJECT_ROOT
    import json

    checker = DataIntegrityChecker(train_dir=train_dir, test_dir=test_dir)
    report = checker.run_full_audit()

    out_file = PROJECT_ROOT / "reports" / "phase0" / "data_audit.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nAudit successfully written to: {out_file}")
    return 0


def run_split(train_dir=None, val_fraction=0.20, seed=42) -> int:
    """Run validation split and experiment registration."""
    print("=" * 60)
    print("STEP 3: RUNNING VALIDATION SPLIT & REGISTRATION")
    print("=" * 60)
    from src.utils.env import resolve_train_dir, PROJECT_ROOT
    from src.data.parser import parse_ground_truth_file
    from src.validation.split import create_entity_validation_split, partition_ground_truth
    from src.validation.leakage import verify_no_leakage
    from src.experiments.registry import register_experiment
    import json

    tdir = resolve_train_dir(train_dir)
    gt_file = tdir / "train_ground_truth.tsv"

    print(f"Loading ground truth: {gt_file}")
    gt_map = parse_ground_truth_file(gt_file)
    s1_ids = list(gt_map.keys())

    print(f"Creating deterministic split for {len(s1_ids):,} entities...")
    split = create_entity_validation_split(s1_ids, val_fraction=val_fraction, seed=seed)

    train_gt, val_gt = partition_ground_truth(gt_map, split)
    res = verify_no_leakage(split.train_s1_ids, split.val_s1_ids, train_gt, val_gt)
    print(f"Leakage Check: {res.status_message}")

    exp_record = register_experiment(
        experiment_id="EXP-P0-001",
        notes="Phase 0: Experiment and Evaluation Foundation established. Deterministic 80/20 entity split.",
        validation_seed=seed,
        validation_fraction=val_fraction,
        code_version="0.1.0",
        data_version="amazon-ml-challenge-2026-v1",
        metrics={
            "train_entities": split.train_entities,
            "val_entities": split.val_entities,
            "total_entities": split.total_entities,
        },
        extra_metadata=split.to_metadata_dict(),
    )

    out_file = PROJECT_ROOT / "reports" / "phase0" / "validation_split_summary.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "experiment_id": exp_record.experiment_id,
        "split_metadata": split.to_metadata_dict(),
        "leakage_verification": {
            "is_leak_free": res.is_leak_free,
            "overlap_count": res.s1_entity_overlap_count,
            "message": res.status_message,
        },
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Split metadata successfully written to: {out_file}")
    print(f"Experiment registered in experiments/EXP-P0-001/metadata.json")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Amazon ML Challenge 2026 Phase 0 Runner.")
    parser.add_argument("--all", action="store_true", help="Run tests, audit, and split")
    parser.add_argument("--test", action="store_true", help="Run automated test suite only")
    parser.add_argument("--audit", action="store_true", help="Run data integrity audit only")
    parser.add_argument("--split", action="store_true", help="Run validation split & registration only")
    parser.add_argument("--train-dir", default=None, help="Override train data directory")
    parser.add_argument("--test-dir", default=None, help="Override test data directory")

    args = parser.parse_args()

    # Default to --all if no specific action provided
    if not (args.test or args.audit or args.split):
        args.all = True

    if args.all or args.test:
        code = run_tests()
        if code != 0:
            print(f"Test suite failed with exit code {code}")
            return code

    if args.all or args.audit:
        code = run_audit(train_dir=args.train_dir, test_dir=args.test_dir)
        if code != 0:
            return code

    if args.all or args.split:
        code = run_split(train_dir=args.train_dir)
        if code != 0:
            return code

    print("\nPhase 0 execution completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
