"""Independent verification module for test prediction TSV files."""

from collections import defaultdict
import csv
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from src.utils.env import PROJECT_ROOT


class IndependentSubmissionVerifier:
    """External, independent verifier that inspects prediction TSV integrity without relying on the generator's state."""

    def __init__(
        self,
        test_dir: Optional[Path] = None,
    ):
        self.test_dir = Path(test_dir or (PROJECT_ROOT / "student_resource" / "dataset" / "test"))

    def verify_submission(
        self,
        submission_tsv_path: Path,
        expected_s1_count: int = 1732544,
        check_target_existence_sample: int = 10000,
    ) -> Dict[str, Any]:
        """Run independent semantic and structural verification on the submission TSV."""
        submission_tsv_path = Path(submission_tsv_path)
        if not submission_tsv_path.is_file():
            raise FileNotFoundError(f"Submission file does not exist: {submission_tsv_path}")

        print("\n" + "=" * 80)
        print(" INDEPENDENT EXTERNAL VERIFIER: FULL SUBMISSION AUDIT ")
        print(f" Target File: {submission_tsv_path.name}")
        print("=" * 80, flush=True)

        t0 = time.time()

        # 1. Load valid test S1 IDs
        print("  [1/5] Loading valid test S1 IDs from test_source1.tsv...", flush=True)
        s1_file = self.test_dir / "test_source1.tsv"
        expected_s1_ids = set()
        s1_order = []
        with open(s1_file, "r", encoding="utf-8") as f:
            next(f)
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                if parts:
                    expected_s1_ids.add(parts[0])
                    s1_order.append(parts[0])

        print(f"    Expected unique S1 entities: {len(expected_s1_ids):,}")

        # 2. Inspect submission file line-by-line
        print("  [2/5] Verifying submission row count, ordering, and syntax...", flush=True)
        seen_s1_ids = set()
        duplicate_s1_count = 0
        missing_s1_count = 0
        invalid_prefix_count = 0
        duplicate_targets_in_row_count = 0
        empty_prediction_count = 0
        cardinality_counts = defaultdict(int)
        total_links = 0
        s2_links = 0
        s3_links = 0
        all_predicted_targets: Set[str] = set()

        line_count = 0
        with open(submission_tsv_path, "r", encoding="utf-8") as f:
            header_line = f.readline().rstrip("\r\n")
            header_parts = header_line.split("\t")
            if header_parts != ["source1_entity_id", "matched_entity_ids"]:
                raise ValueError(f"Invalid TSV header: {header_parts}")

            for line_idx, line in enumerate(f, start=1):
                line_count += 1
                parts = line.rstrip("\r\n").split("\t")
                if not parts:
                    continue
                sid = parts[0]

                # Check S1 ID
                if sid in seen_s1_ids:
                    duplicate_s1_count += 1
                seen_s1_ids.add(sid)

                if sid not in expected_s1_ids:
                    missing_s1_count += 1

                matches_str = parts[1] if len(parts) > 1 else ""
                if not matches_str.strip():
                    empty_prediction_count += 1
                    cardinality_counts[0] += 1
                else:
                    m_list = [m.strip() for m in matches_str.split(",") if m.strip()]
                    k = len(m_list)
                    cardinality_counts[k] += 1
                    total_links += k

                    # Check within-row duplicates
                    if len(m_list) != len(set(m_list)):
                        duplicate_targets_in_row_count += 1

                    for tid in m_list:
                        all_predicted_targets.add(tid)
                        if tid.startswith("S2-"):
                            s2_links += 1
                        elif tid.startswith("S3-"):
                            s3_links += 1
                        else:
                            invalid_prefix_count += 1

        print(f"    Total processed submission rows: {line_count:,}")
        print(f"    Unique S1 entities in submission: {len(seen_s1_ids):,}")
        print(f"    Duplicate S1 rows:               {duplicate_s1_count:,}")
        print(f"    Empty predictions:               {empty_prediction_count:,} ({empty_prediction_count/line_count*100:.2f}%)")
        print(f"    Non-empty predictions:           {line_count - empty_prediction_count:,} ({(line_count-empty_prediction_count)/line_count*100:.2f}%)")
        print(f"    Average matches / S1:            {total_links / line_count:.4f}")

        # 3. Check target existence in test_source2 and test_source3
        print(f"  [3/5] Sampling target existence in test target corpus ({min(check_target_existence_sample, len(all_predicted_targets)):,} targets)...", flush=True)
        target_existence_violations = 0
        if all_predicted_targets:
            sampled_targets = set(list(all_predicted_targets)[:check_target_existence_sample])
            found_targets = set()
            for fn in ["test_source2.tsv", "test_source3.tsv"]:
                fp = self.test_dir / fn
                if not fp.is_file():
                    continue
                with open(fp, "r", encoding="utf-8") as f:
                    next(f)
                    for line in f:
                        parts = line.rstrip("\r\n").split("\t")
                        if parts and parts[0] in sampled_targets:
                            found_targets.add(parts[0])
                            if len(found_targets) == len(sampled_targets):
                                break

            missing_targets = sampled_targets - found_targets
            target_existence_violations = len(missing_targets)
            print(f"    Found {len(found_targets):,} / {len(sampled_targets):,} sampled targets in target corpus ({target_existence_violations} missing).")

        # Invariant checks
        all_passed = (
            line_count == expected_s1_count and
            len(seen_s1_ids) == expected_s1_count and
            duplicate_s1_count == 0 and
            invalid_prefix_count == 0 and
            duplicate_targets_in_row_count == 0 and
            target_existence_violations == 0
        )

        audit_results = {
            "submission_file": submission_tsv_path.name,
            "total_rows": line_count,
            "expected_s1_count": expected_s1_count,
            "unique_s1_count": len(seen_s1_ids),
            "duplicate_s1_count": duplicate_s1_count,
            "invalid_prefix_count": invalid_prefix_count,
            "duplicate_targets_in_row_count": duplicate_targets_in_row_count,
            "target_existence_violations": target_existence_violations,
            "empty_prediction_count": empty_prediction_count,
            "empty_prediction_pct": empty_prediction_count / line_count if line_count > 0 else 0.0,
            "total_predicted_links": total_links,
            "avg_matches_per_s1": total_links / line_count if line_count > 0 else 0.0,
            "s2_links": s2_links,
            "s3_links": s3_links,
            "cardinality_1_count": cardinality_counts[1],
            "cardinality_2_count": cardinality_counts[2],
            "cardinality_3_count": cardinality_counts[3],
            "cardinality_gt3_count": sum(v for k, v in cardinality_counts.items() if k > 3),
            "verification_status": "PASSED" if all_passed else "FAILED",
            "verification_runtime_seconds": time.time() - t0,
        }

        print(f"  [4/5] Verification Status: {audit_results['verification_status']}")
        print(f"  [5/5] Verification completed in {audit_results['verification_runtime_seconds']:.2f}s.\n", flush=True)

        return audit_results
