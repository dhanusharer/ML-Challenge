"""Raw Data Integrity Checker and Audit Module.

Performs rigorous, factual checks on raw TSV files:
1. File existence, row counts, exact headers
2. Delimiter verification
3. Entity ID nullity and uniqueness
4. Prefix distributions (S1-, S2-, S3-)
5. Missingness across business_name, business_address, country
6. Country label distributions (open-set, not hardcoded)
7. Ground truth analysis (singletons, 1-match, multi-match, S2 vs S3 composition)
8. Ground truth structural checks (no S1- self-matches, no intra-list duplicates)
9. Raw exact agreement diagnostics on true links (exact name, exact address, exact name+address)

All operations are read-only and memory-efficient (streaming line-by-line / chunked).
"""

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import json
import os
import sys
import time

from src.utils.env import (
    DELIMITER,
    EXPECTED_SOURCE_COLUMNS,
    EXPECTED_GROUND_TRUTH_COLUMNS,
    VALID_MATCH_PREFIXES,
    resolve_train_dir,
    resolve_test_dir,
)


class DataIntegrityChecker:
    """Audits raw challenge data files without in-place modification."""

    def __init__(self, train_dir: Optional[Union[str, Path]] = None, test_dir: Optional[Union[str, Path]] = None):
        self.train_dir = Path(train_dir) if train_dir else resolve_train_dir()
        self.test_dir = Path(test_dir) if test_dir else resolve_test_dir()

    def audit_source_file(
        self,
        filepath: Union[str, Path],
        expected_prefix: str,
        sample_for_dupes: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Audit a single source TSV file (S1, S2, or S3)."""
        path = Path(filepath)
        if not path.is_file():
            return {"file": path.name, "exists": False, "error": "File does not exist"}

        file_size_mb = path.stat().st_size / (1024 * 1024)
        t0 = time.time()

        total_rows = 0
        null_ids = 0
        null_names = 0
        null_addresses = 0
        null_countries = 0
        prefix_counter: Counter = Counter()
        country_counter: Counter = Counter()

        # For duplicate entity ID detection
        seen_ids: Set[str] = set()
        duplicate_ids_count = 0
        sample_duplicate_ids: List[str] = []

        # Name & address duplicate frequency sampling / counters
        name_counter: Counter = Counter()
        addr_counter: Counter = Counter()

        header: List[str] = []

        with open(path, "r", encoding="utf-8", errors="replace") as f:
            header_line = f.readline()
            header = [c.strip() for c in header_line.rstrip("\r\n").split(DELIMITER)]

            for line_idx, line in enumerate(f):
                total_rows += 1
                parts = line.rstrip("\r\n").split(DELIMITER)

                # Guard against ragged rows
                while len(parts) < 4:
                    parts.append("")

                eid, name, addr, country = parts[0].strip(), parts[1].strip(), parts[2].strip(), parts[3].strip()

                # Entity ID checks
                if not eid:
                    null_ids += 1
                else:
                    if eid in seen_ids:
                        duplicate_ids_count += 1
                        if len(sample_duplicate_ids) < 5:
                            sample_duplicate_ids.append(eid)
                    seen_ids.add(eid)

                    # Extract prefix
                    prefix = eid.split("-")[0] + "-" if "-" in eid else "NO_PREFIX"
                    prefix_counter[prefix] += 1

                # Missingness
                if not name:
                    null_names += 1
                if not addr:
                    null_addresses += 1
                if not country:
                    null_countries += 1
                else:
                    country_counter[country] += 1

                # Collect sample duplicates for names/addresses (track top 50,000 to keep memory small)
                if total_rows <= 100000:
                    if name:
                        name_counter[name] += 1
                    if addr:
                        addr_counter[addr] += 1

        elapsed = time.time() - t0

        # Compute duplicate counts from sample
        dup_names_sample = sum(1 for c in name_counter.values() if c > 1)
        dup_addrs_sample = sum(1 for c in addr_counter.values() if c > 1)

        return {
            "file": path.name,
            "exists": True,
            "size_mb": round(file_size_mb, 2),
            "total_rows": total_rows,
            "columns": header,
            "is_schema_valid": header == EXPECTED_SOURCE_COLUMNS,
            "unique_entity_ids": len(seen_ids),
            "duplicate_entity_ids": duplicate_ids_count,
            "sample_duplicate_ids": sample_duplicate_ids,
            "null_entity_ids": null_ids,
            "null_business_names": null_names,
            "null_business_addresses": null_addresses,
            "null_countries": null_countries,
            "prefix_distribution": dict(prefix_counter),
            "expected_prefix": expected_prefix,
            "prefix_consistent": (
                len(prefix_counter) == 1 and expected_prefix in prefix_counter
            ),
            "country_distribution": dict(country_counter),
            "sample_100k_duplicate_names_count": dup_names_sample,
            "sample_100k_duplicate_addresses_count": dup_addrs_sample,
            "audit_duration_seconds": round(elapsed, 2),
        }

    def audit_ground_truth(
        self,
        filepath: Union[str, Path],
    ) -> Dict[str, Any]:
        """Audit train_ground_truth.tsv for structural correctness and match statistics."""
        path = Path(filepath)
        if not path.is_file():
            return {"file": path.name, "exists": False, "error": "File does not exist"}

        file_size_mb = path.stat().st_size / (1024 * 1024)
        t0 = time.time()

        total_rows = 0
        null_s1_ids = 0
        duplicate_s1_ids = 0
        seen_s1: Set[str] = set()
        sample_duplicate_s1: List[str] = []

        num_singletons = 0
        num_one_match = 0
        num_multi_match = 0

        match_count_distribution: Counter = Counter()
        s2_only_count = 0
        s3_only_count = 0
        s2_and_s3_count = 0

        total_links = 0
        total_s2_links = 0
        total_s3_links = 0

        malformed_rows = 0
        self_matches_count = 0
        invalid_prefix_count = 0
        intra_list_duplicates_count = 0

        header: List[str] = []

        with open(path, "r", encoding="utf-8") as f:
            header_line = f.readline()
            header = [c.strip() for c in header_line.rstrip("\r\n").split(DELIMITER)]

            for line_idx, line in enumerate(f, start=2):
                total_rows += 1
                parts = line.rstrip("\r\n").split(DELIMITER)
                if len(parts) == 1:
                    parts = [parts[0], ""]
                elif len(parts) != 2:
                    malformed_rows += 1
                    continue

                s1_id = parts[0].strip()
                matches_str = parts[1].strip()

                if not s1_id:
                    null_s1_ids += 1
                else:
                    if s1_id in seen_s1:
                        duplicate_s1_ids += 1
                        if len(sample_duplicate_s1) < 5:
                            sample_duplicate_s1.append(s1_id)
                    seen_s1.add(s1_id)

                if not matches_str:
                    num_singletons += 1
                    match_count_distribution[0] += 1
                    continue

                raw_mids = [m.strip() for m in matches_str.split(",") if m.strip()]
                num_matches = len(raw_mids)
                match_count_distribution[num_matches] += 1
                total_links += num_matches

                if num_matches == 1:
                    num_one_match += 1
                elif num_matches > 1:
                    num_multi_match += 1

                # Check intra-list duplicates
                if len(raw_mids) != len(set(raw_mids)):
                    intra_list_duplicates_count += 1

                has_s2 = False
                has_s3 = False

                for mid in raw_mids:
                    if mid.startswith("S1-"):
                        self_matches_count += 1
                    elif mid.startswith("S2-"):
                        has_s2 = True
                        total_s2_links += 1
                    elif mid.startswith("S3-"):
                        has_s3 = True
                        total_s3_links += 1
                    else:
                        invalid_prefix_count += 1

                if has_s2 and not has_s3:
                    s2_only_count += 1
                elif has_s3 and not has_s2:
                    s3_only_count += 1
                elif has_s2 and has_s3:
                    s2_and_s3_count += 1

        elapsed = time.time() - t0

        return {
            "file": path.name,
            "exists": True,
            "size_mb": round(file_size_mb, 2),
            "total_rows": total_rows,
            "columns": header,
            "is_schema_valid": header == EXPECTED_GROUND_TRUTH_COLUMNS,
            "unique_s1_entities": len(seen_s1),
            "duplicate_s1_ids": duplicate_s1_ids,
            "sample_duplicate_s1": sample_duplicate_s1,
            "null_s1_ids": null_s1_ids,
            "malformed_rows": malformed_rows,
            "self_matches_count": self_matches_count,
            "invalid_prefix_count": invalid_prefix_count,
            "intra_list_duplicates_count": intra_list_duplicates_count,
            "singleton_count": num_singletons,
            "singleton_percentage": round(100.0 * num_singletons / total_rows, 2) if total_rows > 0 else 0.0,
            "one_match_count": num_one_match,
            "one_match_percentage": round(100.0 * num_one_match / total_rows, 2) if total_rows > 0 else 0.0,
            "multi_match_count": num_multi_match,
            "multi_match_percentage": round(100.0 * num_multi_match / total_rows, 2) if total_rows > 0 else 0.0,
            "s2_only_match_count": s2_only_count,
            "s3_only_match_count": s3_only_count,
            "s2_and_s3_match_count": s2_and_s3_count,
            "total_true_links": total_links,
            "total_s2_links": total_s2_links,
            "total_s3_links": total_s3_links,
            "mean_links_per_s1": round(total_links / total_rows, 4) if total_rows > 0 else 0.0,
            "top_match_counts": dict(match_count_distribution.most_common(10)),
            "audit_duration_seconds": round(elapsed, 2),
        }

    def compute_exact_agreement_diagnostics(
        self,
        sample_size: int = 50000,
    ) -> Dict[str, Any]:
        """Measure descriptive raw exact string agreement on true links.

        AUDIT STATISTICS ONLY — no matcher is constructed.
        Measures:
        - exact name equality
        - exact address equality
        - exact name + address equality
        """
        gt_path = self.train_dir / "train_ground_truth.tsv"
        s1_path = self.train_dir / "train_source1.tsv"
        s2_path = self.train_dir / "train_source2.tsv"
        s3_path = self.train_dir / "train_source3.tsv"

        # 1. Collect true pairs to sample
        sample_pairs: List[Tuple[str, str]] = []  # (s1_id, target_id)
        with open(gt_path, "r", encoding="utf-8") as f:
            next(f)
            for line in f:
                parts = line.rstrip("\r\n").split(DELIMITER)
                if len(parts) == 2 and parts[1].strip():
                    s1 = parts[0].strip()
                    for mid in parts[1].strip().split(","):
                        m = mid.strip()
                        if m:
                            sample_pairs.append((s1, m))
                            if len(sample_pairs) >= sample_size:
                                break
                if len(sample_pairs) >= sample_size:
                    break

        needed_s1 = {s1 for s1, _ in sample_pairs}
        needed_s2 = {mid for _, mid in sample_pairs if mid.startswith("S2-")}
        needed_s3 = {mid for _, mid in sample_pairs if mid.startswith("S3-")}

        # 2. Extract records for needed IDs
        records: Dict[str, Tuple[str, str]] = {}  # id -> (name, address)

        for path, needed in [(s1_path, needed_s1), (s2_path, needed_s2), (s3_path, needed_s3)]:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                next(f)
                for line in f:
                    parts = line.rstrip("\r\n").split(DELIMITER)
                    if len(parts) >= 3:
                        eid = parts[0].strip()
                        if eid in needed:
                            records[eid] = (parts[1].strip(), parts[2].strip())
                            needed.remove(eid)
                            if not needed:
                                break

        # 3. Compute exact agreement
        evaluated_pairs = 0
        exact_name_matches = 0
        exact_address_matches = 0
        exact_both_matches = 0

        for s1, target in sample_pairs:
            if s1 in records and target in records:
                evaluated_pairs += 1
                s1_name, s1_addr = records[s1]
                t_name, t_addr = records[target]

                if s1_name == t_name:
                    exact_name_matches += 1
                if s1_addr == t_addr:
                    exact_address_matches += 1
                if s1_name == t_name and s1_addr == t_addr:
                    exact_both_matches += 1

        return {
            "sample_true_pairs_evaluated": evaluated_pairs,
            "exact_name_equality_count": exact_name_matches,
            "exact_name_equality_percentage": round(100.0 * exact_name_matches / evaluated_pairs, 2) if evaluated_pairs > 0 else 0.0,
            "exact_address_equality_count": exact_address_matches,
            "exact_address_equality_percentage": round(100.0 * exact_address_matches / evaluated_pairs, 2) if evaluated_pairs > 0 else 0.0,
            "exact_name_and_address_equality_count": exact_both_matches,
            "exact_name_and_address_equality_percentage": round(100.0 * exact_both_matches / evaluated_pairs, 2) if evaluated_pairs > 0 else 0.0,
            "diagnostic_note": "Factual audit statistics measuring raw string equality on true links without any matcher.",
        }

    def run_full_audit(self) -> Dict[str, Any]:
        """Execute complete Phase 0 data audit across train and test source files."""
        print("Starting Phase 0 Data Audit...")

        # Audit Train Sources
        print("  Auditing train_source1.tsv...")
        s1_train = self.audit_source_file(self.train_dir / "train_source1.tsv", "S1-")
        print("  Auditing train_source2.tsv...")
        s2_train = self.audit_source_file(self.train_dir / "train_source2.tsv", "S2-")
        print("  Auditing train_source3.tsv...")
        s3_train = self.audit_source_file(self.train_dir / "train_source3.tsv", "S3-")
        print("  Auditing train_ground_truth.tsv...")
        gt_train = self.audit_ground_truth(self.train_dir / "train_ground_truth.tsv")

        # Audit Test Sources (Structural / Data-quality inspection only per Section 16)
        print("  Auditing test_source1.tsv...")
        s1_test = self.audit_source_file(self.test_dir / "test_source1.tsv", "S1-")
        print("  Auditing test_source2.tsv...")
        s2_test = self.audit_source_file(self.test_dir / "test_source2.tsv", "S2-")
        print("  Auditing test_source3.tsv...")
        s3_test = self.audit_source_file(self.test_dir / "test_source3.tsv", "S3-")

        print("  Computing exact agreement diagnostics on ground truth sample...")
        exact_agreement = self.compute_exact_agreement_diagnostics(sample_size=50000)

        train_source_rows = s1_train["total_rows"] + s2_train["total_rows"] + s3_train["total_rows"]
        test_source_rows = s1_test["total_rows"] + s2_test["total_rows"] + s3_test["total_rows"]
        total_six_sources = train_source_rows + test_source_rows
        total_including_gt = total_six_sources + gt_train["total_rows"]

        report = {
            "audit_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "summary_totals": {
                "total_six_source_records": total_six_sources,
                "total_challenge_records_including_ground_truth": total_including_gt,
                "total_train_source_records": train_source_rows,
                "total_test_source_records": test_source_rows,
                "total_ground_truth_records": gt_train["total_rows"],
                "authoritative_definitions": {
                    "total_six_source_records": "Sum of entity data rows across the 6 source TSVs (train_source1/2/3 and test_source1/2/3), strictly excluding header rows and ground truth.",
                    "total_challenge_records_including_ground_truth": "Sum of all data rows across all 7 challenge TSVs including train_ground_truth.tsv, strictly excluding header rows.",
                    "train_source_records": "Sum of entity data rows in train_source1.tsv, train_source2.tsv, train_source3.tsv.",
                    "test_source_records": "Sum of entity data rows in test_source1.tsv, test_source2.tsv, test_source3.tsv.",
                    "ground_truth_records": "Total rows in train_ground_truth.tsv (corresponds 1-to-1 with train_source1 entities)."
                }
            },
            "train_sources": {
                "source1": s1_train,
                "source2": s2_train,
                "source3": s3_train,
            },
            "ground_truth": gt_train,
            "test_sources": {
                "source1": s1_test,
                "source2": s2_test,
                "source3": s3_test,
            },
            "raw_exact_agreement_diagnostics": exact_agreement,
        }
        return report



def main():
    import argparse
    from src.utils.env import PROJECT_ROOT

    parser = argparse.ArgumentParser(description="Run Phase 0 Data Integrity Audit.")
    parser.add_argument("--train-dir", default=None, help="Directory containing train TSVs")
    parser.add_argument("--test-dir", default=None, help="Directory containing test TSVs")
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "reports" / "phase0" / "data_audit.json"),
        help="Path to output JSON audit report",
    )
    args = parser.parse_args()

    checker = DataIntegrityChecker(train_dir=args.train_dir, test_dir=args.test_dir)
    report = checker.run_full_audit()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nPhase 0 Data Audit complete! Written to: {out_path}")
    print(f"Total 6 Source File Data Rows:                  {report['summary_totals']['total_six_source_records']:,}")
    print(f"Total Challenge Data Rows (with Ground Truth):  {report['summary_totals']['total_challenge_records_including_ground_truth']:,}")
    print(f"  - Train Source 1 Rows:    {report['train_sources']['source1']['total_rows']:,}")
    print(f"  - Train Source 2 Rows:    {report['train_sources']['source2']['total_rows']:,}")
    print(f"  - Train Source 3 Rows:    {report['train_sources']['source3']['total_rows']:,}")
    print(f"  - Test Source 1 Rows:     {report['test_sources']['source1']['total_rows']:,}")
    print(f"  - Test Source 2 Rows:     {report['test_sources']['source2']['total_rows']:,}")
    print(f"  - Test Source 3 Rows:     {report['test_sources']['source3']['total_rows']:,}")
    print(f"  - Ground Truth Rows:      {report['ground_truth']['total_rows']:,}")
    print(f"Ground Truth Singletons:    {report['ground_truth']['singleton_count']:,} ({report['ground_truth']['singleton_percentage']}%)")
    print(f"Total True Links:           {report['ground_truth']['total_true_links']:,}")
    diag = report['raw_exact_agreement_diagnostics']
    print(f"Exact Name Agreement (True Links Sample):    {diag['exact_name_equality_percentage']}%")
    print(f"Exact Address Agreement (True Links Sample): {diag['exact_address_equality_percentage']}%")
    print(f"Exact Both Agreement (True Links Sample):    {diag['exact_name_and_address_equality_percentage']}%")


if __name__ == "__main__":
    main()

