#!/usr/bin/env python3
"""Main Phase 1 Candidate Generation and Discovery Runner.

Executes all Phase 1 candidate generation discovery tasks:
1. Automated unit test suite across representations and blocker families
2. Candidate generation benchmark across Block Families A through F
3. Progressive union and complementarity analysis
4. Generation and export of all Phase 1 deliverables under reports/phase1/

Usage:
    python run_phase1.py --all
    python run_phase1.py --test
    python run_phase1.py --benchmark
    python run_phase1.py --quick
    python run_phase1.py --report
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
    """Run Phase 1 and Phase 0 unit test suite."""
    print("=" * 60)
    print("STEP 1: RUNNING AUTOMATED UNIT TEST SUITE")
    print("=" * 60)
    import pytest
    return pytest.main(["tests", "-v"])


def run_benchmark(sample_size: int = 5000, distractor_size: int = 100000, seed: int = 42) -> int:
    """Run Candidate Generation Benchmark across all blocker families."""
    print("=" * 60)
    print(f"STEP 2: RUNNING CANDIDATE GENERATION BENCHMARK (N={sample_size:,})")
    print("=" * 60)
    from src.blocking.benchmark import BlockingBenchmarkRunner

    runner = BlockingBenchmarkRunner(
        sample_size=sample_size,
        distractor_target_size=distractor_size,
        seed=seed,
    )
    runner.load_partitions()
    metrics, candidate_cache = runner.run_all_benchmarks()
    slices = runner.run_slice_analysis(candidate_cache)
    complementarity = runner.run_complementarity_analysis(candidate_cache)
    unions = runner.run_union_experiments(candidate_cache)
    explosions, failures = runner.analyze_explosion_and_failures(candidate_cache)
    france_res = runner.analyze_france_test_data()

    runner.export_all_deliverables(
        metrics=metrics,
        slices=slices,
        unions=unions,
        complementarity=complementarity,
        explosions=explosions,
        failures=failures,
        france_analysis=france_res,
    )
    return 0


def print_summary_report() -> int:
    """Print high-level summary of Phase 1 deliverables."""
    from src.utils.env import PROJECT_ROOT
    import csv

    res_csv = PROJECT_ROOT / "reports" / "phase1" / "blocking_results.csv"
    union_csv = PROJECT_ROOT / "reports" / "phase1" / "union_results.csv"

    if not res_csv.is_file():
        print("No Phase 1 results found. Run `python run_phase1.py --benchmark` first.")
        return 1

    print("\n" + "=" * 80)
    print("PHASE 1 CANDIDATE GENERATION BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"{'Blocker Name':<35} | {'Recall':<8} | {'S2 Rec':<8} | {'S3 Rec':<8} | {'Mean Cands':<10} | {'P95':<6} | {'Red. Ratio':<10}")
    print("-" * 80)
    with open(res_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            print(
                f"{row['block_name']:<35} | "
                f"{float(row['blocking_recall']):<8.4f} | "
                f"{float(row['s2_recall']):<8.4f} | "
                f"{float(row['s3_recall']):<8.4f} | "
                f"{float(row['mean_candidates_per_s1']):<10.1f} | "
                f"{float(row['p95_candidates_per_s1']):<6.0f} | "
                f"{float(row['reduction_ratio']):<10.6f}"
            )

    if union_csv.is_file():
        print("\n" + "=" * 80)
        print("PROGRESSIVE UNION RESULTS")
        print("=" * 80)
        print(f"{'Union Configuration':<40} | {'Total Recall':<12} | {'Incremental':<12} | {'Mean Cands':<10} | {'P95':<6}")
        print("-" * 80)
        with open(union_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                print(
                    f"{row['union_name']:<40} | "
                    f"{float(row['blocking_recall']):<12.4f} | "
                    f"+{float(row['incremental_recall']):<11.4f} | "
                    f"{float(row['mean_candidates_per_s1']):<10.1f} | "
                    f"{float(row['p95_candidates_per_s1']):<6.0f}"
                )

    print("=" * 80)
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1 Candidate Generation and Discovery Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--all", action="store_true", help="Run tests, candidate generation benchmark, and print report")
    parser.add_argument("--test", action="store_true", help="Run automated test suite")
    parser.add_argument("--benchmark", action="store_true", help="Run full candidate generation benchmark")
    parser.add_argument("--quick", action="store_true", help="Run quick smoke benchmark (sample_size=500, distractor_size=10000)")
    parser.add_argument("--report", action="store_true", help="Print summary of Phase 1 results")

    args = parser.parse_args()

    if not any([args.all, args.test, args.benchmark, args.quick, args.report]):
        parser.print_help()
        sys.exit(0)

    if args.test or args.all:
        code = run_tests()
        if code != 0 and not args.all:
            sys.exit(code)

    if args.quick:
        code = run_benchmark(sample_size=500, distractor_size=10000)
        if code != 0:
            sys.exit(code)
        print_summary_report()

    if args.benchmark or args.all:
        code = run_benchmark(sample_size=5000, distractor_size=100000)
        if code != 0:
            sys.exit(code)
        print_summary_report()

    if args.report and not (args.benchmark or args.all or args.quick):
        print_summary_report()


if __name__ == "__main__":
    main()
