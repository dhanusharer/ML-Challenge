#!/usr/bin/env python3
"""Phase 1.5 U6 Full-Scale Validation and Stability Gate CLI Runner.

Executes:
1. Progressive cohort scaling (5K, 25K, 100K)
2. Batch stability and variance analysis (5 non-overlapping folds)
3. Bitwise deterministic reproducibility verification
4. Output generation under reports/phase1_5/

Usage:
    python run_phase1_5.py --all
    python run_phase1_5.py --quick
    python run_phase1_5.py --report
"""

import argparse
import csv
import json
import sys
from pathlib import Path

# Ensure UTF-8 output encoding on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def run_full_gate() -> int:
    """Run full Phase 1.5 stability gate."""
    from src.blocking.stability_gate import StabilityGateRunner

    runner = StabilityGateRunner(
        max_queries=100000,
        distractor_target_size=150000,
        seed=42,
    )
    runner.load_partitions()
    scaling = runner.run_scaling_experiments(cohort_sizes=(5000, 25000, 100000))
    folds, summary = runner.run_batch_stability_experiments(fold_size=5000, num_folds=5)
    reproducible = runner.verify_deterministic_reproducibility(sample_size=2000)

    runner.export_all_deliverables(
        scaling_results=scaling,
        fold_results=folds,
        stability_summary=summary,
        is_reproducible=reproducible,
    )
    return 0


def run_quick_gate() -> int:
    """Run quick smoke stability gate."""
    from src.blocking.stability_gate import StabilityGateRunner

    runner = StabilityGateRunner(
        max_queries=10000,
        distractor_target_size=30000,
        seed=42,
    )
    runner.load_partitions()
    scaling = runner.run_scaling_experiments(cohort_sizes=(2000, 10000))
    folds, summary = runner.run_batch_stability_experiments(fold_size=2000, num_folds=3)
    reproducible = runner.verify_deterministic_reproducibility(sample_size=1000)

    runner.export_all_deliverables(
        scaling_results=scaling,
        fold_results=folds,
        stability_summary=summary,
        is_reproducible=reproducible,
    )
    return 0


def print_report() -> int:
    """Print high-level summary of Phase 1.5 stability gate."""
    from src.utils.env import PROJECT_ROOT

    scale_csv = PROJECT_ROOT / "reports" / "phase1_5" / "scaling_curve.csv"
    batch_csv = PROJECT_ROOT / "reports" / "phase1_5" / "batch_stability.csv"
    sum_json = PROJECT_ROOT / "reports" / "phase1_5" / "stability_gate_summary.json"

    if not scale_csv.is_file():
        print("No Phase 1.5 results found. Run `python run_phase1_5.py --all` first.")
        return 1

    print("\n" + "=" * 90)
    print("PHASE 1.5: U6 FULL-SCALE VALIDATION & STABILITY GATE SUMMARY")
    print("=" * 90)

    print(f"{'Scale (S1)':<12} | {'Recall':<8} | {'S2 Rec':<8} | {'S3 Rec':<8} | {'Mean Cands':<10} | {'P95':<6} | {'Max':<5} | {'Time (s)':<8} | {'QPS':<7} | {'RAM (MB)':<8}")
    print("-" * 90)
    with open(scale_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            print(
                f"{int(r['query_scale']):<12,}"
                f" | {float(r['blocking_recall']):<8.4f}"
                f" | {float(r['s2_recall']):<8.4f}"
                f" | {float(r['s3_recall']):<8.4f}"
                f" | {float(r['mean_candidates']):<10.2f}"
                f" | {float(r['p95_candidates']):<6.0f}"
                f" | {int(r['max_candidates']):<5}"
                f" | {float(r['runtime_seconds']):<8.2f}"
                f" | {float(r['throughput_qps']):<7.0f}"
                f" | {float(r['peak_ram_mb']):<8.1f}"
            )

    if batch_csv.is_file():
        print("\n" + "=" * 90)
        print("SAMPLE VARIANCE ACROSS NON-OVERLAPPING FOLDS")
        print("=" * 90)
        print(f"{'Fold':<10} | {'Recall':<8} | {'S2 Rec':<8} | {'S3 Rec':<8} | {'Mean Cands':<10} | {'P95':<6} | {'Max':<5} | {'Time (s)':<8}")
        print("-" * 90)
        with open(batch_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                print(
                    f"{r['fold_name']:<10}"
                    f" | {float(r['blocking_recall']):<8.4f}"
                    f" | {float(r['s2_recall']):<8.4f}"
                    f" | {float(r['s3_recall']):<8.4f}"
                    f" | {float(r['mean_candidates']):<10.2f}"
                    f" | {float(r['p95_candidates']):<6.0f}"
                    f" | {int(r['max_candidates']):<5}"
                    f" | {float(r['runtime_seconds']):<8.2f}"
                )

    if sum_json.is_file():
        with open(sum_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        var = data.get("variance_across_folds", {})
        ext = data.get("extrapolations", {})
        print("\n" + "=" * 90)
        print(f"GATE STATUS: {data.get('status')} | BITWISE REPRODUCIBILITY: {'VERIFIED' if data.get('deterministic_reproducibility') else 'FAILED'}")
        print(f"Recall Mean: {var.get('mean_recall', 0):.4f} ± {var.get('std_recall', 0):.4f} (CV: {var.get('cv_percent_recall', 0):.2f}%)")
        print(f"Target Search Universe: {data.get('evaluation_target_universe_size', 0):,} records")
        print(f"Full Validation Extrapolation ({ext.get('full_validation_s1_count', 0):,} S1): ~{ext.get('estimated_runtime_minutes', 0)} mins, RAM ~{ext.get('projected_peak_ram_mb', 0)} MB")
        print("=" * 90)

    return 0


def main():
    parser = argparse.ArgumentParser(description="Phase 1.5 U6 Full-Scale Stability Gate Runner")
    parser.add_argument("--all", action="store_true", help="Run full 100K scaling and stability gate")
    parser.add_argument("--quick", action="store_true", help="Run quick smoke gate (10K)")
    parser.add_argument("--report", action="store_true", help="Print summary of Phase 1.5 results")

    args = parser.parse_args()

    if not any([args.all, args.quick, args.report]):
        parser.print_help()
        sys.exit(0)

    if args.quick:
        code = run_quick_gate()
        if code == 0:
            print_report()
        sys.exit(code)

    if args.all:
        code = run_full_gate()
        if code == 0:
            print_report()
        sys.exit(code)

    if args.report and not (args.all or args.quick):
        print_report()


if __name__ == "__main__":
    main()
