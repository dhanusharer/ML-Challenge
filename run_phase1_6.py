#!/usr/bin/env python3
"""Phase 1.6 Full Target-Corpus Feasibility Gate CLI Runner.

Compares U6 candidate generation across three target universes:
- Universe A: Phase 1 Target Universe (~117K records)
- Universe B: Phase 1.5 Target Universe (~496K records)
- Universe C: Full 10.32M Training Target Universe (10,320,219 records)

Usage:
    python run_phase1_6.py --all
    python run_phase1_6.py --quick
    python run_phase1_6.py --report
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


def run_full(query_size: int = 10000) -> int:
    """Run full Phase 1.6 three-universe comparison."""
    from src.blocking.full_corpus_gate import FullCorpusGateRunner

    runner = FullCorpusGateRunner(query_cohort_size=query_size, seed=42)
    results = runner.run_all_comparisons()
    runner.export_deliverables(results)
    return 0


def run_quick() -> int:
    """Run quick smoke test."""
    from src.blocking.full_corpus_gate import FullCorpusGateRunner

    runner = FullCorpusGateRunner(query_cohort_size=1000, seed=42)
    results = runner.run_all_comparisons()
    runner.export_deliverables(results)
    return 0


def print_report() -> int:
    """Print high-level comparison table."""
    from src.utils.env import PROJECT_ROOT

    p_csv = PROJECT_ROOT / "reports" / "phase1_6" / "universe_comparison.csv"
    p_json = PROJECT_ROOT / "reports" / "phase1_6" / "phase1_6_summary.json"

    if not p_csv.is_file():
        print("No Phase 1.6 results found. Run `python run_phase1_6.py --all` first.")
        return 1

    print("\n" + "=" * 105)
    print("PHASE 1.6: FULL TARGET-CORPUS FEASIBILITY GATE — THREE UNIVERSE COMPARISON")
    print("=" * 105)
    print(f"{'Target Universe':<35} | {'Targets':<11} | {'Recall':<8} | {'S2 Rec':<8} | {'S3 Rec':<8} | {'Mean Cands':<10} | {'P95':<6} | {'Max':<5} | {'Time (s)':<8} | {'RAM (MB)':<8}")
    print("-" * 105)

    with open(p_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            print(
                f"{r['universe_name']:<35} | "
                f"{int(r['target_count']):<11,} | "
                f"{float(r['blocking_recall']):<8.4f} | "
                f"{float(r['s2_recall']):<8.4f} | "
                f"{float(r['s3_recall']):<8.4f} | "
                f"{float(r['mean_candidates']):<10.2f} | "
                f"{float(r['p95_candidates']):<6.0f} | "
                f"{int(r['max_candidates']):<5} | "
                f"{float(r['total_time_seconds']):<8.2f} | "
                f"{float(r['peak_ram_mb']):<8.1f}"
            )

    if p_json.is_file():
        with open(p_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        c = data.get("conclusions", {})
        print("\n" + "=" * 105)
        print("EMPIRICAL RECALL TRAJECTORY AS TARGET CORPUS APPROACHES REALITY:")
        print(f"  Universe A (~117K):     {data['target_universes'][0]['blocking_recall']:.4f}")
        print(f"       ↓  (-{c.get('degradation_a_to_b', 0):.4f} distractor drop)")
        print(f"  Universe B (~496K):     {data['target_universes'][1]['blocking_recall']:.4f}")
        print(f"       ↓  (-{c.get('degradation_b_to_c', 0):.4f} full-scale distractor drop)")
        print(f"  Universe C (10.32M):    {c.get('full_corpus_recall', 0):.4f}")
        print(f"  Total Degradation A->C: -{c.get('total_degradation_a_to_c', 0):.4f}")
        print(f"  Full Corpus Output:     Mean {c.get('full_corpus_mean_candidates', 0):.2f} candidates/S1 (P95: {c.get('full_corpus_p95_candidates', 0):.0f}, Max: {c.get('full_corpus_max_candidates', 0)})")
        print("=" * 105)

    return 0


def main():
    parser = argparse.ArgumentParser(description="Phase 1.6 Full Target-Corpus Feasibility Gate")
    parser.add_argument("--all", action="store_true", help="Run full 10K comparison across all 3 universes")
    parser.add_argument("--quick", action="store_true", help="Run quick smoke test (1K)")
    parser.add_argument("--report", action="store_true", help="Print summary comparison table")

    args = parser.parse_args()

    if not any([args.all, args.quick, args.report]):
        parser.print_help()
        sys.exit(0)

    if args.quick:
        code = run_quick()
        if code == 0:
            print_report()
        sys.exit(code)

    if args.all:
        code = run_full(query_size=10000)
        if code == 0:
            print_report()
        sys.exit(code)

    if args.report and not (args.all or args.quick):
        print_report()


if __name__ == "__main__":
    main()
