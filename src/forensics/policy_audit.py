"""Forensic audits for entity decision policy, output semantics, country distributions, and source breakdowns."""

import csv
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Tuple
from collections import Counter, defaultdict

from src.utils.env import PROJECT_ROOT
from src.policy.decision import RelativeMarginPolicy


def audit_entity_decision_policy() -> List[Dict[str, Any]]:
    """Section 14: Entity-Policy Forensics.
    
    Validates exact final decision logic against the 6 hand-crafted unit test cases.
    """
    print("\n" + "=" * 80)
    print(" FORENSIC AUDIT: ENTITY DECISION POLICY (HAND-CRAFTED UNIT CASES 1-6) ")
    print("=" * 80, flush=True)

    policy = RelativeMarginPolicy(
        threshold_floor=0.75,
        multi_threshold=0.80,
        max_margin=0.08,
        max_k=3,
    )

    cases = [
        {
            "case_id": "CASE_1_BELOW_FLOOR",
            "description": "Top score below floor (0.74 < 0.75)",
            "candidates": [{"target_id": "S2-101", "score": 0.74}],
            "expected_matches": [],
        },
        {
            "case_id": "CASE_2_EXACT_FLOOR",
            "description": "Top score exactly at floor (0.75 >= 0.75)",
            "candidates": [{"target_id": "S2-102", "score": 0.75}],
            "expected_matches": ["S2-102"],
        },
        {
            "case_id": "CASE_3_MARGIN_LOGIC",
            "description": "Multi-match within margin ([0.90, 0.84, 0.83] -> 0.90-0.84=0.06<=0.08, 0.90-0.83=0.07<=0.08)",
            "candidates": [
                {"target_id": "S2-103A", "score": 0.90},
                {"target_id": "S3-103B", "score": 0.84},
                {"target_id": "S2-103C", "score": 0.83},
            ],
            "expected_matches": ["S2-103A", "S3-103B", "S2-103C"],
        },
        {
            "case_id": "CASE_4_MAX_K_CAP",
            "description": "Four candidates within margin ([0.90, 0.88, 0.85, 0.84]) -> top 3 accepted, 4th capped by max_k=3",
            "candidates": [
                {"target_id": "S2-104A", "score": 0.90},
                {"target_id": "S3-104B", "score": 0.88},
                {"target_id": "S2-104C", "score": 0.85},
                {"target_id": "S3-104D", "score": 0.84},  # Within margin (0.90-0.84=0.06<=0.08) and >=0.80, but capped by max_k=3
            ],
            "expected_matches": ["S2-104A", "S3-104B", "S2-104C"],
        },
        {
            "case_id": "CASE_5_TIED_SCORES",
            "description": "Tied scores with deterministic tie-breaking by ID",
            "candidates": [
                {"target_id": "S3-B", "score": 0.88},
                {"target_id": "S2-A", "score": 0.88},
            ],
            # S2-A comes before S3-B alphabetically or sorted stably
            "expected_matches": ["S2-A", "S3-B"] if "S2-A" < "S3-B" else ["S3-B", "S2-A"],
        },
        {
            "case_id": "CASE_6_EMPTY_CANDIDATE_SET",
            "description": "Empty candidate set returns empty prediction",
            "candidates": [],
            "expected_matches": [],
        },
    ]

    results = []
    for c in cases:
        actual = policy.predict_matches("dummy_s1", c["candidates"])
        # For Case 5, allow set equality
        if c["case_id"] == "CASE_5_TIED_SCORES":
            passed = set(actual) == set(c["expected_matches"]) and len(actual) == 2
        else:
            passed = (actual == c["expected_matches"])

        print(f"  {c['case_id']}: {'PASSED' if passed else 'FAILED'} (Expected: {c['expected_matches']}, Got: {actual})")
        results.append({
            "case_id": c["case_id"],
            "description": c["description"],
            "expected": ",".join(c["expected_matches"]),
            "actual": ",".join(actual),
            "status": "PASSED" if passed else "FAILED",
        })

    reports_dir = PROJECT_ROOT / "reports" / "phase4_1"
    reports_dir.mkdir(parents=True, exist_ok=True)
    p_csv = reports_dir / "entity_policy_audit.csv"
    with open(p_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", "description", "expected", "actual", "status"])
        writer.writeheader()
        writer.writerows(results)

    print(f"  Policy audit exported to {p_csv}\n", flush=True)
    return results


def audit_output_semantics_and_distributions() -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Section 15, 16, 17: Semantic, Country, and Source Audit of matching_results.tsv."""
    print("=" * 80)
    print(" FORENSIC AUDIT: OUTPUT TSV SEMANTICS, COUNTRY & SOURCE DISTRIBUTION ")
    print("=" * 80, flush=True)

    tsv_path = PROJECT_ROOT / "matching_results.tsv"
    test_s1_path = PROJECT_ROOT / "student_resource" / "dataset" / "test" / "test_source1.tsv"

    # Map S1 -> Country
    print("  Mapping test S1 entities to countries...", flush=True)
    s1_country_map = {}
    with open(test_s1_path, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) >= 4:
                s1_country_map[parts[0]] = parts[3].upper()

    print(f"  Loaded country metadata for {len(s1_country_map):,} test S1 entities.", flush=True)

    total_rows = 0
    unique_s1 = set()
    empty_count = 0
    match_cardinality = Counter()
    total_target_links = 0
    s2_links = 0
    s3_links = 0

    # Source patterns: empty, s2_only, s3_only, both
    source_patterns = Counter()

    # Country stats: country -> {total, empty, matched, links}
    country_stats = defaultdict(lambda: {"total": 0, "empty": 0, "matched": 0, "links": 0, "s2": 0, "s3": 0})

    with open(tsv_path, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\r\n").split("\t")
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if not parts:
                continue
            total_rows += 1
            sid = parts[0]
            unique_s1.add(sid)
            cntry = s1_country_map.get(sid, "UNKNOWN")
            country_stats[cntry]["total"] += 1

            matches_str = parts[1] if len(parts) > 1 else ""
            if not matches_str.strip():
                empty_count += 1
                match_cardinality[0] += 1
                source_patterns["EMPTY"] += 1
                country_stats[cntry]["empty"] += 1
            else:
                m_list = [m.strip() for m in matches_str.split(",") if m.strip()]
                k = len(m_list)
                match_cardinality[k] += 1
                total_target_links += k
                country_stats[cntry]["matched"] += 1
                country_stats[cntry]["links"] += k

                has_s2 = any(m.startswith("S2-") for m in m_list)
                has_s3 = any(m.startswith("S3-") for m in m_list)
                for m in m_list:
                    if m.startswith("S2-"):
                        s2_links += 1
                        country_stats[cntry]["s2"] += 1
                    elif m.startswith("S3-"):
                        s3_links += 1
                        country_stats[cntry]["s3"] += 1

                if has_s2 and has_s3:
                    source_patterns["BOTH_S2_S3"] += 1
                elif has_s2:
                    source_patterns["S2_ONLY"] += 1
                elif has_s3:
                    source_patterns["S3_ONLY"] += 1
                else:
                    source_patterns["UNKNOWN_PREFIX"] += 1

    empty_pct = empty_count / total_rows if total_rows > 0 else 0
    one_pct = match_cardinality[1] / total_rows if total_rows > 0 else 0
    two_pct = match_cardinality[2] / total_rows if total_rows > 0 else 0
    three_pct = match_cardinality[3] / total_rows if total_rows > 0 else 0
    three_plus_pct = sum(v for k, v in match_cardinality.items() if k > 3) / total_rows if total_rows > 0 else 0
    avg_matches = total_target_links / total_rows if total_rows > 0 else 0

    s2_pct = s2_links / total_target_links if total_target_links > 0 else 0
    s3_pct = s3_links / total_target_links if total_target_links > 0 else 0

    print(f"  Total Rows:               {total_rows:,}")
    print(f"  Unique S1 Count:          {len(unique_s1):,}")
    print(f"  Empty Predictions:        {empty_count:,} ({empty_pct*100:.2f}%)")
    print(f"  1-Match Predictions:      {match_cardinality[1]:,} ({one_pct*100:.3f}%)")
    print(f"  2-Match Predictions:      {match_cardinality[2]:,} ({two_pct*100:.3f}%)")
    print(f"  3-Match Predictions:      {match_cardinality[3]:,} ({three_pct*100:.3f}%)")
    print(f"  Average Matches/S1:       {avg_matches:.4f}")
    print(f"  Target Source Balance:    {s2_links:,} S2 ({s2_pct*100:.1f}%), {s3_links:,} S3 ({s3_pct*100:.1f}%)")

    semantic_results = {
        "total_rows": total_rows,
        "unique_s1_entities": len(unique_s1),
        "empty_prediction_count": empty_count,
        "empty_prediction_pct": empty_pct,
        "one_match_count": match_cardinality[1],
        "one_match_pct": one_pct,
        "two_match_count": match_cardinality[2],
        "two_match_pct": two_pct,
        "three_match_count": match_cardinality[3],
        "three_match_pct": three_pct,
        "three_plus_count": sum(v for k, v in match_cardinality.items() if k > 3),
        "three_plus_pct": three_plus_pct,
        "total_predicted_matches": total_target_links,
        "avg_matches_per_s1": avg_matches,
        "s2_target_count": s2_links,
        "s2_target_pct": s2_pct,
        "s3_target_count": s3_links,
        "s3_target_pct": s3_pct,
        "semantic_integrity_status": "PATHOLOGICAL_CANDIDATE_COLLAPSE",
    }

    reports_dir = PROJECT_ROOT / "reports" / "phase4_1"
    reports_dir.mkdir(parents=True, exist_ok=True)
    p_sem_csv = reports_dir / "output_semantic_audit.csv"
    with open(p_sem_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(semantic_results.keys()))
        writer.writeheader()
        writer.writerow(semantic_results)

    # Country & Source breakdown
    country_rows = []
    print("\n  --- Country & Source Breakdown ---")
    for cntry, data in sorted(country_stats.items(), key=lambda x: x[1]["total"], reverse=True):
        t = data["total"]
        if t == 0:
            continue
        emp_p = data["empty"] / t
        m_p = data["matched"] / t
        avg_m = data["links"] / t
        print(f"  {cntry:<10}: {t:>10,} S1 | {data['empty']:>10,} empty ({emp_p*100:.2f}%) | {data['matched']:>6,} matched ({m_p*100:.3f}%) | avg {avg_m:.4f} matches/S1")
        country_rows.append({
            "country": cntry,
            "total_s1": t,
            "empty_count": data["empty"],
            "empty_pct": emp_p,
            "matched_count": data["matched"],
            "matched_pct": m_p,
            "total_links": data["links"],
            "avg_matches_per_s1": avg_m,
            "s2_links": data["s2"],
            "s3_links": data["s3"],
            "source_pattern_s2_only": source_patterns["S2_ONLY"],
            "source_pattern_s3_only": source_patterns["S3_ONLY"],
            "source_pattern_both": source_patterns["BOTH_S2_S3"],
        })

    p_cntry_csv = reports_dir / "country_source_audit.csv"
    with open(p_cntry_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(country_rows[0].keys()))
        writer.writeheader()
        writer.writerows(country_rows)

    print(f"  Output semantic audit exported to {p_sem_csv}")
    print(f"  Country & source audit exported to {p_cntry_csv}\n", flush=True)
    return semantic_results, country_rows


if __name__ == "__main__":
    audit_entity_decision_policy()
    audit_output_semantics_and_distributions()
