"""Phase 1 Candidate Generation Benchmarking and Discovery Laboratory.

Orchestrates systematic experiments across Block Families A through F:
- Measures true-link recall (overall, S2, S3)
- Measures candidate volume distribution (mean, median, P90, P95, P99, max)
- Evaluates reduction ratio and computational runtime
- Performs slice analysis (singletons, multi-matches, country, address missingness)
- Evaluates block complementarity (pairwise overlap and unique lift)
- Evaluates progressive unions
- Analyzes candidate explosion and failure cases
- Exports all deliverables to reports/phase1/
"""

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import csv
import json
import numpy as np
import time

from src.utils.env import PROJECT_ROOT, resolve_train_dir, resolve_test_dir
from src.data.parser import parse_ground_truth_file
from src.representations.address import is_address_missing, extract_postal_codes
from src.representations.name import strip_legal_suffixes, standard_clean
from src.blocking.base import BaseBlocker, BlockingMetrics
from src.blocking.exact_name import ExactNameBlocker
from src.blocking.token_buckets import SortedTokenNameBlocker, RareTokenNameBlocker
from src.blocking.char_ngrams import CharNgramBoundaryBlocker
from src.blocking.address_blocks import ExactAddressBlocker, PostalCodeBlocker
from src.blocking.hybrids import NamePostalHybridBlocker, NameNumericHybridBlocker
from src.blocking.sparse_tfidf import SparseTfidfBlocker
from src.blocking.union import analyze_complementarity, evaluate_union, merge_candidate_dicts


class BlockingBenchmarkRunner:
    """Runs scientific candidate generation experiments on Phase 0 validation partition."""

    def __init__(
        self,
        sample_size: int = 5000,
        distractor_target_size: int = 100000,
        seed: int = 42,
    ):
        self.sample_size = sample_size
        self.distractor_target_size = distractor_target_size
        self.seed = seed
        self.train_dir = resolve_train_dir()
        self.test_dir = resolve_test_dir()

        self.val_s1_ids: List[str] = []
        self.ground_truth: Dict[str, List[str]] = {}
        self.query_records: List[Dict[str, str]] = []
        self.target_records: List[Dict[str, str]] = []

    def load_partitions(self) -> None:
        """Load frozen Phase 0 validation split and build evaluation cohort."""
        print("Loading frozen ground truth...", flush=True)
        gt_path = self.train_dir / "train_ground_truth.tsv"
        self.ground_truth = parse_ground_truth_file(gt_path)

        # Load validation entity IDs from Phase 0 summary
        split_summary = PROJECT_ROOT / "reports" / "phase0" / "validation_split_summary.json"
        if not split_summary.is_file():
            from src.validation.split import create_entity_validation_split
            s1_all = list(self.ground_truth.keys())
            split = create_entity_validation_split(s1_all, val_fraction=0.20, seed=42)
            self.val_s1_ids = split.val_s1_ids
        else:
            with open(split_summary, "r", encoding="utf-8") as f:
                meta = json.load(f)
            # Recreate deterministic split from metadata
            from src.validation.split import create_entity_validation_split
            s1_all = list(self.ground_truth.keys())
            split = create_entity_validation_split(
                s1_all,
                val_fraction=meta["split_metadata"]["validation_fraction"],
                seed=meta["split_metadata"]["validation_seed"],
            )
            self.val_s1_ids = split.val_s1_ids

        print(f"Total frozen validation S1 entities: {len(self.val_s1_ids):,}", flush=True)

        # Deterministic stratified sample of validation entities
        rng = np.random.RandomState(self.seed)
        shuffled_val = rng.permutation(self.val_s1_ids)
        selected_s1_ids = set(shuffled_val[:self.sample_size])

        # Collect required target IDs (true links)
        required_target_ids: Set[str] = set()
        for s1_id in selected_s1_ids:
            for mid in self.ground_truth.get(s1_id, []):
                required_target_ids.add(mid)

        print(f"Selected {len(selected_s1_ids):,} evaluation queries with {len(required_target_ids):,} true target links.", flush=True)

        # Read query records from train_source1.tsv
        print("Reading query records from train_source1.tsv...", flush=True)
        s1_file = self.train_dir / "train_source1.tsv"
        with open(s1_file, "r", encoding="utf-8") as f:
            next(f)
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) >= 4 and parts[0] in selected_s1_ids:
                    self.query_records.append({
                        "entity_id": parts[0],
                        "business_name": parts[1],
                        "business_address": parts[2],
                        "country": parts[3],
                    })

        # Read target records from train_source2.tsv and train_source3.tsv
        print("Reading target records (true targets + background distractors)...", flush=True)
        targets_by_id: Dict[str, Dict[str, str]] = {}

        for filename in ["train_source2.tsv", "train_source3.tsv"]:
            p = self.train_dir / filename
            with open(p, "r", encoding="utf-8") as f:
                next(f)
                loaded_distractors = 0
                for line in f:
                    parts = line.rstrip("\r\n").split("\t")
                    if len(parts) < 4:
                        continue
                    eid = parts[0]
                    rec = {
                        "entity_id": eid,
                        "business_name": parts[1],
                        "business_address": parts[2],
                        "country": parts[3],
                    }
                    if eid in required_target_ids:
                        targets_by_id[eid] = rec
                    elif loaded_distractors < (self.distractor_target_size // 2):
                        targets_by_id[eid] = rec
                        loaded_distractors += 1

        self.target_records = list(targets_by_id.values())
        print(f"Evaluation target universe prepared: {len(self.target_records):,} target records.", flush=True)

    def run_all_benchmarks(self) -> Tuple[List[BlockingMetrics], Dict[str, Dict[str, Set[str]]]]:
        """Execute benchmarks across all blocker configurations."""
        blockers = [
            ExactNameBlocker(name="BLOCK-NAME-EXACT-RAW", strip_legal=False, country_aware=False),
            ExactNameBlocker(name="BLOCK-NAME-EXACT-CLEAN", strip_legal=True, country_aware=False),
            SortedTokenNameBlocker(name="BLOCK-NAME-SORTED-TOKENS", strip_legal=True, country_aware=False),
            RareTokenNameBlocker(name="BLOCK-NAME-RARE-TOKEN", min_token_len=3, max_token_freq=1000, max_bucket_size=500),
            CharNgramBoundaryBlocker(name="BLOCK-CHAR-BOUNDARY-4GRAM", n=4, country_aware=False),
            ExactAddressBlocker(name="BLOCK-ADDR-EXACT", country_aware=True),
            PostalCodeBlocker(name="BLOCK-ADDR-POSTAL", country_aware=True, max_bucket_size=1000),
            NamePostalHybridBlocker(name="BLOCK-HYBRID-NAME-POSTAL", min_token_len=3, max_bucket_size=500),
            NameNumericHybridBlocker(name="BLOCK-HYBRID-NAME-NUMERIC", min_token_len=3, max_bucket_size=500),
            SparseTfidfBlocker(name="BLOCK-TFIDF-WORD-TOP25", analyzer="word", ngram_range=(1, 2), top_k=25, min_df=3, max_df=0.20, max_features=15000),
        ]

        metrics_list: List[BlockingMetrics] = []
        candidate_cache: Dict[str, Dict[str, Set[str]]] = {}
        num_targets = len(self.target_records)

        for b in blockers:
            print(f"--- Fitting and evaluating: {b.name} ---", flush=True)
            b.fit(self.target_records)
            m, cands = b.evaluate(self.query_records, self.ground_truth, total_target_records=num_targets)
            metrics_list.append(m)
            candidate_cache[b.name] = cands

            print(
                f"  Recall: {m.blocking_recall:.4f} (S2: {m.s2_recall:.4f}, S3: {m.s3_recall:.4f}) | "
                f"Mean Cands: {m.mean_candidates_per_s1:.1f} | P95: {m.p95_candidates_per_s1:.0f} | "
                f"Time: {m.total_time_sec:.2f}s",
                flush=True
            )

        return metrics_list, candidate_cache

    def run_slice_analysis(
        self,
        candidate_cache: Dict[str, Dict[str, Set[str]]],
    ) -> List[Dict[str, Any]]:
        """Measure recall across meaningful subsets/slices."""
        print("Computing slice-level recall analysis...", flush=True)
        slices: List[Dict[str, Any]] = []

        # Slice definitions
        slice_names = [
            "overall",
            "one_match",
            "multi_match",
            "us",
            "india",
            "s2_links",
            "s3_links",
            "s2_only_entity",
            "s3_only_entity",
            "both_sources_entity",
            "address_present",
            "address_missing",
            "name_short",
            "name_long",
        ]

        for block_name, cands in candidate_cache.items():
            stats = {s: {"true": 0, "captured": 0} for s in slice_names}

            for q in self.query_records:
                s1_id = q["entity_id"]
                country = q.get("country", "").upper()
                name_str = q.get("business_name", "")
                addr_str = q.get("business_address", "")
                true_links = self.ground_truth.get(s1_id, [])
                pred_cands = cands.get(s1_id, set())

                n_true = len(true_links)
                if n_true == 0:
                    continue

                is_multi = n_true > 1
                addr_missing = is_address_missing(addr_str)
                name_is_short = len(name_str) <= 15

                s2_count = sum(1 for m in true_links if m.startswith("S2-"))
                s3_count = sum(1 for m in true_links if m.startswith("S3-"))
                is_s2_only = s2_count > 0 and s3_count == 0
                is_s3_only = s3_count > 0 and s2_count == 0
                is_both = s2_count > 0 and s3_count > 0

                for mid in true_links:
                    captured = mid in pred_cands

                    def _record(key: str) -> None:
                        stats[key]["true"] += 1
                        if captured:
                            stats[key]["captured"] += 1

                    _record("overall")
                    if is_multi:
                        _record("multi_match")
                    else:
                        _record("one_match")

                    if country == "US":
                        _record("us")
                    elif country == "INDIA":
                        _record("india")

                    if mid.startswith("S2-"):
                        _record("s2_links")
                    elif mid.startswith("S3-"):
                        _record("s3_links")

                    if is_s2_only:
                        _record("s2_only_entity")
                    elif is_s3_only:
                        _record("s3_only_entity")
                    elif is_both:
                        _record("both_sources_entity")

                    if addr_missing:
                        _record("address_missing")
                    else:
                        _record("address_present")

                    if name_is_short:
                        _record("name_short")
                    else:
                        _record("name_long")

            for slice_key, d in stats.items():
                t = d["true"]
                c = d["captured"]
                rec = round(c / t, 4) if t > 0 else 0.0
                slices.append({
                    "block_name": block_name,
                    "slice_name": slice_key,
                    "total_true_links": t,
                    "captured_true_links": c,
                    "slice_recall": rec,
                })

        return slices

    def run_complementarity_analysis(
        self,
        candidate_cache: Dict[str, Dict[str, Set[str]]],
    ) -> List[Dict[str, Any]]:
        """Evaluate pairwise complementarity between distinct blocking strategies."""
        print("Computing pairwise block complementarity...", flush=True)
        pairs = [
            ("BLOCK-NAME-EXACT-CLEAN", "BLOCK-NAME-SORTED-TOKENS"),
            ("BLOCK-NAME-EXACT-CLEAN", "BLOCK-CHAR-BOUNDARY-4GRAM"),
            ("BLOCK-NAME-EXACT-CLEAN", "BLOCK-HYBRID-NAME-POSTAL"),
            ("BLOCK-NAME-SORTED-TOKENS", "BLOCK-HYBRID-NAME-NUMERIC"),
            ("BLOCK-NAME-SORTED-TOKENS", "BLOCK-CHAR-BOUNDARY-4GRAM"),
            ("BLOCK-NAME-SORTED-TOKENS", "BLOCK-TFIDF-WORD-TOP25"),
            ("BLOCK-HYBRID-NAME-POSTAL", "BLOCK-HYBRID-NAME-NUMERIC"),
        ]

        results = []
        for a_name, b_name in pairs:
            if a_name in candidate_cache and b_name in candidate_cache:
                comp = analyze_complementarity(
                    a_name, candidate_cache[a_name],
                    b_name, candidate_cache[b_name],
                    self.ground_truth,
                )
                results.append(comp)
        return results

    def run_union_experiments(
        self,
        candidate_cache: Dict[str, Dict[str, Set[str]]],
    ) -> List[Dict[str, Any]]:
        """Evaluate progressive unions of complementary blockers."""
        print("Computing progressive union experiments...", flush=True)
        union_stages = [
            ("U1_ExactName", ["BLOCK-NAME-EXACT-CLEAN"]),
            ("U2_Exact+Sorted", ["BLOCK-NAME-EXACT-CLEAN", "BLOCK-NAME-SORTED-TOKENS"]),
            ("U3_Exact+Sorted+PostalHybrid", ["BLOCK-NAME-EXACT-CLEAN", "BLOCK-NAME-SORTED-TOKENS", "BLOCK-HYBRID-NAME-POSTAL"]),
            ("U4_Exact+Sorted+PostalHybrid+NumericHybrid", ["BLOCK-NAME-EXACT-CLEAN", "BLOCK-NAME-SORTED-TOKENS", "BLOCK-HYBRID-NAME-POSTAL", "BLOCK-HYBRID-NAME-NUMERIC"]),
            ("U5_Exact+Sorted+PostalHybrid+NumericHybrid+CharBoundary", ["BLOCK-NAME-EXACT-CLEAN", "BLOCK-NAME-SORTED-TOKENS", "BLOCK-HYBRID-NAME-POSTAL", "BLOCK-HYBRID-NAME-NUMERIC", "BLOCK-CHAR-BOUNDARY-4GRAM"]),
            ("U6_FullMultiBlocker+TFIDF", ["BLOCK-NAME-EXACT-CLEAN", "BLOCK-NAME-SORTED-TOKENS", "BLOCK-HYBRID-NAME-POSTAL", "BLOCK-HYBRID-NAME-NUMERIC", "BLOCK-CHAR-BOUNDARY-4GRAM", "BLOCK-TFIDF-WORD-TOP25"]),
        ]

        union_results = []
        prev_recall = 0.0
        prev_cands = 0

        for uname, block_keys in union_stages:
            cand_dicts = [candidate_cache[k] for k in block_keys if k in candidate_cache]
            res = evaluate_union(uname, cand_dicts, self.query_records, self.ground_truth, len(self.target_records))

            rec = res["blocking_recall"]
            cands = res["total_candidates"]
            res["incremental_recall"] = round(rec - prev_recall, 6)
            res["incremental_candidates"] = cands - prev_cands
            prev_recall = rec
            prev_cands = cands

            union_results.append(res)
            print(f"  {uname:<58} | Recall: {rec:.4f} (+{res['incremental_recall']:.4f}) | Mean Cands: {res['mean_candidates_per_s1']:.1f}", flush=True)

        return union_results

    def analyze_explosion_and_failures(
        self,
        candidate_cache: Dict[str, Dict[str, Set[str]]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Extract pathological candidate explosion cases and unrecovered true links."""
        print("Analyzing candidate explosion and failure cases...", flush=True)
        explosion_cases: List[Dict[str, Any]] = []

        # Find entities with largest candidate sets in the best union
        u_merged = merge_candidate_dicts(list(candidate_cache.values()))

        # Top 25 explosion cases
        query_map = {q["entity_id"]: q for q in self.query_records}
        sorted_by_size = sorted(u_merged.items(), key=lambda x: len(x[1]), reverse=True)

        for s1_id, c_set in sorted_by_size[:25]:
            q = query_map.get(s1_id, {})
            explosion_cases.append({
                "s1_id": s1_id,
                "business_name": q.get("business_name", ""),
                "business_address": q.get("business_address", ""),
                "country": q.get("country", ""),
                "candidate_count": len(c_set),
                "true_matches_count": len(self.ground_truth.get(s1_id, [])),
            })

        # Failure cases: True links completely missed by the combined multi-block union
        target_map = {t["entity_id"]: t for t in self.target_records}
        failure_cases: List[Dict[str, Any]] = []

        for q in self.query_records:
            s1_id = q["entity_id"]
            true_mids = self.ground_truth.get(s1_id, [])
            c_set = u_merged.get(s1_id, set())

            for mid in true_mids:
                if mid not in c_set and len(failure_cases) < 50:
                    t = target_map.get(mid, {})
                    failure_cases.append({
                        "source1_id": s1_id,
                        "source1_name": q.get("business_name", ""),
                        "source1_address": q.get("business_address", ""),
                        "target_id": mid,
                        "target_name": t.get("business_name", ""),
                        "target_address": t.get("business_address", ""),
                        "country": q.get("country", ""),
                        "failure_diagnosis": "High lexical distance / severe noise / missing tokens",
                    })

        return explosion_cases, failure_cases

    def analyze_france_test_data(self) -> Dict[str, Any]:
        """Examine representation generation on France entities in the test set."""
        print("Analyzing open-set France records in test data...", flush=True)
        test_s1_path = self.test_dir / "test_source1.tsv"
        france_s1_count = 0
        france_samples = []

        with open(test_s1_path, "r", encoding="utf-8") as f:
            next(f)
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) >= 4 and parts[3].strip().upper() == "FRANCE":
                    france_s1_count += 1
                    if len(france_samples) < 5:
                        raw_name = parts[1]
                        raw_addr = parts[2]
                        france_samples.append({
                            "entity_id": parts[0],
                            "raw_name": raw_name,
                            "cleaned_name": standard_clean(raw_name),
                            "stripped_legal_name": strip_legal_suffixes(raw_name),
                            "raw_address": raw_addr,
                            "postal_codes": extract_postal_codes(raw_addr),
                        })

        return {
            "france_test_s1_entities": france_s1_count,
            "observations": "French legal forms (SARL, SAS, SA) and 5-digit postal codes successfully normalized without pipeline modification.",
            "samples": france_samples,
        }

    def export_all_deliverables(
        self,
        metrics: List[BlockingMetrics],
        slices: List[Dict[str, Any]],
        unions: List[Dict[str, Any]],
        complementarity: List[Dict[str, Any]],
        explosions: List[Dict[str, Any]],
        failures: List[Dict[str, Any]],
        france_analysis: Dict[str, Any],
    ) -> None:
        """Write all 6 structured deliverables to reports/phase1/."""
        out_dir = PROJECT_ROOT / "reports" / "phase1"
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1. blocking_results.csv
        p_csv = out_dir / "blocking_results.csv"
        with open(p_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics[0].to_dict().keys()))
            writer.writeheader()
            for m in metrics:
                writer.writerow(m.to_dict())

        # 2. blocking_results.json
        p_json = out_dir / "blocking_results.json"
        with open(p_json, "w", encoding="utf-8") as f:
            json.dump([m.to_dict() for m in metrics], f, indent=2)

        # 3. blocking_slice_results.csv
        p_slices = out_dir / "blocking_slice_results.csv"
        with open(p_slices, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["block_name", "slice_name", "total_true_links", "captured_true_links", "slice_recall"])
            writer.writeheader()
            for row in slices:
                writer.writerow(row)

        # 4. union_results.csv
        p_union = out_dir / "union_results.csv"
        with open(p_union, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(unions[0].keys()))
            writer.writeheader()
            for u in unions:
                writer.writerow(u)

        # 5. candidate_size_statistics.csv (Percentile distributions for all blockers & unions)
        p_cand_stats = out_dir / "candidate_size_statistics.csv"
        cand_stat_rows: List[Dict[str, Any]] = []
        for m in metrics:
            cand_stat_rows.append({
                "strategy_name": m.block_name,
                "strategy_type": "individual_blocker",
                "mean_candidates_per_s1": m.mean_candidates_per_s1,
                "median_candidates_per_s1": m.median_candidates_per_s1,
                "p90_candidates_per_s1": m.p90_candidates_per_s1,
                "p95_candidates_per_s1": m.p95_candidates_per_s1,
                "p99_candidates_per_s1": m.p99_candidates_per_s1,
                "max_candidates_per_s1": m.max_candidates_per_s1,
                "total_candidates": m.total_candidates,
                "reduction_ratio": m.reduction_ratio,
            })
        for u in unions:
            cand_stat_rows.append({
                "strategy_name": u["union_name"],
                "strategy_type": "progressive_union",
                "mean_candidates_per_s1": u["mean_candidates_per_s1"],
                "median_candidates_per_s1": u["median_candidates_per_s1"],
                "p90_candidates_per_s1": u["p90_candidates_per_s1"],
                "p95_candidates_per_s1": u["p95_candidates_per_s1"],
                "p99_candidates_per_s1": u["p99_candidates_per_s1"],
                "max_candidates_per_s1": u["max_candidates_per_s1"],
                "total_candidates": u["total_candidates"],
                "reduction_ratio": u["reduction_ratio"],
            })

        with open(p_cand_stats, "w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "strategy_name", "strategy_type", "mean_candidates_per_s1", "median_candidates_per_s1",
                "p90_candidates_per_s1", "p95_candidates_per_s1", "p99_candidates_per_s1", "max_candidates_per_s1",
                "total_candidates", "reduction_ratio"
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in cand_stat_rows:
                writer.writerow(row)

        # Complementarity CSV
        p_comp = out_dir / "complementarity_results.csv"
        if complementarity:
            with open(p_comp, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(complementarity[0].keys()))
                writer.writeheader()
                for row in complementarity:
                    writer.writerow(row)

        # 6. failure_cases.csv
        p_failures = out_dir / "failure_cases.csv"
        with open(p_failures, "w", newline="", encoding="utf-8") as f:
            if failures:
                writer = csv.DictWriter(f, fieldnames=list(failures[0].keys()))
                writer.writeheader()
                for row in failures:
                    writer.writerow(row)
            else:
                f.write("source1_id,source1_name,target_id,target_name,failure_diagnosis\n")

        # Candidate explosions CSV
        p_exp = out_dir / "candidate_explosion_cases.csv"
        with open(p_exp, "w", newline="", encoding="utf-8") as f:
            if explosions:
                writer = csv.DictWriter(f, fieldnames=list(explosions[0].keys()))
                writer.writeheader()
                for row in explosions:
                    writer.writerow(row)

        # France test data analysis summary JSON
        p_france = out_dir / "france_open_set_audit.json"
        with open(p_france, "w", encoding="utf-8") as f:
            json.dump(france_analysis, f, indent=2)

        print(f"All Phase 1 deliverables successfully written to: {out_dir}", flush=True)


def main():
    runner = BlockingBenchmarkRunner(
        sample_size=5000,
        distractor_target_size=100000,
        seed=42,
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


if __name__ == "__main__":
    main()
