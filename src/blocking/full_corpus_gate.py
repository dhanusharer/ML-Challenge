"""Phase 1.6: Full Target-Corpus Feasibility Gate.

Compares U6 candidate generation across three target universes:
- Universe A: Phase 1 Target Universe (~117K records)
- Universe B: Phase 1.5 Target Universe (~496K records)
- Universe C: Full 10.32M Target Corpus (10,320,219 records from train_source2 + train_source3)

Measures the exact empirical degradation curve:
96.54% (at 117K) -> 93.85% (at 496K) -> ?? (at 10.32M)
along with candidate volume, P95/P99/max percentiles, index build time, throughput, and peak RAM.
"""

from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple
import csv
import heapq
import json
import numpy as np
import time
import psutil
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from src.utils.env import PROJECT_ROOT, resolve_train_dir
from src.data.parser import parse_ground_truth_file
from src.representations.normalizer import standard_clean
from src.representations.name import (
    strip_legal_suffixes,
    get_sorted_tokens_name,
    get_name_tokens,
    get_char_ngrams,
)
from src.representations.address import extract_postal_codes, extract_numeric_tokens
from src.blocking.u6_ensemble import U6EnsembleBlocker


class FullCorpusGateRunner:
    """Orchestrates Phase 1.6 comparison across Universes A, B, and C."""

    def __init__(
        self,
        query_cohort_size: int = 10000,
        seed: int = 42,
    ):
        self.query_cohort_size = query_cohort_size
        self.seed = seed
        self.train_dir = resolve_train_dir()

        self.val_s1_ids: List[str] = []
        self.ground_truth: Dict[str, List[str]] = {}
        self.query_records: List[Dict[str, str]] = []
        self.required_target_ids: Set[str] = set()

    def load_queries(self) -> None:
        """Load frozen validation S1 queries and ground truth."""
        print("=" * 70, flush=True)
        print("PHASE 1.6: LOADING FROZEN QUERIES & GROUND TRUTH", flush=True)
        print("=" * 70, flush=True)

        gt_path = self.train_dir / "train_ground_truth.tsv"
        self.ground_truth = parse_ground_truth_file(gt_path)

        split_summary = PROJECT_ROOT / "reports" / "phase0" / "validation_split_summary.json"
        if split_summary.is_file():
            with open(split_summary, "r", encoding="utf-8") as f:
                meta = json.load(f)
            from src.validation.split import create_entity_validation_split
            s1_all = list(self.ground_truth.keys())
            split = create_entity_validation_split(
                s1_all,
                val_fraction=meta["split_metadata"]["validation_fraction"],
                seed=meta["split_metadata"]["validation_seed"],
            )
            self.val_s1_ids = split.val_s1_ids
        else:
            from src.validation.split import create_entity_validation_split
            s1_all = list(self.ground_truth.keys())
            split = create_entity_validation_split(s1_all, val_fraction=0.20, seed=42)
            self.val_s1_ids = split.val_s1_ids

        rng = np.random.RandomState(self.seed)
        shuffled = rng.permutation(self.val_s1_ids)
        selected_s1_ids = set(shuffled[:self.query_cohort_size])
        selected_order = list(shuffled[:self.query_cohort_size])

        for s1_id in selected_s1_ids:
            for mid in self.ground_truth.get(s1_id, []):
                self.required_target_ids.add(mid)

        print(f"Validation cohort: {len(selected_s1_ids):,} S1 queries with {len(self.required_target_ids):,} true links.", flush=True)

        s1_file = self.train_dir / "train_source1.tsv"
        q_dict: Dict[str, Dict[str, str]] = {}
        with open(s1_file, "r", encoding="utf-8") as f:
            next(f)
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) >= 4 and parts[0] in selected_s1_ids:
                    q_dict[parts[0]] = {
                        "entity_id": parts[0],
                        "business_name": parts[1],
                        "business_address": parts[2],
                        "country": parts[3],
                    }

        self.query_records = [q_dict[sid] for sid in selected_order if sid in q_dict]
        print(f"Loaded {len(self.query_records):,} query records in memory.", flush=True)

    def evaluate_universe_a_b(
        self,
        distractor_count: int,
        universe_name: str,
    ) -> Dict[str, Any]:
        """Evaluate U6 on Target Universe A (~117K) or B (~496K)."""
        print(f"\n--- Benchmarking {universe_name} ---", flush=True)
        t_start = time.time()
        process = psutil.Process()
        m0 = process.memory_info().rss / (1024 * 1024)

        targets_dict: Dict[str, Dict[str, str]] = {}
        distractors_loaded = 0
        max_distractors_per_source = distractor_count // 2

        for filename in ["train_source2.tsv", "train_source3.tsv"]:
            p = self.train_dir / filename
            with open(p, "r", encoding="utf-8") as f:
                next(f)
                src_distractors = 0
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
                    if eid in self.required_target_ids:
                        targets_dict[eid] = rec
                    elif src_distractors < max_distractors_per_source:
                        targets_dict[eid] = rec
                        src_distractors += 1
                        distractors_loaded += 1

        target_records = list(targets_dict.values())
        print(f"  Target count: {len(target_records):,} ({len(self.required_target_ids):,} true + {distractors_loaded:,} distractors)", flush=True)

        t_fit_start = time.time()
        blocker = U6EnsembleBlocker(tfidf_top_k=25, max_tfidf_features=15000, max_bucket_size=500)
        blocker.fit(target_records)
        fit_time = time.time() - t_fit_start

        res = blocker.evaluate_cohort(
            queries=self.query_records,
            ground_truth=self.ground_truth,
            batch_size=250,
        )

        peak_ram = process.memory_info().rss / (1024 * 1024)
        total_time = time.time() - t_start

        res["universe_name"] = universe_name
        res["target_count"] = len(target_records)
        res["fit_time_seconds"] = fit_time
        res["total_time_seconds"] = total_time
        res["peak_ram_mb"] = peak_ram

        print(
            f"  Recall: {res['blocking_recall']:.4f} (S2: {res['s2_recall']:.4f}, S3: {res['s3_recall']:.4f}) | "
            f"Mean Cands: {res['mean_candidates']:.2f} | P95: {res['p95_candidates']:.0f} | Max: {res['max_candidates']} | "
            f"Fit: {fit_time:.2f}s | Query: {res['runtime_seconds']:.2f}s | Throughput: {res['throughput_queries_per_sec']:.0f} q/s | RAM: {peak_ram:.1f} MB",
            flush=True
        )
        return res

    def evaluate_universe_c_full(self) -> Dict[str, Any]:
        """Evaluate U6 on the FULL 10.32M target corpus using streaming query-side indexing + chunked TF-IDF."""
        print(f"\n--- Benchmarking Universe C: FULL 10.32M TARGET CORPUS ---", flush=True)
        t_start = time.time()
        process = psutil.Process()
        m0 = process.memory_info().rss / (1024 * 1024)

        n_queries = len(self.query_records)

        # 1. Build Query-Side Deterministic Inverted Indexes
        print("  [Step 1/4] Building query-side deterministic inverted indexes...", flush=True)
        t_index_start = time.time()
        query_exact: Dict[str, List[int]] = defaultdict(list)
        query_sorted: Dict[str, List[int]] = defaultdict(list)
        query_numeric: Dict[str, List[int]] = defaultdict(list)
        query_char: Dict[str, List[int]] = defaultdict(list)

        query_texts: List[str] = []

        for q_idx, q in enumerate(self.query_records):
            raw_name = q.get("business_name", "")
            raw_addr = q.get("business_address", "")
            clean_name = strip_legal_suffixes(raw_name)

            if clean_name:
                query_exact[clean_name].append(q_idx)

            sorted_name = get_sorted_tokens_name(raw_name, strip_legal=True)
            if sorted_name:
                query_sorted[sorted_name].append(q_idx)

            tokens = get_name_tokens(clean_name)
            if tokens and len(tokens[0]) >= 3:
                first_tok = tokens[0]
                nums = extract_numeric_tokens(raw_addr)
                for num in nums:
                    query_numeric[f"{first_tok}::{num}"].append(q_idx)

            if clean_name:
                compact = clean_name.replace(" ", "")
                if len(compact) < 4:
                    sig = compact
                else:
                    sig = f"{compact[:4]}::{compact[-4:]}"
                if sig:
                    query_char[sig].append(q_idx)

            query_texts.append(f"{clean_name} {standard_clean(raw_addr)}".strip())

        # Candidates set per query
        candidates: List[Set[str]] = [set() for _ in range(n_queries)]
        max_cands_per_query = 500

        # 2. Fit TF-IDF Vocabulary on a representative 500K target sample
        print("  [Step 2/4] Fitting TF-IDF vocabulary on representative target sample...", flush=True)
        vocab_samples: List[str] = []
        for filename in ["train_source2.tsv", "train_source3.tsv"]:
            p = self.train_dir / filename
            with open(p, "r", encoding="utf-8") as f:
                next(f)
                for i, line in enumerate(f):
                    if i >= 250000:
                        break
                    parts = line.split("\t")
                    if len(parts) >= 3:
                        name = strip_legal_suffixes(parts[1])
                        addr = standard_clean(parts[2])
                        vocab_samples.append(f"{name} {addr}".strip())

        vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=5,
            max_df=0.20,
            max_features=15000,
            sublinear_tf=True,
            dtype=np.float32,
        )
        vectorizer.fit(vocab_samples)
        del vocab_samples
        print(f"  Vocabulary fitted ({len(vectorizer.vocabulary_):,} features).", flush=True)

        # Precompute query matrix Q
        Q = vectorizer.transform(query_texts)  # (n_queries, n_features)

        # Min-heaps for running top-25 TF-IDF: heap per query containing (score, eid)
        top_k = 25
        tfidf_heaps: List[List[Tuple[float, str]]] = [[] for _ in range(n_queries)]

        index_build_time = time.time() - t_index_start
        print(f"  Index prep and vocabulary fitted in {index_build_time:.2f}s.", flush=True)

        # 3. Stream all 10.32M target records through deterministic matchers and chunked TF-IDF
        print("  [Step 3/4] Streaming through full 10.32M target corpus...", flush=True)
        t_query_start = time.time()

        total_targets_streamed = 0
        chunk_size = 500000
        chunk_texts: List[str] = []
        chunk_eids: List[str] = []

        for filename in ["train_source2.tsv", "train_source3.tsv"]:
            p = self.train_dir / filename
            with open(p, "r", encoding="utf-8") as f:
                next(f)
                for line in f:
                    parts = line.rstrip("\r\n").split("\t")
                    if len(parts) < 3:
                        continue
                    eid = parts[0]
                    raw_name = parts[1]
                    raw_addr = parts[2]
                    total_targets_streamed += 1

                    # Deterministic branch
                    clean_name = strip_legal_suffixes(raw_name)
                    if clean_name in query_exact:
                        for q_idx in query_exact[clean_name]:
                            if len(candidates[q_idx]) < max_cands_per_query:
                                candidates[q_idx].add(eid)

                    sorted_name = get_sorted_tokens_name(raw_name, strip_legal=True)
                    if sorted_name in query_sorted:
                        for q_idx in query_sorted[sorted_name]:
                            if len(candidates[q_idx]) < max_cands_per_query:
                                candidates[q_idx].add(eid)

                    tokens = get_name_tokens(clean_name)
                    if tokens and len(tokens[0]) >= 3:
                        first_tok = tokens[0]
                        nums = extract_numeric_tokens(raw_addr)
                        for num in nums:
                            k = f"{first_tok}::{num}"
                            if k in query_numeric:
                                for q_idx in query_numeric[k]:
                                    if len(candidates[q_idx]) < max_cands_per_query:
                                        candidates[q_idx].add(eid)

                    if clean_name:
                        compact = clean_name.replace(" ", "")
                        if len(compact) < 4:
                            sig = compact
                        else:
                            sig = f"{compact[:4]}::{compact[-4:]}"
                        if sig in query_char:
                            for q_idx in query_char[sig]:
                                if len(candidates[q_idx]) < max_cands_per_query:
                                    candidates[q_idx].add(eid)

                    # Accumulate for chunked TF-IDF
                    chunk_texts.append(f"{clean_name} {standard_clean(raw_addr)}".strip())
                    chunk_eids.append(eid)

                    # Process chunk when full
                    if len(chunk_texts) >= chunk_size:
                        self._process_tfidf_chunk(Q, vectorizer, chunk_texts, chunk_eids, tfidf_heaps, top_k)
                        chunk_texts = []
                        chunk_eids = []
                        print(f"    Streamed {total_targets_streamed:,} / 10,320,219 records...", flush=True)

        # Process final remaining chunk
        if chunk_texts:
            self._process_tfidf_chunk(Q, vectorizer, chunk_texts, chunk_eids, tfidf_heaps, top_k)
            chunk_texts = []
            chunk_eids = []

        query_time = time.time() - t_query_start
        print(f"  Streaming complete: {total_targets_streamed:,} records processed in {query_time:.2f}s.", flush=True)

        # 4. Union deterministic candidates and top-25 TF-IDF candidates
        print("  [Step 4/4] Merging candidate unions and computing true-link recall...", flush=True)
        total_true_links = 0
        captured_true_links = 0
        s2_true_links = 0
        s2_captured = 0
        s3_true_links = 0
        s3_captured = 0
        candidate_sizes: List[int] = []

        for q_idx, q in enumerate(self.query_records):
            s1_id = q["entity_id"]
            # Add top-25 TF-IDF candidates
            for score, mid in tfidf_heaps[q_idx]:
                candidates[q_idx].add(mid)

            c_set = candidates[q_idx]
            candidate_sizes.append(len(c_set))

            true_links = self.ground_truth.get(s1_id, [])
            for mid in true_links:
                total_true_links += 1
                is_s2 = mid.startswith("S2-")
                is_s3 = mid.startswith("S3-")

                if is_s2:
                    s2_true_links += 1
                elif is_s3:
                    s3_true_links += 1

                if mid in c_set:
                    captured_true_links += 1
                    if is_s2:
                        s2_captured += 1
                    elif is_s3:
                        s3_captured += 1

        peak_ram = process.memory_info().rss / (1024 * 1024)
        total_time = time.time() - t_start
        throughput = n_queries / query_time if query_time > 0 else 0.0

        c_arr = np.array(candidate_sizes, dtype=np.int32)
        cartesian_space = n_queries * total_targets_streamed
        total_pairs = int(c_arr.sum())
        reduction = 1.0 - (total_pairs / cartesian_space) if cartesian_space > 0 else 1.0

        res = {
            "universe_name": "Universe C (Full 10.32M Target Corpus)",
            "target_count": total_targets_streamed,
            "total_queries": n_queries,
            "total_true_links": total_true_links,
            "captured_true_links": captured_true_links,
            "blocking_recall": captured_true_links / total_true_links if total_true_links > 0 else 0.0,
            "s2_true_links": s2_true_links,
            "s2_captured_links": s2_captured,
            "s2_recall": s2_captured / s2_true_links if s2_true_links > 0 else 0.0,
            "s3_true_links": s3_true_links,
            "s3_captured_links": s3_captured,
            "s3_recall": s3_captured / s3_true_links if s3_true_links > 0 else 0.0,
            "total_candidates": total_pairs,
            "mean_candidates": float(np.mean(c_arr)),
            "median_candidates": float(np.median(c_arr)),
            "p90_candidates": float(np.percentile(c_arr, 90)),
            "p95_candidates": float(np.percentile(c_arr, 95)),
            "p99_candidates": float(np.percentile(c_arr, 99)),
            "max_candidates": int(np.max(c_arr)),
            "reduction_ratio": reduction,
            "fit_time_seconds": index_build_time,
            "runtime_seconds": query_time,
            "total_time_seconds": total_time,
            "throughput_queries_per_sec": throughput,
            "peak_ram_mb": peak_ram,
        }

        print(
            f"  Recall: {res['blocking_recall']:.4f} (S2: {res['s2_recall']:.4f}, S3: {res['s3_recall']:.4f}) | "
            f"Mean Cands: {res['mean_candidates']:.2f} | P95: {res['p95_candidates']:.0f} | Max: {res['max_candidates']} | "
            f"Fit: {index_build_time:.2f}s | Stream: {query_time:.2f}s | Throughput: {throughput:.0f} q/s | RAM: {peak_ram:.1f} MB",
            flush=True
        )
        return res

    def _process_tfidf_chunk(
        self,
        Q: csr_matrix,
        vectorizer: TfidfVectorizer,
        chunk_texts: List[str],
        chunk_eids: List[str],
        tfidf_heaps: List[List[Tuple[float, str]]],
        top_k: int,
    ) -> None:
        """Vectorize a chunk of targets and update running top-k heaps for queries."""
        X_chunk = vectorizer.transform(chunk_texts)
        # Compute dot product: Q (n_queries, D) x X_chunk.T (D, n_chunk) -> (n_queries, n_chunk)
        scores_mat = Q.dot(X_chunk.T)

        for q_idx in range(Q.shape[0]):
            row = scores_mat.getrow(q_idx)
            if row.nnz == 0:
                continue

            heap = tfidf_heaps[q_idx]

            # If non-zeros are small, inspect all
            if row.nnz <= top_k:
                for col_idx, s in zip(row.indices, row.data):
                    target_eid = chunk_eids[col_idx]
                    if len(heap) < top_k:
                        heapq.heappush(heap, (float(s), target_eid))
                    elif s > heap[0][0]:
                        heapq.heappushpop(heap, (float(s), target_eid))
            else:
                # Top k in chunk
                part = np.argpartition(row.data, -top_k)[-top_k:]
                for p_idx in part:
                    s = float(row.data[p_idx])
                    target_eid = chunk_eids[row.indices[p_idx]]
                    if len(heap) < top_k:
                        heapq.heappush(heap, (s, target_eid))
                    elif s > heap[0][0]:
                        heapq.heappushpop(heap, (s, target_eid))

    def run_all_comparisons(self) -> List[Dict[str, Any]]:
        """Run full three-universe comparison on identical query cohort."""
        self.load_queries()

        # Universe A (~117K)
        res_a = self.evaluate_universe_a_b(distractor_count=100000, universe_name="Universe A (Phase 1 Universe, ~117K Targets)")

        # Universe B (~496K)
        res_b = self.evaluate_universe_a_b(distractor_count=450000, universe_name="Universe B (Phase 1.5 Universe, ~496K Targets)")

        # Universe C (Full 10.32M)
        res_c = self.evaluate_universe_c_full()

        return [res_a, res_b, res_c]

    def export_deliverables(self, results: List[Dict[str, Any]]) -> None:
        """Export Phase 1.6 deliverables."""
        out_dir = PROJECT_ROOT / "reports" / "phase1_6"
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1. universe_comparison.csv
        p_csv = out_dir / "universe_comparison.csv"
        fields = [
            "universe_name", "target_count", "total_queries", "total_true_links", "captured_true_links",
            "blocking_recall", "s2_recall", "s3_recall", "mean_candidates", "median_candidates",
            "p90_candidates", "p95_candidates", "p99_candidates", "max_candidates", "reduction_ratio",
            "fit_time_seconds", "runtime_seconds", "total_time_seconds", "throughput_queries_per_sec", "peak_ram_mb"
        ]

        with open(p_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for r in results:
                row = {k: r.get(k) for k in fields}
                writer.writerow(row)

        # 2. phase1_6_summary.json
        p_json = out_dir / "phase1_6_summary.json"
        summary_payload = {
            "gate_name": "PHASE_1_6_FULL_TARGET_CORPUS_GATE",
            "query_cohort_size": self.query_cohort_size,
            "target_universes": results,
            "conclusions": {
                "degradation_a_to_b": results[0]["blocking_recall"] - results[1]["blocking_recall"],
                "degradation_b_to_c": results[1]["blocking_recall"] - results[2]["blocking_recall"],
                "total_degradation_a_to_c": results[0]["blocking_recall"] - results[2]["blocking_recall"],
                "full_corpus_recall": results[2]["blocking_recall"],
                "full_corpus_s2_recall": results[2]["s2_recall"],
                "full_corpus_s3_recall": results[2]["s3_recall"],
                "full_corpus_mean_candidates": results[2]["mean_candidates"],
                "full_corpus_p95_candidates": results[2]["p95_candidates"],
                "full_corpus_max_candidates": results[2]["max_candidates"],
            }
        }
        with open(p_json, "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, indent=2)

        print(f"\nPhase 1.6 deliverables successfully exported to: {out_dir}", flush=True)


def main():
    runner = FullCorpusGateRunner(query_cohort_size=10000, seed=42)
    results = runner.run_all_comparisons()
    runner.export_deliverables(results)


if __name__ == "__main__":
    main()
