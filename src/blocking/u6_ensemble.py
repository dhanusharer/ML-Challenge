"""U6 Multi-Blocker Candidate Generation Ensemble.

Integrated, high-throughput implementation of the Phase 1 U6 ensemble:
- Exact Clean Name (Legal Stripped)
- Sorted Tokens Name
- Name Token + Postal Code Hybrid
- Name Token + Numeric Address Hybrid
- Character Boundary 4-Grams
- Sparse Word TF-IDF Top-25 Retrieval

Optimized for scale safety, low memory overhead, and batched streaming evaluation.
"""

from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import time
import psutil

from src.representations.normalizer import standard_clean
from src.representations.name import (
    strip_legal_suffixes,
    get_sorted_tokens_name,
    get_name_tokens,
    get_char_ngrams,
)
from src.representations.address import extract_postal_codes, extract_numeric_tokens


class U6EnsembleBlocker:
    """Integrated high-scale U6 candidate generator."""

    def __init__(
        self,
        tfidf_top_k: int = 25,
        max_tfidf_features: int = 15000,
        max_bucket_size: int = 500,
    ):
        self.tfidf_top_k = tfidf_top_k
        self.max_tfidf_features = max_tfidf_features
        self.max_bucket_size = max_bucket_size

        # Inverted index mappings: key -> list of target integer indices
        self.index_exact: Dict[str, List[int]] = defaultdict(list)
        self.index_sorted: Dict[str, List[int]] = defaultdict(list)
        self.index_postal: Dict[str, List[int]] = defaultdict(list)
        self.index_numeric: Dict[str, List[int]] = defaultdict(list)
        self.index_char: Dict[str, List[int]] = defaultdict(list)

        self.vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.20,
            max_features=max_tfidf_features,
            sublinear_tf=True,
            dtype=np.float32,
        )
        self.target_matrix: Optional[csr_matrix] = None
        self.target_matrix_t: Optional[csr_matrix] = None
        self.target_ids: List[str] = []
        self.is_fitted: bool = False

    def _prepare_text(self, name_raw: str, addr_raw: str) -> str:
        name = strip_legal_suffixes(name_raw)
        addr = standard_clean(addr_raw)
        return f"{name} {addr}".strip()

    def fit(self, targets: Sequence[Dict[str, str]]) -> "U6EnsembleBlocker":
        """Index all target records across deterministic inverted indexes and TF-IDF matrix."""
        t0 = time.time()
        self.target_ids = [t["entity_id"] for t in targets]
        n_targets = len(targets)

        target_texts: List[str] = []

        for idx, t in enumerate(targets):
            name_raw = t.get("business_name", "")
            addr_raw = t.get("business_address", "")

            # 1. Exact clean
            clean_name = strip_legal_suffixes(name_raw)
            if clean_name:
                self.index_exact[clean_name].append(idx)

            # 2. Sorted tokens
            sorted_name = get_sorted_tokens_name(name_raw, strip_legal=True)
            if sorted_name:
                self.index_sorted[sorted_name].append(idx)

            # 3 & 4. Hybrids
            tokens = get_name_tokens(clean_name)
            if tokens and len(tokens[0]) >= 3:
                first_tok = tokens[0]

                # Postal hybrid
                postals = extract_postal_codes(addr_raw)
                for p in postals:
                    self.index_postal[f"{first_tok}::{p}"].append(idx)

                # Numeric hybrid
                nums = extract_numeric_tokens(addr_raw)
                for num in nums:
                    self.index_numeric[f"{first_tok}::{num}"].append(idx)

            # 5. Char boundary compound 4-gram (first::last)
            if clean_name:
                compact = clean_name.replace(" ", "")
                if len(compact) < 4:
                    sig = compact
                else:
                    sig = f"{compact[:4]}::{compact[-4:]}"
                if sig:
                    self.index_char[sig].append(idx)

            # 6. TF-IDF text
            target_texts.append(self._prepare_text(name_raw, addr_raw))

        # Apply bucket caps to prevent candidate explosion
        for idx_dict in [self.index_postal, self.index_numeric, self.index_char]:
            keys_to_cap = [k for k, v in idx_dict.items() if len(v) > self.max_bucket_size]
            for k in keys_to_cap:
                idx_dict[k] = idx_dict[k][:self.max_bucket_size]

        # Fit TF-IDF matrix (dynamically adapt min_df/max_df if dataset is tiny, e.g. in tests)
        if n_targets < 50:
            self.vectorizer.min_df = 1
            self.vectorizer.max_df = 1.0

        self.target_matrix = self.vectorizer.fit_transform(target_texts)
        self.target_matrix_t = self.target_matrix.T.tocsr()
        self.is_fitted = True

        fit_time = time.time() - t0
        print(f"U6EnsembleBlocker fitted on {n_targets:,} target records in {fit_time:.2f}s.", flush=True)
        return self

    def retrieve_batch_candidates(
        self,
        query_batch: Sequence[Dict[str, str]],
    ) -> List[Set[str]]:
        """Retrieve candidate ID sets for a batch of query records."""
        if not self.is_fitted or self.target_matrix_t is None:
            raise RuntimeError("U6EnsembleBlocker must be fitted before query retrieval.")

        batch_size = len(query_batch)
        results: List[Set[int]] = [set() for _ in range(batch_size)]

        # 1-5. Inverted index retrieval
        query_texts: List[str] = []
        for i, q in enumerate(query_batch):
            name_raw = q.get("business_name", "")
            addr_raw = q.get("business_address", "")
            clean_name = strip_legal_suffixes(name_raw)

            # Exact clean
            if clean_name in self.index_exact:
                results[i].update(self.index_exact[clean_name])

            # Sorted tokens
            sorted_name = get_sorted_tokens_name(name_raw, strip_legal=True)
            if sorted_name in self.index_sorted:
                results[i].update(self.index_sorted[sorted_name])

            # Hybrids
            tokens = get_name_tokens(clean_name)
            if tokens and len(tokens[0]) >= 3:
                first_tok = tokens[0]

                # Postal hybrid
                postals = extract_postal_codes(addr_raw)
                for p in postals:
                    k = f"{first_tok}::{p}"
                    if k in self.index_postal:
                        results[i].update(self.index_postal[k])

                # Numeric hybrid
                nums = extract_numeric_tokens(addr_raw)
                for num in nums:
                    k = f"{first_tok}::{num}"
                    if k in self.index_numeric:
                        results[i].update(self.index_numeric[k])

            # Boundary compound 4-grams (first::last)
            if clean_name:
                compact = clean_name.replace(" ", "")
                if len(compact) < 4:
                    sig = compact
                else:
                    sig = f"{compact[:4]}::{compact[-4:]}"
                if sig and sig in self.index_char:
                    results[i].update(self.index_char[sig])

            query_texts.append(self._prepare_text(name_raw, addr_raw))

        # 6. Batched TF-IDF top-k retrieval
        if self.tfidf_top_k > 0:
            query_mat = self.vectorizer.transform(query_texts)
            # Batched sparse dot product: shape (batch_size, n_targets)
            scores = query_mat.dot(self.target_matrix_t)
            n_targets = self.target_matrix.shape[0]
            k = min(self.tfidf_top_k, n_targets)

            for i in range(batch_size):
                row = scores.getrow(i)
                if row.nnz == 0:
                    continue
                # If nonzeros <= k, add all nonzeros
                if row.nnz <= k:
                    results[i].update(row.indices)
                else:
                    # Partition top k
                    part = np.argpartition(row.data, -k)[-k:]
                    top_indices = row.indices[part]
                    results[i].update(top_indices)

        # Convert target integer indices to string entity IDs
        id_results: List[Set[str]] = []
        for cand_set in results:
            id_results.append({self.target_ids[idx] for idx in cand_set})

        return id_results

    def evaluate_cohort(
        self,
        queries: Sequence[Dict[str, str]],
        ground_truth: Dict[str, Sequence[str]],
        batch_size: int = 250,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Dict[str, Any]:
        """Stream queries in batches and measure recall, candidate volume, and memory."""
        t0 = time.time()
        process = psutil.Process()
        initial_ram_mb = process.memory_info().rss / (1024 * 1024)
        peak_ram_mb = initial_ram_mb

        total_queries = len(queries)
        total_true_links = 0
        captured_true_links = 0
        s2_true_links = 0
        s2_captured = 0
        s3_true_links = 0
        s3_captured = 0

        candidate_sizes: List[int] = []
        failed_link_examples: List[Dict[str, Any]] = []

        for b_start in range(0, total_queries, batch_size):
            b_end = min(b_start + batch_size, total_queries)
            q_batch = queries[b_start:b_end]

            cand_sets = self.retrieve_batch_candidates(q_batch)

            for i, q in enumerate(q_batch):
                s1_id = q["entity_id"]
                true_links = ground_truth.get(s1_id, [])
                cands = cand_sets[i]

                candidate_sizes.append(len(cands))

                for mid in true_links:
                    total_true_links += 1
                    is_s2 = mid.startswith("S2-")
                    is_s3 = mid.startswith("S3-")

                    if is_s2:
                        s2_true_links += 1
                    elif is_s3:
                        s3_true_links += 1

                    if mid in cands:
                        captured_true_links += 1
                        if is_s2:
                            s2_captured += 1
                        elif is_s3:
                            s3_captured += 1
                    elif len(failed_link_examples) < 100:
                        failed_link_examples.append({
                            "s1_id": s1_id,
                            "s1_name": q.get("business_name", ""),
                            "s1_address": q.get("business_address", ""),
                            "target_id": mid,
                            "country": q.get("country", ""),
                        })

            current_ram = process.memory_info().rss / (1024 * 1024)
            if current_ram > peak_ram_mb:
                peak_ram_mb = current_ram

            if progress_callback:
                progress_callback(b_end, total_queries)

        total_time = time.time() - t0
        throughput = total_queries / total_time if total_time > 0 else 0.0

        c_arr = np.array(candidate_sizes, dtype=np.int32)
        total_target_records = len(self.target_ids)
        total_pairs = int(c_arr.sum())
        cartesian_space = total_queries * total_target_records
        reduction = 1.0 - (total_pairs / cartesian_space) if cartesian_space > 0 else 1.0

        return {
            "total_queries": total_queries,
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
            "runtime_seconds": total_time,
            "throughput_queries_per_sec": throughput,
            "peak_ram_mb": peak_ram_mb,
            "failures_sampled": failed_link_examples,
        }
