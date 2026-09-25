"""Phase 1.7: Full-Corpus Recall Recovery Lab.

Investigates six targeted candidate recovery mechanisms on the full 10.32M target corpus:
- Experiment A: TF-IDF top-k scaling (k = 25, 40, 50, 75, 100)
- Experiment B: Frequency-aware retrieval & bucket-cap prevention
- Experiment C: Cross-script recovery (address fallback + offline Indic transliteration)
- Experiment D: URL / domain normalization
- Experiment E: Controlled leetspeak / character substitution
- Experiment F: Complete alias cases via secondary address numeric structure

Evaluates individual recall contributions, candidate volume distributions,
percentiles (P95, P99, max), runtimes, and peak RAM on the full 10.32M corpus.
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
import re
import unicodedata
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
    LEGAL_SUFFIXES,
)
from src.representations.address import extract_postal_codes, extract_numeric_tokens
from src.representations.domain import clean_domain_name, get_compact_domain_signature
from src.representations.transliteration import (
    has_indic_script,
    transliterate_indic_to_latin,
    get_indic_transliterated_signature,
)
from src.representations.leetspeak import (
    strip_accents,
    normalize_leetspeak_text,
    get_leetspeak_signature,
)

# Generic business tokens that cause distractor explosion / bucket capping
GENERIC_BUSINESS_TOKENS: Set[str] = {
    "enterprises", "enterprise", "solutions", "solution", "services", "service",
    "industries", "industry", "holdings", "holding", "group", "groups",
    "international", "intl", "global", "national", "standard", "general",
    "products", "product", "trading", "commercial", "technologies", "technology",
    "consulting", "consultancy", "management", "logistics", "associates", "associate",
    "india", "indian", "corporation", "corp", "company", "co", "limited", "ltd",
    "private", "pvt", "inc", "llc", "llp", "center", "centre", "ventures",
    "systems", "system", "traders", "agency", "agencies", "impex"
}


def get_informative_name_tokens(raw_name: str) -> List[str]:
    """Extract name tokens filtered of generic stopwords and legal suffixes."""
    cleaned = strip_legal_suffixes(raw_name)
    tokens = [t for t in cleaned.split() if t]
    informative = [t for t in tokens if t not in GENERIC_BUSINESS_TOKENS and len(t) >= 3]
    return informative if informative else tokens


def extract_address_numeric_compounds(raw_address: Optional[str]) -> List[str]:
    """Extract distinct address numbers and alphanumeric plot/building codes."""
    if not raw_address:
        return []
    # Plot numbers like 70C/4G, 1002C, H62/03, P-88
    tokens = re.findall(r"\b\d+[a-zA-Z]?(?:[/-]\d+[a-zA-Z]?)*\b", raw_address)
    compounds = re.findall(r"\b[A-Za-z]\d+\b", raw_address)
    all_num = [t.lower() for t in tokens + compounds]
    # Filter out trivial 1-digit numbers like '1', '2' unless compound
    distinct = [n for n in all_num if len(n) >= 2 or "/" in n or "-" in n]
    return distinct if distinct else all_num


class RecoveryLabRunner:
    """Orchestrates Phase 1.7 experiments on the full 10.32M target corpus."""

    def __init__(
        self,
        query_cohort_size: int = 1000,
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
        print("=" * 75, flush=True)
        print("PHASE 1.7: LOADING FROZEN VALIDATION COHORT & GROUND TRUTH", flush=True)
        print("=" * 75, flush=True)

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

    def run_full_recovery_benchmark(self) -> Dict[str, Any]:
        """Execute unified streaming pass over the full 10.32M target corpus."""
        t_start = time.time()
        process = psutil.Process()
        m0 = process.memory_info().rss / (1024 * 1024)

        n_queries = len(self.query_records)

        # -----------------------------------------------------------------
        # Step 1: Build Query-Side Inverted Indexes for All Experiments
        # -----------------------------------------------------------------
        print("\n[Step 1/4] Building query-side indexes for Experiments A-F...", flush=True)
        t_index_start = time.time()

        # Deterministic Baseline U6 indexes
        q_exact: Dict[str, List[int]] = defaultdict(list)
        q_sorted: Dict[str, List[int]] = defaultdict(list)
        q_numeric_base: Dict[str, List[int]] = defaultdict(list)
        q_char: Dict[str, List[int]] = defaultdict(list)

        # Experiment B: Frequency-aware compound keys (informative token + numeric)
        q_numeric_freq_aware: Dict[str, List[int]] = defaultdict(list)

        # Experiment C: Cross-script (Transliteration + Indic address numeric fallback)
        q_indic_translit: Dict[str, List[int]] = defaultdict(list)
        q_indic_address_numeric: Dict[str, List[int]] = defaultdict(list)

        # Experiment D: Domain / URL compact signature
        q_domain_compact: Dict[str, List[int]] = defaultdict(list)

        # Experiment E: Leetspeak / Accent signature
        q_leetspeak_compact: Dict[str, List[int]] = defaultdict(list)

        # Experiment F: Complete alias secondary address compound keys
        q_alias_address: Dict[str, List[int]] = defaultdict(list)

        query_texts: List[str] = []

        for q_idx, q in enumerate(self.query_records):
            raw_name = q.get("business_name", "")
            raw_addr = q.get("business_address", "")
            country = q.get("country", "").upper()
            clean_name = strip_legal_suffixes(raw_name)

            # Baseline U6 Keys
            if clean_name:
                q_exact[clean_name].append(q_idx)
            sorted_name = get_sorted_tokens_name(raw_name, strip_legal=True)
            if sorted_name:
                q_sorted[sorted_name].append(q_idx)

            base_tokens = get_name_tokens(clean_name)
            base_nums = extract_numeric_tokens(raw_addr)
            if base_tokens and len(base_tokens[0]) >= 3:
                for num in base_nums:
                    q_numeric_base[f"{base_tokens[0]}::{num}"].append(q_idx)

            if clean_name:
                compact = clean_name.replace(" ", "")
                sig = compact if len(compact) < 4 else f"{compact[:4]}::{compact[-4:]}"
                if sig:
                    q_char[sig].append(q_idx)

            # Exp B: Frequency-aware compound keys
            inf_tokens = get_informative_name_tokens(raw_name)
            distinct_nums = extract_address_numeric_compounds(raw_addr)
            if inf_tokens:
                chosen_token = inf_tokens[0]
                for num in distinct_nums:
                    q_numeric_freq_aware[f"{chosen_token}::{num}"].append(q_idx)
                # Also index second informative token if available for high selectivity
                if len(inf_tokens) > 1:
                    q_numeric_freq_aware[f"{inf_tokens[1]}::{distinct_nums[0]}"].append(q_idx) if distinct_nums else None

            # Exp C: Cross-script representations
            if clean_name:
                # S1 name is Latin; query translit signature is the Latin compact root
                latin_root = get_compact_domain_signature(clean_name, min_length=4)
                if latin_root:
                    q_indic_translit[latin_root[:5]].append(q_idx)  # Prefix matching

            # Indic address fallback: When country is India or address has Indian state/city
            if country == "INDIA" or "india" in raw_addr.lower() or any(st in raw_addr.lower() for st in ["maharashtra", "delhi", "karnataka", "tamil nadu", "bengal", "gujarat", "telangana", "uttar pradesh"]):
                for num in distinct_nums:
                    if len(num) >= 2 or "/" in num or "-" in num:
                        # Anchor with postal code or first address token
                        postals = extract_postal_codes(raw_addr)
                        if postals:
                            q_indic_address_numeric[f"IN::{postals[0]}::{num}"].append(q_idx)
                        else:
                            addr_toks = standard_clean(raw_addr).split()
                            city_anchor = addr_toks[-1] if addr_toks else "in"
                            q_indic_address_numeric[f"IN::{city_anchor}::{num}"].append(q_idx)

            # Exp D: Domain compact signature
            domain_sig = get_compact_domain_signature(raw_name, min_length=5)
            if domain_sig:
                q_domain_compact[domain_sig].append(q_idx)

            # Exp E: Leetspeak / Accent signature
            leet_sig = get_leetspeak_signature(raw_name, min_length=4)
            if leet_sig and leet_sig != domain_sig:
                q_leetspeak_compact[leet_sig].append(q_idx)

            # Exp F: Complete alias secondary address compound keys
            # Combine postal code + numeric compound
            postals = extract_postal_codes(raw_addr)
            for num in distinct_nums:
                if len(num) >= 2:
                    if postals:
                        q_alias_address[f"{country}::{postals[0]}::{num}"].append(q_idx)
                    else:
                        addr_tokens = standard_clean(raw_addr).split()
                        if len(addr_tokens) >= 2:
                            loc = addr_tokens[-1]
                            q_alias_address[f"{country}::{loc}::{num}"].append(q_idx)

            # Text representation for TF-IDF
            query_texts.append(f"{clean_name} {standard_clean(raw_addr)}".strip())

        # Candidate collectors per query for each experiment
        max_cands = 500
        cands_baseline: List[Set[str]] = [set() for _ in range(n_queries)]
        cands_exp_b: List[Set[str]] = [set() for _ in range(n_queries)]
        cands_exp_c: List[Set[str]] = [set() for _ in range(n_queries)]
        cands_exp_d: List[Set[str]] = [set() for _ in range(n_queries)]
        cands_exp_e: List[Set[str]] = [set() for _ in range(n_queries)]
        cands_exp_f: List[Set[str]] = [set() for _ in range(n_queries)]

        # -----------------------------------------------------------------
        # Step 2: Fit TF-IDF Vocabulary (Representative 500K sample)
        # -----------------------------------------------------------------
        print("[Step 2/4] Fitting TF-IDF vocabulary on representative 500K sample...", flush=True)
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

        Q = vectorizer.transform(query_texts)

        # Min-heaps for top-100 TF-IDF scores
        max_k = 100
        tfidf_heaps: List[List[Tuple[float, str]]] = [[] for _ in range(n_queries)]

        index_build_time = time.time() - t_index_start
        print(f"  Query indexes & TF-IDF prepped in {index_build_time:.2f}s.", flush=True)

        # -----------------------------------------------------------------
        # Step 3: Stream All 10.32M Target Records
        # -----------------------------------------------------------------
        print("[Step 3/4] Streaming through full 10.32M target corpus...", flush=True)
        t_stream_start = time.time()
        total_targets = 0
        chunk_size = 500000
        chunk_texts: List[str] = []
        chunk_eids: List[str] = []

        # Caps on bucket additions to prevent single distractor explosion
        bucket_hit_counts: Dict[str, int] = defaultdict(int)
        MAX_BUCKET_HITS = 250

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
                    country = parts[3].upper() if len(parts) >= 4 else ""
                    total_targets += 1

                    clean_name = strip_legal_suffixes(raw_name)

                    # --- Baseline U6 Matches ---
                    if clean_name in q_exact:
                        for q_idx in q_exact[clean_name]:
                            if len(cands_baseline[q_idx]) < max_cands:
                                cands_baseline[q_idx].add(eid)

                    sorted_name = get_sorted_tokens_name(raw_name, strip_legal=True)
                    if sorted_name in q_sorted:
                        for q_idx in q_sorted[sorted_name]:
                            if len(cands_baseline[q_idx]) < max_cands:
                                cands_baseline[q_idx].add(eid)

                    t_base_tokens = get_name_tokens(clean_name)
                    t_base_nums = extract_numeric_tokens(raw_addr)
                    if t_base_tokens and len(t_base_tokens[0]) >= 3:
                        for num in t_base_nums:
                            k = f"{t_base_tokens[0]}::{num}"
                            if k in q_numeric_base:
                                for q_idx in q_numeric_base[k]:
                                    if len(cands_baseline[q_idx]) < max_cands:
                                        cands_baseline[q_idx].add(eid)

                    if clean_name:
                        compact = clean_name.replace(" ", "")
                        sig = compact if len(compact) < 4 else f"{compact[:4]}::{compact[-4:]}"
                        if sig in q_char:
                            for q_idx in q_char[sig]:
                                if len(cands_baseline[q_idx]) < max_cands:
                                    cands_baseline[q_idx].add(eid)

                    # --- Experiment B: Frequency-aware compound keys ---
                    t_inf_tokens = get_informative_name_tokens(raw_name)
                    t_distinct_nums = extract_address_numeric_compounds(raw_addr)
                    if t_inf_tokens:
                        for tok in t_inf_tokens[:2]:
                            for num in t_distinct_nums:
                                k = f"{tok}::{num}"
                                if k in q_numeric_freq_aware:
                                    if bucket_hit_counts[k] < MAX_BUCKET_HITS:
                                        bucket_hit_counts[k] += 1
                                        for q_idx in q_numeric_freq_aware[k]:
                                            if len(cands_exp_b[q_idx]) < max_cands:
                                                cands_exp_b[q_idx].add(eid)

                    # --- Experiment C: Cross-script Indic Transliteration & Address ---
                    is_indic = has_indic_script(raw_name)
                    if is_indic:
                        # C1: Address numeric fallback for Indic targets
                        for num in t_distinct_nums:
                            if len(num) >= 2 or "/" in num or "-" in num:
                                t_postals = extract_postal_codes(raw_addr)
                                if t_postals:
                                    k = f"IN::{t_postals[0]}::{num}"
                                    if k in q_indic_address_numeric:
                                        for q_idx in q_indic_address_numeric[k]:
                                            cands_exp_c[q_idx].add(eid)
                                else:
                                    t_addr_toks = standard_clean(raw_addr).split()
                                    if t_addr_toks:
                                        city_anchor = t_addr_toks[-1]
                                        k = f"IN::{city_anchor}::{num}"
                                        if k in q_indic_address_numeric:
                                            for q_idx in q_indic_address_numeric[k]:
                                                cands_exp_c[q_idx].add(eid)

                        # C2: Transliteration
                        t_translit_sig = get_indic_transliterated_signature(raw_name, min_length=4)
                        if t_translit_sig:
                            k_prefix = t_translit_sig[:5]
                            if k_prefix in q_indic_translit:
                                for q_idx in q_indic_translit[k_prefix]:
                                    cands_exp_c[q_idx].add(eid)

                    # --- Experiment D: Domain / URL normalization ---
                    t_domain_sig = get_compact_domain_signature(raw_name, min_length=5)
                    if t_domain_sig and t_domain_sig in q_domain_compact:
                        for q_idx in q_domain_compact[t_domain_sig]:
                            cands_exp_d[q_idx].add(eid)

                    # --- Experiment E: Leetspeak / Accent normalization ---
                    t_leet_sig = get_leetspeak_signature(raw_name, min_length=4)
                    if t_leet_sig and t_leet_sig in q_leetspeak_compact:
                        for q_idx in q_leetspeak_compact[t_leet_sig]:
                            cands_exp_e[q_idx].add(eid)

                    # --- Experiment F: Complete alias secondary address blocking ---
                    t_postals = extract_postal_codes(raw_addr)
                    for num in t_distinct_nums:
                        if len(num) >= 2:
                            if t_postals:
                                k = f"{country}::{t_postals[0]}::{num}"
                                if k in q_alias_address:
                                    for q_idx in q_alias_address[k]:
                                        cands_exp_f[q_idx].add(eid)
                            else:
                                t_addr_toks = standard_clean(raw_addr).split()
                                if len(t_addr_toks) >= 2:
                                    loc = t_addr_toks[-1]
                                    k = f"{country}::{loc}::{num}"
                                    if k in q_alias_address:
                                        for q_idx in q_alias_address[k]:
                                            cands_exp_f[q_idx].add(eid)

                    # Accumulate for chunked TF-IDF
                    chunk_texts.append(f"{clean_name} {standard_clean(raw_addr)}".strip())
                    chunk_eids.append(eid)

                    if len(chunk_texts) >= chunk_size:
                        self._process_tfidf_chunk(Q, vectorizer, chunk_texts, chunk_eids, tfidf_heaps, max_k)
                        chunk_texts = []
                        chunk_eids = []
                        print(f"    Streamed {total_targets:,} / 10,320,219 records...", flush=True)

        # Process final chunk
        if chunk_texts:
            self._process_tfidf_chunk(Q, vectorizer, chunk_texts, chunk_eids, tfidf_heaps, max_k)
            chunk_texts = []
            chunk_eids = []

        stream_time = time.time() - t_stream_start
        print(f"  Streaming complete: {total_targets:,} records processed in {stream_time:.2f}s.", flush=True)

        # -----------------------------------------------------------------
        # Step 4: Evaluate All Experiments and Configurations
        # -----------------------------------------------------------------
        print("[Step 4/4] Computing empirical metrics across Experiments A-F...", flush=True)

        # Helper to compute recall & candidate metrics for a given candidate map
        def compute_metrics(
            c_list: List[Set[str]],
            name: str,
            runtime_sec: float,
        ) -> Dict[str, Any]:
            total_true = 0
            captured_true = 0
            s2_true = 0
            s2_captured = 0
            s3_true = 0
            s3_captured = 0
            sizes = []

            for q_idx, q in enumerate(self.query_records):
                s1_id = q["entity_id"]
                c_set = c_list[q_idx]
                sizes.append(len(c_set))

                for mid in self.ground_truth.get(s1_id, []):
                    total_true += 1
                    is_s2 = mid.startswith("S2-")
                    is_s3 = mid.startswith("S3-")
                    if is_s2:
                        s2_true += 1
                    elif is_s3:
                        s3_true += 1

                    if mid in c_set:
                        captured_true += 1
                        if is_s2:
                            s2_captured += 1
                        elif is_s3:
                            s3_captured += 1

            s_arr = np.array(sizes, dtype=np.int32)
            cartesian = n_queries * total_targets
            total_pairs = int(s_arr.sum())
            reduction = 1.0 - (total_pairs / cartesian) if cartesian > 0 else 1.0

            return {
                "experiment": name,
                "blocking_recall": captured_true / total_true if total_true > 0 else 0.0,
                "s2_recall": s2_captured / s2_true if s2_true > 0 else 0.0,
                "s3_recall": s3_captured / s3_true if s3_true > 0 else 0.0,
                "captured_true_links": captured_true,
                "total_true_links": total_true,
                "mean_candidates": float(np.mean(s_arr)),
                "median_candidates": float(np.median(s_arr)),
                "p90_candidates": float(np.percentile(s_arr, 90)),
                "p95_candidates": float(np.percentile(s_arr, 95)),
                "p99_candidates": float(np.percentile(s_arr, 99)),
                "max_candidates": int(np.max(s_arr)),
                "total_candidates": total_pairs,
                "reduction_ratio": reduction,
                "runtime_seconds": runtime_sec,
            }

        # Helper to slice top-k TF-IDF candidates
        def get_tfidf_candidates(k: int) -> List[Set[str]]:
            out = [set() for _ in range(n_queries)]
            for q_idx in range(n_queries):
                heap = tfidf_heaps[q_idx]
                # Heap contains (score, eid). Largest scores are at the end when sorted
                sorted_items = sorted(heap, key=lambda x: x[0], reverse=True)[:k]
                for s, eid in sorted_items:
                    out[q_idx].add(eid)
            return out

        # Helper to union multiple candidate lists
        def union_candidates(*cand_lists: List[Set[str]]) -> List[Set[str]]:
            merged = [set() for _ in range(n_queries)]
            for c_list in cand_lists:
                for q_idx in range(n_queries):
                    merged[q_idx].update(c_list[q_idx])
            return merged

        # 1. Experiment A: TF-IDF top-k sweep (k = 25, 40, 50, 75, 100)
        # Combined with baseline deterministic
        topk_results = []
        cands_tfidf = {}
        for k in [25, 40, 50, 75, 100]:
            cands_tfidf[k] = get_tfidf_candidates(k)
            u_base_k = union_candidates(cands_baseline, cands_tfidf[k])
            res_k = compute_metrics(u_base_k, f"TF-IDF k={k} (Baseline Union)", stream_time)
            res_k["k"] = k
            topk_results.append(res_k)
            print(
                f"  Exp A (k={k:3d}): Recall={res_k['blocking_recall']:.4f} (S2: {res_k['s2_recall']:.4f}, S3: {res_k['s3_recall']:.4f}) | "
                f"Mean={res_k['mean_candidates']:.2f} | P95={res_k['p95_candidates']:.0f} | Max={res_k['max_candidates']}",
                flush=True
            )

        # 2. Experiment B: Frequency-aware compound keys (added to baseline U6)
        u_b = union_candidates(cands_baseline, cands_exp_b, cands_tfidf[25])
        res_b = compute_metrics(u_b, "Exp B: Frequency-Aware Retrieval", stream_time)
        print(f"  Exp B: Recall={res_b['blocking_recall']:.4f} | Mean={res_b['mean_candidates']:.2f} | Max={res_b['max_candidates']}", flush=True)

        # 3. Experiment C: Cross-script recovery (Transliteration + Indic address)
        u_c = union_candidates(cands_baseline, cands_exp_c, cands_tfidf[25])
        res_c = compute_metrics(u_c, "Exp C: Cross-Script Recovery", stream_time)
        print(f"  Exp C: Recall={res_c['blocking_recall']:.4f} | Mean={res_c['mean_candidates']:.2f} | Max={res_c['max_candidates']}", flush=True)

        # 4. Experiment D: URL / domain normalization
        u_d = union_candidates(cands_baseline, cands_exp_d, cands_tfidf[25])
        res_d = compute_metrics(u_d, "Exp D: Domain Normalization", stream_time)
        print(f"  Exp D: Recall={res_d['blocking_recall']:.4f} | Mean={res_d['mean_candidates']:.2f} | Max={res_d['max_candidates']}", flush=True)

        # 5. Experiment E: Leetspeak / Accent normalization
        u_e = union_candidates(cands_baseline, cands_exp_e, cands_tfidf[25])
        res_e = compute_metrics(u_e, "Exp E: Leetspeak / Accent", stream_time)
        print(f"  Exp E: Recall={res_e['blocking_recall']:.4f} | Mean={res_e['mean_candidates']:.2f} | Max={res_e['max_candidates']}", flush=True)

        # 6. Experiment F: Complete alias secondary address compound keys
        u_f = union_candidates(cands_baseline, cands_exp_f, cands_tfidf[25])
        res_f = compute_metrics(u_f, "Exp F: Complete Alias Recovery", stream_time)
        print(f"  Exp F: Recall={res_f['blocking_recall']:.4f} | Mean={res_f['mean_candidates']:.2f} | Max={res_f['max_candidates']}", flush=True)

        # -----------------------------------------------------------------
        # Pareto Configurations Comparison
        # -----------------------------------------------------------------
        print("\n--- Benchmarking Engineering Configurations (Pareto Frontier) ---", flush=True)

        # Configuration A: Baseline U6 (k=25, Phase 1.6 baseline)
        cfg_a = compute_metrics(union_candidates(cands_baseline, cands_tfidf[25]), "Config A (Phase 1.6 Baseline U6, k=25)", stream_time)

        # Configuration B: Fast Recovery (Baseline + Domain + Frequency-Aware + k=40)
        cfg_b = compute_metrics(union_candidates(cands_baseline, cands_exp_b, cands_exp_d, cands_tfidf[40]), "Config B (Fast Recovery: Domain + FreqAware + k=40)", stream_time)

        # Configuration C: Balanced Full Recovery (Baseline + Domain + FreqAware + CrossScript + Leetspeak + k=50)
        cfg_c = compute_metrics(union_candidates(cands_baseline, cands_exp_b, cands_exp_c, cands_exp_d, cands_exp_e, cands_tfidf[50]), "Config C (Balanced Full Recovery: All Enhancements + k=50)", stream_time)

        # Configuration D: Deep Recovery (All Enhancements + Alias + k=75)
        cfg_d = compute_metrics(union_candidates(cands_baseline, cands_exp_b, cands_exp_c, cands_exp_d, cands_exp_e, cands_exp_f, cands_tfidf[75]), "Config D (Deep Recovery: All Enhancements + Alias + k=75)", stream_time)

        # Configuration E: Maximum Recall (All Enhancements + Alias + k=100)
        cfg_e = compute_metrics(union_candidates(cands_baseline, cands_exp_b, cands_exp_c, cands_exp_d, cands_exp_e, cands_exp_f, cands_tfidf[100]), "Config E (Maximum Recall: All Enhancements + Alias + k=100)", stream_time)

        pareto_configs = [cfg_a, cfg_b, cfg_c, cfg_d, cfg_e]
        for cfg in pareto_configs:
            print(
                f"  {cfg['experiment']}: Recall={cfg['blocking_recall']:.4f} (S2: {cfg['s2_recall']:.4f}, S3: {cfg['s3_recall']:.4f}) | "
                f"Mean={cfg['mean_candidates']:.2f} | P95={cfg['p95_candidates']:.0f} | P99={cfg['p99_candidates']:.0f} | Max={cfg['max_candidates']}",
                flush=True
            )

        peak_ram = process.memory_info().rss / (1024 * 1024)
        total_time = time.time() - t_start

        return {
            "query_cohort_size": n_queries,
            "target_corpus_size": total_targets,
            "stream_time_seconds": stream_time,
            "total_time_seconds": total_time,
            "peak_ram_mb": peak_ram,
            "topk_results": topk_results,
            "experiment_results": [
                cfg_a, res_b, res_c, res_d, res_e, res_f
            ],
            "pareto_configurations": pareto_configs,
        }

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
        scores_mat = Q.dot(X_chunk.T)

        for q_idx in range(Q.shape[0]):
            row = scores_mat.getrow(q_idx)
            if row.nnz == 0:
                continue

            heap = tfidf_heaps[q_idx]

            if row.nnz <= top_k:
                for col_idx, s in zip(row.indices, row.data):
                    target_eid = chunk_eids[col_idx]
                    if len(heap) < top_k:
                        heapq.heappush(heap, (float(s), target_eid))
                    elif s > heap[0][0]:
                        heapq.heappushpop(heap, (float(s), target_eid))
            else:
                part = np.argpartition(row.data, -top_k)[-top_k:]
                for p_idx in part:
                    s = float(row.data[p_idx])
                    target_eid = chunk_eids[row.indices[p_idx]]
                    if len(heap) < top_k:
                        heapq.heappush(heap, (s, target_eid))
                    elif s > heap[0][0]:
                        heapq.heappushpop(heap, (s, target_eid))
