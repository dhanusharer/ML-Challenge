"""Candidate generator and pair loader for Phase 2 candidate matching.

Extracts candidates under ARM B (Config B) and ARM C (Config C), records exact
retrieval provenance, computes true positive labels, and tracks BLOCKING_MISS records.
"""

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import heapq
import numpy as np
import time
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from src.representations.normalizer import standard_clean
from src.representations.name import (
    strip_legal_suffixes,
    get_sorted_tokens_name,
    get_name_tokens,
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
from src.blocking.recovery_lab import (
    get_informative_name_tokens,
    extract_address_numeric_compounds,
)


class CandidateArmLoader:
    """Generates candidate sets under ARM B (Config B) or ARM C (Config C)."""

    def __init__(
        self,
        target_records: List[Dict[str, Any]],
        arm: str = "ARM_B",
    ):
        self.target_records = target_records
        self.arm = arm.upper()
        self.top_k = 40 if self.arm == "ARM_B" else 50
        self.include_cross_script = (self.arm == "ARM_C")
        self.include_leetspeak = (self.arm == "ARM_C")

        # Fit TF-IDF on targets
        self._fit_vectorizer()

    def _fit_vectorizer(self) -> None:
        """Fit vocabulary on target corpus texts."""
        texts = []
        for r in self.target_records[:50000]:
            name = strip_legal_suffixes(r.get("business_name", ""))
            addr = standard_clean(r.get("business_address", ""))
            texts.append(f"{name} {addr}".strip())

        self.vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.25,
            max_features=10000,
            sublinear_tf=True,
            dtype=np.float32,
        )
        self.vectorizer.fit(texts)

        # Precompute target matrix
        target_texts = [
            f"{strip_legal_suffixes(r.get('business_name', ''))} {standard_clean(r.get('business_address', ''))}".strip()
            for r in self.target_records
        ]
        self.X_targets = self.vectorizer.transform(target_texts)

    def generate_candidates_for_queries(
        self,
        query_records: List[Dict[str, Any]],
        ground_truth: Dict[str, List[str]],
    ) -> Dict[str, Any]:
        """Generate candidates for query records with provenance and true labels."""
        t0 = time.time()
        n_queries = len(query_records)
        target_id_to_record = {r["entity_id"]: r for r in self.target_records}

        # 1. Build Query Inverted Indexes
        q_exact: Dict[str, List[int]] = defaultdict(list)
        q_sorted: Dict[str, List[int]] = defaultdict(list)
        q_numeric: Dict[str, List[int]] = defaultdict(list)
        q_char: Dict[str, List[int]] = defaultdict(list)
        q_domain: Dict[str, List[int]] = defaultdict(list)
        q_translit: Dict[str, List[int]] = defaultdict(list)
        q_indic_addr: Dict[str, List[int]] = defaultdict(list)
        q_leet: Dict[str, List[int]] = defaultdict(list)

        query_texts: List[str] = []

        for q_idx, q in enumerate(query_records):
            raw_name = q.get("business_name", "")
            raw_addr = q.get("business_address", "")
            country = q.get("country", "").upper()
            clean_name = strip_legal_suffixes(raw_name)

            if clean_name:
                q_exact[clean_name].append(q_idx)

            sorted_name = get_sorted_tokens_name(raw_name, strip_legal=True)
            if sorted_name:
                q_sorted[sorted_name].append(q_idx)

            # Frequency-aware compound keys
            inf_tokens = get_informative_name_tokens(raw_name)
            nums = extract_address_numeric_compounds(raw_addr)
            if inf_tokens:
                for num in nums:
                    q_numeric[f"{inf_tokens[0]}::{num}"].append(q_idx)

            if clean_name:
                compact = clean_name.replace(" ", "")
                sig = compact if len(compact) < 4 else f"{compact[:4]}::{compact[-4:]}"
                if sig:
                    q_char[sig].append(q_idx)

            # Domain normalization
            dom_sig = get_compact_domain_signature(raw_name, min_length=4)
            if dom_sig:
                q_domain[dom_sig].append(q_idx)

            # Cross-Script (ARM C)
            if self.include_cross_script and clean_name:
                latin_root = get_compact_domain_signature(clean_name, min_length=4)
                if latin_root:
                    q_translit[latin_root[:5]].append(q_idx)
                for num in nums:
                    if len(num) >= 2:
                        postals = extract_postal_codes(raw_addr)
                        p_anchor = postals[0] if postals else "addr"
                        q_indic_addr[f"{p_anchor}::{num}"].append(q_idx)

            # Leetspeak (ARM C)
            if self.include_leetspeak:
                leet_sig = get_leetspeak_signature(raw_name, min_length=4)
                if leet_sig:
                    q_leet[leet_sig].append(q_idx)

            query_texts.append(f"{clean_name} {standard_clean(raw_addr)}".strip())

        # 2. Match Target Records against Query Inverted Indexes
        # Candidate map per query: target_eid -> provenance dict
        query_candidates: List[Dict[str, Dict[str, Any]]] = [{} for _ in range(n_queries)]

        for r in self.target_records:
            eid = r["entity_id"]
            raw_name = r.get("business_name", "")
            raw_addr = r.get("business_address", "")
            clean_name = strip_legal_suffixes(raw_name)

            def _record_hit(q_idx: int, block_name: str):
                c_map = query_candidates[q_idx]
                if eid not in c_map:
                    c_map[eid] = {
                        "exact_name": False,
                        "sorted_name": False,
                        "name_numeric": False,
                        "char_ngram": False,
                        "domain": False,
                        "transliteration": False,
                        "leetspeak": False,
                        "tfidf": False,
                        "tfidf_score": 0.0,
                        "tfidf_rank": 1000,
                        "support_count": 0,
                    }
                if not c_map[eid].get(block_name, False):
                    c_map[eid][block_name] = True
                    c_map[eid]["support_count"] += 1

            if clean_name in q_exact:
                for q_idx in q_exact[clean_name]:
                    _record_hit(q_idx, "exact_name")

            sorted_name = get_sorted_tokens_name(raw_name, strip_legal=True)
            if sorted_name in q_sorted:
                for q_idx in q_sorted[sorted_name]:
                    _record_hit(q_idx, "sorted_name")

            inf_tokens = get_informative_name_tokens(raw_name)
            nums = extract_address_numeric_compounds(raw_addr)
            if inf_tokens:
                for num in nums:
                    k = f"{inf_tokens[0]}::{num}"
                    if k in q_numeric:
                        for q_idx in q_numeric[k]:
                            _record_hit(q_idx, "name_numeric")

            if clean_name:
                compact = clean_name.replace(" ", "")
                sig = compact if len(compact) < 4 else f"{compact[:4]}::{compact[-4:]}"
                if sig in q_char:
                    for q_idx in q_char[sig]:
                        _record_hit(q_idx, "char_ngram")

            dom_sig = get_compact_domain_signature(raw_name, min_length=4)
            if dom_sig and dom_sig in q_domain:
                for q_idx in q_domain[dom_sig]:
                    _record_hit(q_idx, "domain")

            if self.include_cross_script and has_indic_script(raw_name):
                t_translit_sig = get_indic_transliterated_signature(raw_name, min_length=4)
                if t_translit_sig and t_translit_sig[:5] in q_translit:
                    for q_idx in q_translit[t_translit_sig[:5]]:
                        _record_hit(q_idx, "transliteration")
                for num in nums:
                    if len(num) >= 2:
                        postals = extract_postal_codes(raw_addr)
                        p_anchor = postals[0] if postals else "addr"
                        k = f"{p_anchor}::{num}"
                        if k in q_indic_addr:
                            for q_idx in q_indic_addr[k]:
                                _record_hit(q_idx, "transliteration")

            if self.include_leetspeak:
                leet_sig = get_leetspeak_signature(raw_name, min_length=4)
                if leet_sig and leet_sig in q_leet:
                    for q_idx in q_leet[leet_sig]:
                        _record_hit(q_idx, "leetspeak")

        # 3. TF-IDF Top-k Retrieval
        Q = self.vectorizer.transform(query_texts)
        scores_mat = Q.dot(self.X_targets.T)  # (n_queries, n_targets)

        for q_idx in range(n_queries):
            row = scores_mat.getrow(q_idx)
            if row.nnz > 0:
                k = min(self.top_k, row.nnz)
                top_part = np.argpartition(row.data, -k)[-k:]
                sorted_idx = top_part[np.argsort(-row.data[top_part])]

                for rank_pos, p_idx in enumerate(sorted_idx, start=1):
                    col_idx = row.indices[p_idx]
                    s = float(row.data[p_idx])
                    target_eid = self.target_records[col_idx]["entity_id"]

                    c_map = query_candidates[q_idx]
                    if target_eid not in c_map:
                        c_map[target_eid] = {
                            "exact_name": False,
                            "sorted_name": False,
                            "name_numeric": False,
                            "char_ngram": False,
                            "domain": False,
                            "transliteration": False,
                            "leetspeak": False,
                            "tfidf": True,
                            "tfidf_score": s,
                            "tfidf_rank": rank_pos,
                            "support_count": 1,
                        }
                    else:
                        if not c_map[target_eid]["tfidf"]:
                            c_map[target_eid]["tfidf"] = True
                            c_map[target_eid]["support_count"] += 1
                        c_map[target_eid]["tfidf_score"] = s
                        c_map[target_eid]["tfidf_rank"] = rank_pos

        # 4. Formulate query candidate structures and compute coverage metrics
        total_true_links = 0
        captured_true_links = 0
        blocking_misses: List[Dict[str, str]] = []
        structured_cohort: List[Dict[str, Any]] = []

        for q_idx, q in enumerate(query_records):
            s1_id = q["entity_id"]
            gt_links = ground_truth.get(s1_id, [])
            total_true_links += len(gt_links)

            cand_items = []
            for tid, prov in query_candidates[q_idx].items():
                if tid in target_id_to_record:
                    cand_items.append({
                        "target_record": target_id_to_record[tid],
                        "provenance": prov,
                        "label": 1 if tid in gt_links else 0,
                    })

            cand_tids = {item["target_record"]["entity_id"] for item in cand_items}
            for mid in gt_links:
                if mid in cand_tids:
                    captured_true_links += 1
                else:
                    blocking_misses.append({
                        "source1_entity_id": s1_id,
                        "missing_target_id": mid,
                        "source1_name": q.get("business_name", ""),
                    })

            structured_cohort.append({
                "query_record": q,
                "ground_truth_targets": set(gt_links),
                "candidates": cand_items,
            })

        gen_time = time.time() - t0
        blocking_recall = captured_true_links / total_true_links if total_true_links > 0 else 0.0

        return {
            "arm": self.arm,
            "top_k": self.top_k,
            "query_count": n_queries,
            "target_count": len(self.target_records),
            "generation_time_seconds": gen_time,
            "total_true_links": total_true_links,
            "captured_true_links": captured_true_links,
            "blocking_recall": blocking_recall,
            "blocking_miss_count": len(blocking_misses),
            "blocking_misses": blocking_misses,
            "structured_cohort": structured_cohort,
        }
