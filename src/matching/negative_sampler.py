"""Stratified hard-negative sampling for Phase 2 candidate matching (charter §7 & §8).

Constructs representative, reproducible negative training pairs from actual candidate sets.
Preserves balanced proportions of:
1. Top-ranked TF-IDF distractors
2. Multi-block collision negatives (surfaced by multiple retrieval mechanisms)
3. Name distractors (shared name token or high character overlap)
4. Address distractors (shared house/plot numeric or postal code)
5. General candidate negatives
"""

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import numpy as np


class StratifiedNegativeSampler:
    """Selects high-signal stratified negative pairs for training."""

    def __init__(
        self,
        negatives_per_positive: int = 6,
        seed: int = 42,
    ):
        self.neg_ratio = negatives_per_positive
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def sample_query_negatives(
        self,
        s1_id: str,
        candidates: Sequence[Dict[str, Any]],
        ground_truth_targets: Set[str],
    ) -> List[Dict[str, Any]]:
        """Sample stratified negatives for a single S1 entity query.

        Candidates is a list of dicts: {'target_record': dict, 'provenance': dict}
        """
        # Separate positives and candidate negatives
        positives: List[Dict[str, Any]] = []
        negatives: List[Dict[str, Any]] = []

        for cand in candidates:
            t_rec = cand["target_record"]
            tid = t_rec.get("entity_id", "")
            if tid in ground_truth_targets:
                positives.append(cand)
            else:
                negatives.append(cand)

        n_pos = len(positives)
        if n_pos == 0:
            target_n_neg = self.neg_ratio
        else:
            target_n_neg = n_pos * self.neg_ratio

        if len(negatives) <= target_n_neg:
            return negatives

        # Partition negatives into strata
        top_tfidf: List[Dict[str, Any]] = []
        multi_block: List[Dict[str, Any]] = []
        name_distractors: List[Dict[str, Any]] = []
        addr_distractors: List[Dict[str, Any]] = []
        other_neg: List[Dict[str, Any]] = []

        for neg in negatives:
            prov = neg.get("provenance", {})
            rank = prov.get("tfidf_rank", 1000)
            score = prov.get("tfidf_score", 0.0)
            support = prov.get("support_count", 1)

            if rank <= 5 or score >= 0.40:
                top_tfidf.append(neg)
            elif support >= 2:
                multi_block.append(neg)
            elif prov.get("exact_name", False) or prov.get("sorted_name", False):
                name_distractors.append(neg)
            elif prov.get("name_numeric", False):
                addr_distractors.append(neg)
            else:
                other_neg.append(neg)

        # Allocate negative quotas across strata
        # 30% top TF-IDF, 30% multi-block, 20% name, 10% addr, 10% other
        q_tfidf = max(1, int(target_n_neg * 0.30))
        q_multi = max(1, int(target_n_neg * 0.30))
        q_name = max(1, int(target_n_neg * 0.20))
        q_addr = max(1, int(target_n_neg * 0.10))
        q_other = max(1, target_n_neg - (q_tfidf + q_multi + q_name + q_addr))

        sampled: List[Dict[str, Any]] = []

        def _sample_from_pool(pool: List[Dict[str, Any]], count: int) -> None:
            if not pool:
                return
            if len(pool) <= count:
                sampled.extend(pool)
            else:
                chosen_idx = self.rng.choice(len(pool), size=count, replace=False)
                for idx in chosen_idx:
                    sampled.append(pool[idx])

        _sample_from_pool(top_tfidf, q_tfidf)
        _sample_from_pool(multi_block, q_multi)
        _sample_from_pool(name_distractors, q_name)
        _sample_from_pool(addr_distractors, q_addr)
        _sample_from_pool(other_neg, q_other)

        # If quotas underfilled, fill remaining from leftover pool
        if len(sampled) < target_n_neg:
            sampled_ids = {c["target_record"]["entity_id"] for c in sampled}
            remaining = [c for c in negatives if c["target_record"]["entity_id"] not in sampled_ids]
            needed = target_n_neg - len(sampled)
            _sample_from_pool(remaining, needed)

        return sampled[:target_n_neg]
