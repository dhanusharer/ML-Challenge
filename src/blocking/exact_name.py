"""Block Family A: Exact Normalized Name Blocker (BLOCK-NAME-EXACT).

Builds an inverted index mapping exact normalized business names to target records.
Supports:
- Baseline cleaned name (Unicode NFKC, lowercase, punctuation removed)
- Legal-suffix-stripped exact name
- Optional country-partitioned exact name
- Frequency cap to prevent pathological candidate explosion
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Set
import time

from src.blocking.base import BaseBlocker
from src.representations.name import standard_clean, strip_legal_suffixes


class ExactNameBlocker(BaseBlocker):
    """Inverted index blocker on exact normalized business names."""

    def __init__(
        self,
        name: str = "BLOCK-NAME-EXACT",
        strip_legal: bool = False,
        country_aware: bool = False,
        max_bucket_size: Optional[int] = 1000,
    ):
        super().__init__(name=name, config={
            "strip_legal": strip_legal,
            "country_aware": country_aware,
            "max_bucket_size": max_bucket_size,
        })
        self.strip_legal = strip_legal
        self.country_aware = country_aware
        self.max_bucket_size = max_bucket_size
        self.index: Dict[str, List[str]] = defaultdict(list)

    def _get_key(self, record: Dict[str, str]) -> str:
        raw_name = record.get("business_name", "")
        clean_name = strip_legal_suffixes(raw_name) if self.strip_legal else standard_clean(raw_name)
        if not clean_name:
            return ""
        if self.country_aware:
            country = record.get("country", "").strip().upper()
            return f"{country}::{clean_name}"
        return clean_name

    def fit(self, targets: Sequence[Dict[str, str]]) -> "ExactNameBlocker":
        t0 = time.time()
        self.index.clear()
        for rec in targets:
            key = self._get_key(rec)
            if key:
                self.index[key].append(rec["entity_id"])
        self.indexing_time_sec = round(time.time() - t0, 3)
        return self

    def retrieve(self, queries: Sequence[Dict[str, str]]) -> Dict[str, Set[str]]:
        candidates: Dict[str, Set[str]] = {}
        for q in queries:
            s1_id = q["entity_id"]
            key = self._get_key(q)
            if not key or key not in self.index:
                candidates[s1_id] = set()
                continue

            matches = self.index[key]
            if self.max_bucket_size and len(matches) > self.max_bucket_size:
                # Truncate oversized buckets to prevent explosion
                candidates[s1_id] = set(matches[:self.max_bucket_size])
            else:
                candidates[s1_id] = set(matches)

        return candidates
