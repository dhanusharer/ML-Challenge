"""Block Family C: Character N-Gram Blocking.

Implements character-level candidate retrieval to recover true matches with typos,
abbreviations, or transliteration differences:
1. Rarest Character N-Gram Index (frequency-filtered inverted index)
2. Character N-Gram Boundary Signature (First N-Gram + Last N-Gram compound key)
"""

from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import time

from src.blocking.base import BaseBlocker
from src.representations.name import get_char_ngrams, strip_legal_suffixes


class CharNgramBoundaryBlocker(BaseBlocker):
    """Compound boundary n-gram blocker (first n-gram + last n-gram).

    Captures spelling variations inside the business name while anchoring on boundaries.
    """

    def __init__(
        self,
        name: str = "BLOCK-CHAR-NGRAM-BOUNDARY",
        n: int = 4,
        country_aware: bool = False,
        max_bucket_size: int = 1000,
    ):
        super().__init__(name=name, config={
            "n": n,
            "country_aware": country_aware,
            "max_bucket_size": max_bucket_size,
        })
        self.n = n
        self.country_aware = country_aware
        self.max_bucket_size = max_bucket_size
        self.index: Dict[str, List[str]] = defaultdict(list)

    def _get_key(self, record: Dict[str, str]) -> str:
        raw_name = record.get("business_name", "")
        clean_name = strip_legal_suffixes(raw_name).replace(" ", "")
        if len(clean_name) < self.n:
            sig = clean_name
        else:
            first_ng = clean_name[:self.n]
            last_ng = clean_name[-self.n:]
            sig = f"{first_ng}::{last_ng}"

        if not sig:
            return ""
        if self.country_aware:
            country = record.get("country", "").strip().upper()
            return f"{country}::{sig}"
        return sig

    def fit(self, targets: Sequence[Dict[str, str]]) -> "CharNgramBoundaryBlocker":
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
                candidates[s1_id] = set(matches[:self.max_bucket_size])
            else:
                candidates[s1_id] = set(matches)
        return candidates
