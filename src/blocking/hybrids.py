"""Block Family E: Name + Address Hybrid Blockers.

Combines name evidence with structural address evidence (postal codes, numeric tokens, country)
to generate high-precision, explosion-resistant candidate buckets.
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Set
import time

from src.blocking.base import BaseBlocker
from src.representations.address import extract_numeric_tokens, extract_postal_codes
from src.representations.name import get_name_tokens, strip_legal_suffixes


class NamePostalHybridBlocker(BaseBlocker):
    """Compound key of first informative name token + postal / PIN code."""

    def __init__(
        self,
        name: str = "BLOCK-HYBRID-NAME-POSTAL",
        min_token_len: int = 3,
        max_bucket_size: int = 500,
    ):
        super().__init__(name=name, config={
            "min_token_len": min_token_len,
            "max_bucket_size": max_bucket_size,
        })
        self.min_token_len = min_token_len
        self.max_bucket_size = max_bucket_size
        self.index: Dict[str, List[str]] = defaultdict(list)

    def _get_keys(self, record: Dict[str, str]) -> List[str]:
        raw_name = record.get("business_name", "")
        clean_name = strip_legal_suffixes(raw_name)
        name_toks = [t for t in get_name_tokens(clean_name) if len(t) >= self.min_token_len]
        if not name_toks:
            return []

        first_name_tok = name_toks[0]
        postal_codes = extract_postal_codes(record.get("business_address", ""))
        if not postal_codes:
            return []

        country = record.get("country", "").strip().upper()
        keys = []
        for pc in postal_codes:
            keys.append(f"{country}::{first_name_tok}::{pc}")
        return keys

    def fit(self, targets: Sequence[Dict[str, str]]) -> "NamePostalHybridBlocker":
        t0 = time.time()
        self.index.clear()
        for rec in targets:
            for key in self._get_keys(rec):
                self.index[key].append(rec["entity_id"])
        self.indexing_time_sec = round(time.time() - t0, 3)
        return self

    def retrieve(self, queries: Sequence[Dict[str, str]]) -> Dict[str, Set[str]]:
        candidates: Dict[str, Set[str]] = {}
        for q in queries:
            s1_id = q["entity_id"]
            keys = self._get_keys(q)
            matched_set: Set[str] = set()
            for key in keys:
                matches = self.index.get(key, [])
                if self.max_bucket_size and len(matches) > self.max_bucket_size:
                    matched_set.update(matches[:self.max_bucket_size])
                else:
                    matched_set.update(matches)
            candidates[s1_id] = matched_set
        return candidates


class NameNumericHybridBlocker(BaseBlocker):
    """Compound key of first informative name token + first address numeric token."""

    def __init__(
        self,
        name: str = "BLOCK-HYBRID-NAME-NUMERIC",
        min_token_len: int = 3,
        max_bucket_size: int = 500,
    ):
        super().__init__(name=name, config={
            "min_token_len": min_token_len,
            "max_bucket_size": max_bucket_size,
        })
        self.min_token_len = min_token_len
        self.max_bucket_size = max_bucket_size
        self.index: Dict[str, List[str]] = defaultdict(list)

    def _get_keys(self, record: Dict[str, str]) -> List[str]:
        raw_name = record.get("business_name", "")
        clean_name = strip_legal_suffixes(raw_name)
        name_toks = [t for t in get_name_tokens(clean_name) if len(t) >= self.min_token_len]
        if not name_toks:
            return []

        first_name_tok = name_toks[0]
        numeric_toks = extract_numeric_tokens(record.get("business_address", ""))
        if not numeric_toks:
            return []

        first_num = numeric_toks[0]
        country = record.get("country", "").strip().upper()
        return [f"{country}::{first_name_tok}::{first_num}"]

    def fit(self, targets: Sequence[Dict[str, str]]) -> "NameNumericHybridBlocker":
        t0 = time.time()
        self.index.clear()
        for rec in targets:
            for key in self._get_keys(rec):
                self.index[key].append(rec["entity_id"])
        self.indexing_time_sec = round(time.time() - t0, 3)
        return self

    def retrieve(self, queries: Sequence[Dict[str, str]]) -> Dict[str, Set[str]]:
        candidates: Dict[str, Set[str]] = {}
        for q in queries:
            s1_id = q["entity_id"]
            keys = self._get_keys(q)
            matched_set: Set[str] = set()
            for key in keys:
                matches = self.index.get(key, [])
                if self.max_bucket_size and len(matches) > self.max_bucket_size:
                    matched_set.update(matches[:self.max_bucket_size])
                else:
                    matched_set.update(matches)
            candidates[s1_id] = matched_set
        return candidates
