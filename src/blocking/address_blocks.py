"""Block Family D: Address-Based Blocking.

Implements structural address blocking:
1. Exact Cleaned Address
2. Postal / PIN Code Inverted Index (US 5-digit, India 6-digit, France 5-digit)
3. Numeric Tokens (house / plot / street numbers)

Crucial rule: Missing addresses in S2/S3 (~3% of records) produce an empty key
and are never penalized or crashed.
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Set
import time

from src.blocking.base import BaseBlocker
from src.representations.address import (
    extract_numeric_tokens,
    extract_postal_codes,
    is_address_missing,
    standard_clean,
)


class ExactAddressBlocker(BaseBlocker):
    """Blocks on exact normalized business address."""

    def __init__(
        self,
        name: str = "BLOCK-ADDR-EXACT",
        country_aware: bool = True,
        max_bucket_size: int = 1000,
    ):
        super().__init__(name=name, config={
            "country_aware": country_aware,
            "max_bucket_size": max_bucket_size,
        })
        self.country_aware = country_aware
        self.max_bucket_size = max_bucket_size
        self.index: Dict[str, List[str]] = defaultdict(list)

    def _get_key(self, record: Dict[str, str]) -> str:
        addr = record.get("business_address", "")
        if is_address_missing(addr):
            return ""
        clean_addr = standard_clean(addr)
        if not clean_addr:
            return ""
        if self.country_aware:
            country = record.get("country", "").strip().upper()
            return f"{country}::{clean_addr}"
        return clean_addr

    def fit(self, targets: Sequence[Dict[str, str]]) -> "ExactAddressBlocker":
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


class PostalCodeBlocker(BaseBlocker):
    """Blocks on extracted 5 or 6 digit postal / PIN codes."""

    def __init__(
        self,
        name: str = "BLOCK-ADDR-POSTAL",
        country_aware: bool = True,
        max_bucket_size: int = 2000,
    ):
        super().__init__(name=name, config={
            "country_aware": country_aware,
            "max_bucket_size": max_bucket_size,
        })
        self.country_aware = country_aware
        self.max_bucket_size = max_bucket_size
        self.index: Dict[str, List[str]] = defaultdict(list)

    def _get_keys(self, record: Dict[str, str]) -> List[str]:
        addr = record.get("business_address", "")
        codes = extract_postal_codes(addr)
        if not codes:
            return []
        country = record.get("country", "").strip().upper() if self.country_aware else ""
        keys = []
        for c in codes:
            keys.append(f"{country}::{c}" if country else c)
        return keys

    def fit(self, targets: Sequence[Dict[str, str]]) -> "PostalCodeBlocker":
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
