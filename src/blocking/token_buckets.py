"""Block Family B: Name Token Buckets and Signatures.

Implements token-based blocking strategies:
1. Sorted-Token Signature (word-order transposition invariant)
2. Most Informative / Rare Token Blocking (frequency-aware)
3. Token-Pair Blocking
4. Frequency capping to prevent candidate explosion on common business words
"""

from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import time

from src.blocking.base import BaseBlocker
from src.representations.name import get_name_tokens, get_sorted_tokens_name, strip_legal_suffixes


class SortedTokenNameBlocker(BaseBlocker):
    """Blocks on sorted tokens of the business name (order invariant)."""

    def __init__(
        self,
        name: str = "BLOCK-NAME-SORTED-TOKENS",
        strip_legal: bool = True,
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
        sig = get_sorted_tokens_name(record.get("business_name", ""), strip_legal=self.strip_legal)
        if not sig:
            return ""
        if self.country_aware:
            country = record.get("country", "").strip().upper()
            return f"{country}::{sig}"
        return sig

    def fit(self, targets: Sequence[Dict[str, str]]) -> "SortedTokenNameBlocker":
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


class RareTokenNameBlocker(BaseBlocker):
    """Blocks on the rarest / most informative token in the business name."""

    def __init__(
        self,
        name: str = "BLOCK-NAME-RARE-TOKEN",
        min_token_len: int = 3,
        max_token_freq: int = 1000,
        max_bucket_size: int = 500,
        country_aware: bool = False,
    ):
        super().__init__(name=name, config={
            "min_token_len": min_token_len,
            "max_token_freq": max_token_freq,
            "max_bucket_size": max_bucket_size,
            "country_aware": country_aware,
        })
        self.min_token_len = min_token_len
        self.max_token_freq = max_token_freq
        self.max_bucket_size = max_bucket_size
        self.country_aware = country_aware
        self.token_freq: Counter = Counter()
        self.index: Dict[str, List[str]] = defaultdict(list)

    def _extract_tokens(self, record: Dict[str, str]) -> List[str]:
        raw_name = record.get("business_name", "")
        clean_name = strip_legal_suffixes(raw_name)
        tokens = [t for t in get_name_tokens(clean_name) if len(t) >= self.min_token_len]
        return tokens

    def fit(self, targets: Sequence[Dict[str, str]]) -> "RareTokenNameBlocker":
        t0 = time.time()
        self.index.clear()
        self.token_freq.clear()

        # Step 1: Count target token document frequency
        target_tokens_map: List[Tuple[str, str, List[str]]] = []  # (eid, country, tokens)
        for rec in targets:
            eid = rec["entity_id"]
            country = rec.get("country", "").strip().upper()
            toks = self._extract_tokens(rec)
            target_tokens_map.append((eid, country, toks))
            for t in set(toks):
                self.token_freq[t] += 1

        # Step 2: Index each target by its rarest token (or tokens below max_token_freq)
        for eid, country, toks in target_tokens_map:
            if not toks:
                continue
            # Pick token with lowest corpus frequency
            rare_tok = min(toks, key=lambda t: self.token_freq[t])
            if self.token_freq[rare_tok] <= self.max_token_freq:
                key = f"{country}::{rare_tok}" if self.country_aware else rare_tok
                self.index[key].append(eid)

        self.indexing_time_sec = round(time.time() - t0, 3)
        return self

    def retrieve(self, queries: Sequence[Dict[str, str]]) -> Dict[str, Set[str]]:
        candidates: Dict[str, Set[str]] = {}
        for q in queries:
            s1_id = q["entity_id"]
            country = q.get("country", "").strip().upper()
            toks = self._extract_tokens(q)
            if not toks:
                candidates[s1_id] = set()
                continue

            # Query under the query's rarest token that exists in target vocabulary
            valid_toks = [t for t in toks if t in self.token_freq]
            if not valid_toks:
                candidates[s1_id] = set()
                continue

            rare_tok = min(valid_toks, key=lambda t: self.token_freq[t])
            key = f"{country}::{rare_tok}" if self.country_aware else rare_tok

            matches = self.index.get(key, [])
            if self.max_bucket_size and len(matches) > self.max_bucket_size:
                candidates[s1_id] = set(matches[:self.max_bucket_size])
            else:
                candidates[s1_id] = set(matches)
        return candidates
