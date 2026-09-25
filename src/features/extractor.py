"""High-performance pairwise feature extractor for Phase 2 candidate matching.

Extracts all registered feature groups (Name, Address, Country, Cross-Script,
Domain, Leetspeak, Rarity, Provenance, and Interactions) in batched chunks.
Produces clean, non-leaking, bounded float32 feature matrices.
"""

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import math
import re
import unicodedata
import numpy as np

from src.features.schema import get_feature_names
from src.representations.normalizer import standard_clean
from src.representations.name import (
    strip_legal_suffixes,
    get_sorted_tokens_name,
    get_name_tokens,
    get_char_ngrams,
)
from src.representations.address import extract_postal_codes, extract_numeric_tokens
from src.representations.domain import clean_domain_name, get_compact_domain_signature
from src.representations.transliteration import (
    has_indic_script,
    transliterate_indic_to_latin,
)
from src.representations.leetspeak import (
    strip_accents,
    normalize_leetspeak_text,
    get_leetspeak_signature,
)


def compute_levenshtein_ratio(s1: str, s2: str, max_len: int = 50) -> float:
    """Compute normalized Levenshtein ratio in [0, 1].

    Bounded by max_len for high-throughput candidate scoring.
    """
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0

    s1 = s1[:max_len]
    s2 = s2[:max_len]
    len1, len2 = len(s1), len(s2)

    # Use single row DP
    dp = list(range(len2 + 1))
    for i in range(1, len1 + 1):
        prev = dp[0]
        dp[0] = i
        c1 = s1[i - 1]
        for j in range(1, len2 + 1):
            temp = dp[j]
            cost = 0 if c1 == s2[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = temp

    dist = dp[len2]
    max_dist = max(len1, len2)
    return max(0.0, 1.0 - (dist / max_dist))


def compute_ngram_cosine(text1: str, text2: str, n: int = 3) -> float:
    """Compute character n-gram cosine similarity."""
    if not text1 or not text2:
        return 0.0
    if text1 == text2:
        return 1.0

    compact1 = text1.replace(" ", "")
    compact2 = text2.replace(" ", "")
    if len(compact1) < n or len(compact2) < n:
        return 1.0 if compact1 == compact2 else 0.0

    # Build multiset frequencies
    counts1: Dict[str, int] = {}
    for i in range(len(compact1) - n + 1):
        ng = compact1[i:i + n]
        counts1[ng] = counts1.get(ng, 0) + 1

    counts2: Dict[str, int] = {}
    for i in range(len(compact2) - n + 1):
        ng = compact2[i:i + n]
        counts2[ng] = counts2.get(ng, 0) + 1

    # Dot product and magnitudes
    dot = 0.0
    for ng, c1 in counts1.items():
        if ng in counts2:
            dot += c1 * counts2[ng]

    mag1 = math.sqrt(sum(c * c for c in counts1.values()))
    mag2 = math.sqrt(sum(c * c for c in counts2.values()))
    if mag1 == 0.0 or mag2 == 0.0:
        return 0.0
    return min(1.0, dot / (mag1 * mag2))


class PairwiseFeatureExtractor:
    """Extracts the complete 38-feature matrix for candidate pairs."""

    def __init__(self, corpus_idf_dict: Optional[Dict[str, float]] = None):
        self.feature_names = get_feature_names(enabled_only=True)
        self.corpus_idf = corpus_idf_dict or {}

    def extract_pair(
        self,
        q_record: Dict[str, Any],
        t_record: Dict[str, Any],
        provenance: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """Extract all pairwise features for a single (query, target) pair."""
        prov = provenance or {}

        # Raw string extraction
        raw_name_q = q_record.get("business_name", "") or ""
        raw_name_t = t_record.get("business_name", "") or ""
        raw_addr_q = q_record.get("business_address", "") or ""
        raw_addr_t = t_record.get("business_address", "") or ""
        country_q = (q_record.get("country", "") or "").upper()
        country_t = (t_record.get("country", "") or "").upper()
        target_eid = t_record.get("entity_id", "")

        # Cleaned strings
        clean_name_q = strip_legal_suffixes(raw_name_q)
        clean_name_t = strip_legal_suffixes(raw_name_t)
        clean_addr_q = standard_clean(raw_addr_q)
        clean_addr_t = standard_clean(raw_addr_t)

        # -------------------------------------------------------------
        # Group A: Business Name (G1)
        # -------------------------------------------------------------
        name_exact_clean = 1.0 if clean_name_q and clean_name_q == clean_name_t else 0.0

        compact_q = clean_name_q.replace(" ", "")
        compact_t = clean_name_t.replace(" ", "")
        name_compact_exact = 1.0 if compact_q and compact_q == compact_t else 0.0

        tokens_q = set(get_name_tokens(clean_name_q))
        tokens_t = set(get_name_tokens(clean_name_t))
        union_tokens = tokens_q.union(tokens_t)
        inter_tokens = tokens_q.intersection(tokens_t)
        name_token_jaccard = len(inter_tokens) / len(union_tokens) if union_tokens else 0.0

        min_tok_len = min(len(tokens_q), len(tokens_t))
        name_token_overlap_coeff = len(inter_tokens) / min_tok_len if min_tok_len > 0 else 0.0

        name_char_ngram_cosine = compute_ngram_cosine(clean_name_q, clean_name_t, n=3)
        name_edit_similarity = compute_levenshtein_ratio(clean_name_q, clean_name_t, max_len=50)

        # Longest common prefix
        min_len = min(len(clean_name_q), len(clean_name_t))
        max_len = max(len(clean_name_q), len(clean_name_t))
        lcp = 0
        while lcp < min_len and clean_name_q[lcp] == clean_name_t[lcp]:
            lcp += 1
        name_prefix_match_len = lcp / min_len if min_len > 0 else 0.0
        name_length_ratio = min_len / max_len if max_len > 0 else 0.0

        first_q = clean_name_q.split()[0] if clean_name_q else ""
        first_t = clean_name_t.split()[0] if clean_name_t else ""
        name_first_token_match = 1.0 if first_q and first_q == first_t else 0.0

        sorted_q = set(get_sorted_tokens_name(raw_name_q, strip_legal=True).split())
        sorted_t = set(get_sorted_tokens_name(raw_name_t, strip_legal=True).split())
        sorted_union = sorted_q.union(sorted_t)
        name_sorted_token_jaccard = len(sorted_q.intersection(sorted_t)) / len(sorted_union) if sorted_union else 0.0

        # -------------------------------------------------------------
        # Group B: Address (G2)
        # -------------------------------------------------------------
        addr_missing_either = 1.0 if not clean_addr_q or not clean_addr_t else 0.0
        addr_exact_clean = 1.0 if clean_addr_q and clean_addr_q == clean_addr_t else 0.0

        addr_toks_q = set(clean_addr_q.split())
        addr_toks_t = set(clean_addr_t.split())
        addr_union = addr_toks_q.union(addr_toks_t)
        addr_inter = addr_toks_q.intersection(addr_toks_t)
        addr_token_jaccard = len(addr_inter) / len(addr_union) if addr_union else 0.0

        min_addr_toks = min(len(addr_toks_q), len(addr_toks_t))
        addr_token_overlap_coeff = len(addr_inter) / min_addr_toks if min_addr_toks > 0 else 0.0

        addr_char_ngram_cosine = compute_ngram_cosine(clean_addr_q, clean_addr_t, n=3)

        nums_q = set(extract_numeric_tokens(raw_addr_q))
        nums_t = set(extract_numeric_tokens(raw_addr_t))
        num_union = nums_q.union(nums_t)
        addr_numeric_jaccard = len(nums_q.intersection(nums_t)) / len(num_union) if num_union else 0.0
        addr_numeric_exact_match = 1.0 if nums_q and nums_q == nums_t else 0.0

        postals_q = extract_postal_codes(raw_addr_q)
        postals_t = extract_postal_codes(raw_addr_t)
        if postals_q and postals_t:
            addr_postal_match = 1.0 if postals_q[0] == postals_t[0] else 0.0
        elif not postals_q and not postals_t:
            addr_postal_match = 0.0
        else:
            addr_postal_match = -1.0  # missing on one side

        # -------------------------------------------------------------
        # Group C: Country & Source (G3)
        # -------------------------------------------------------------
        country_exact_match = 1.0 if country_q and country_q == country_t else 0.0
        source_is_s2 = 1.0 if target_eid.startswith("S2-") else 0.0
        source_is_s3 = 1.0 if target_eid.startswith("S3-") else 0.0

        # -------------------------------------------------------------
        # Group D: Cross-Script Transliteration (G6)
        # -------------------------------------------------------------
        indic_q = has_indic_script(raw_name_q)
        indic_t = has_indic_script(raw_name_t)
        is_cross_script = 1.0 if (indic_q != indic_t) else 0.0

        if is_cross_script:
            translit_t = transliterate_indic_to_latin(raw_name_t) if indic_t else clean_name_t
            translit_q = transliterate_indic_to_latin(raw_name_q) if indic_q else clean_name_q
            t_toks_q = set(translit_q.split())
            t_toks_t = set(translit_t.split())
            t_union = t_toks_q.union(t_toks_t)
            translit_name_token_jaccard = len(t_toks_q.intersection(t_toks_t)) / len(t_union) if t_union else 0.0
            translit_name_char_ngram_cosine = compute_ngram_cosine(translit_q, translit_t, n=3)
        else:
            translit_name_token_jaccard = 0.0
            translit_name_char_ngram_cosine = 0.0

        # -------------------------------------------------------------
        # Group E: URL / Domain Normalization (G7)
        # -------------------------------------------------------------
        domain_sig_q = get_compact_domain_signature(raw_name_q, min_length=4)
        domain_sig_t = get_compact_domain_signature(raw_name_t, min_length=4)
        is_domain = 1.0 if ("." in raw_name_q or "." in raw_name_t or "com" in raw_name_t.lower() or "com" in raw_name_q.lower()) else 0.0
        domain_compact_match = 1.0 if domain_sig_q and domain_sig_q == domain_sig_t else 0.0

        # -------------------------------------------------------------
        # Group F: Controlled Leetspeak / Accent (G8)
        # -------------------------------------------------------------
        leet_sig_q = get_leetspeak_signature(raw_name_q, min_length=4)
        leet_sig_t = get_leetspeak_signature(raw_name_t, min_length=4)
        leetspeak_exact_match = 1.0 if leet_sig_q and leet_sig_q == leet_sig_t else 0.0

        accent_q = strip_accents(clean_name_q)
        accent_t = strip_accents(clean_name_t)
        accent_stripped_match = 1.0 if accent_q and accent_q == accent_t else 0.0

        # -------------------------------------------------------------
        # Group G: Frequency / Rarity (G4)
        # -------------------------------------------------------------
        shared_rare = 0.0
        min_idf = 0.0
        if inter_tokens:
            idfs = [self.corpus_idf.get(t, 5.0) for t in inter_tokens]
            shared_rare = float(sum(1 for idf in idfs if idf >= 4.0))
            min_idf = float(min(idfs))

        # -------------------------------------------------------------
        # Group H: Retrieval Provenance (G5)
        # -------------------------------------------------------------
        retrieval_support_count = float(prov.get("support_count", 1))
        retrieved_by_exact_name = 1.0 if prov.get("exact_name", False) else 0.0
        retrieved_by_sorted_name = 1.0 if prov.get("sorted_name", False) else 0.0
        retrieved_by_name_numeric = 1.0 if prov.get("name_numeric", False) else 0.0
        retrieved_by_char_ngram = 1.0 if prov.get("char_ngram", False) else 0.0
        retrieved_by_domain = 1.0 if prov.get("domain", False) else 0.0
        retrieved_by_tfidf = 1.0 if prov.get("tfidf", False) else 0.0
        tfidf_similarity_score = float(prov.get("tfidf_score", 0.0))
        tfidf_rank = float(prov.get("tfidf_rank", 1000.0))

        # -------------------------------------------------------------
        # Group J: Meaningful Interaction Features
        # -------------------------------------------------------------
        inter_name_x_addr_jaccard = name_token_jaccard * addr_token_jaccard
        inter_name_exact_x_addr_jaccard = name_exact_clean * addr_token_jaccard
        inter_name_jaccard_x_country = name_token_jaccard * country_exact_match
        inter_support_x_name_jaccard = math.log1p(retrieval_support_count) * name_token_jaccard

        return {
            "name_exact_clean": name_exact_clean,
            "name_compact_exact": name_compact_exact,
            "name_token_jaccard": name_token_jaccard,
            "name_token_overlap_coeff": name_token_overlap_coeff,
            "name_char_ngram_cosine": name_char_ngram_cosine,
            "name_edit_similarity": name_edit_similarity,
            "name_prefix_match_len": name_prefix_match_len,
            "name_length_ratio": name_length_ratio,
            "name_first_token_match": name_first_token_match,
            "name_sorted_token_jaccard": name_sorted_token_jaccard,
            "addr_exact_clean": addr_exact_clean,
            "addr_token_jaccard": addr_token_jaccard,
            "addr_token_overlap_coeff": addr_token_overlap_coeff,
            "addr_char_ngram_cosine": addr_char_ngram_cosine,
            "addr_numeric_exact_match": addr_numeric_exact_match,
            "addr_numeric_jaccard": addr_numeric_jaccard,
            "addr_postal_match": addr_postal_match,
            "addr_is_missing_either": addr_missing_either,
            "country_exact_match": country_exact_match,
            "source_is_s2": source_is_s2,
            "source_is_s3": source_is_s3,
            "is_cross_script_pair": is_cross_script,
            "translit_name_token_jaccard": translit_name_token_jaccard,
            "translit_name_char_ngram_cosine": translit_name_char_ngram_cosine,
            "is_domain_name_pair": is_domain,
            "domain_compact_match": domain_compact_match,
            "leetspeak_exact_match": leetspeak_exact_match,
            "accent_stripped_match": accent_stripped_match,
            "shared_rare_token_count": shared_rare,
            "min_shared_token_idf": min_idf,
            "retrieval_support_count": retrieval_support_count,
            "retrieved_by_exact_name": retrieved_by_exact_name,
            "retrieved_by_sorted_name": retrieved_by_sorted_name,
            "retrieved_by_name_numeric": retrieved_by_name_numeric,
            "retrieved_by_char_ngram": retrieved_by_char_ngram,
            "retrieved_by_domain": retrieved_by_domain,
            "retrieved_by_tfidf": retrieved_by_tfidf,
            "tfidf_similarity_score": tfidf_similarity_score,
            "tfidf_rank": tfidf_rank,
            "inter_name_x_addr_jaccard": inter_name_x_addr_jaccard,
            "inter_name_exact_x_addr_jaccard": inter_name_exact_x_addr_jaccard,
            "inter_name_jaccard_x_country": inter_name_jaccard_x_country,
            "inter_support_x_name_jaccard": inter_support_x_name_jaccard,
        }

    def extract_matrix(
        self,
        pairs: Sequence[Tuple[Dict[str, Any], Dict[str, Any], Optional[Dict[str, Any]]]],
    ) -> np.ndarray:
        """Extract dense float32 matrix for a batch of candidate pairs."""
        n_pairs = len(pairs)
        n_features = len(self.feature_names)
        matrix = np.zeros((n_pairs, n_features), dtype=np.float32)

        for i, (q_rec, t_rec, prov) in enumerate(pairs):
            feat_dict = self.extract_pair(q_rec, t_rec, prov)
            for j, f_name in enumerate(self.feature_names):
                matrix[i, j] = feat_dict.get(f_name, 0.0)

        # Quality check: replace any unexpected NaNs or infs with 0.0
        np.nan_to_num(matrix, copy=False, nan=0.0, posinf=1.0, neginf=-1.0)
        return matrix
