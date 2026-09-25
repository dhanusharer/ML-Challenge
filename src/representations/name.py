"""Business name representation generators for Phase 1 candidate generation.

Transforms business names into diverse representations:
1. Standard cleaned (Unicode + Lowercase + Punctuation removed)
2. Legal-suffix-stripped representation (US, India, France legal forms)
3. Tokenized words
4. Sorted-token representation (permutation invariant)
5. Character n-grams (3-grams, 4-grams)
6. Token n-grams (bi-grams)
"""

from typing import List, Set
import re
from src.representations.normalizer import standard_clean

# Comprehensive legal forms covering US, India, and France
LEGAL_SUFFIXES: Set[str] = {
    # US / UK / International
    "inc", "incorporated", "corp", "corporation", "co", "company", "llc", "l.l.c",
    "llp", "l.l.p", "ltd", "limited", "lp", "plc", "enterprises", "holdings",
    "group", "services", "solutions", "international", "intl", "industries",
    # India
    "pvt", "private", "pvt ltd", "private limited", "proprietorship", "partnership",
    # France
    "sarl", "s.a.r.l", "sas", "s.a.s", "sasu", "s.a.s.u", "sa", "s.a", "eurl", "snc", "scop", "ste", "societe", "gie"
}

# Regex to strip trailing/leading legal forms and noise tokens
_LEGAL_SUFFIX_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(s) for s in sorted(LEGAL_SUFFIXES, key=len, reverse=True)) + r")\b",
    re.IGNORECASE
)


def get_name_tokens(raw_name: str) -> List[str]:
    """Extract alphanumeric tokens from business name."""
    cleaned = standard_clean(raw_name)
    if not cleaned:
        return []
    return [t for t in cleaned.split() if t]


def strip_legal_suffixes(raw_name: str) -> str:
    """Remove standard legal entity suffixes from business name."""
    cleaned = standard_clean(raw_name)
    if not cleaned:
        return ""
    stripped = _LEGAL_SUFFIX_PATTERN.sub(" ", cleaned)
    tokens = [t for t in stripped.split() if t]
    return " ".join(tokens)


def get_sorted_tokens_name(raw_name: str, strip_legal: bool = True) -> str:
    """Produce sorted-token signature invariant to word-order transposition."""
    text = strip_legal_suffixes(raw_name) if strip_legal else standard_clean(raw_name)
    tokens = sorted(text.split())
    return " ".join(tokens)


def get_char_ngrams(raw_name: str, n: int = 3, strip_legal: bool = False) -> List[str]:
    """Generate character n-grams from cleaned business name."""
    text = strip_legal_suffixes(raw_name) if strip_legal else standard_clean(raw_name)
    text_compact = text.replace(" ", "")
    if len(text_compact) < n:
        return [text_compact] if text_compact else []
    return [text_compact[i:i + n] for i in range(len(text_compact) - n + 1)]


def get_token_ngrams(raw_name: str, n: int = 2) -> List[str]:
    """Generate adjacent token n-grams (e.g., word bi-grams)."""
    tokens = get_name_tokens(raw_name)
    if len(tokens) < n:
        return [" ".join(tokens)] if tokens else []
    return [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
