"""Domain and URL normalization for candidate generation (Phase 1.7 Exp D).

Normalizes domain-formatted business names (e.g. 'indutrust.com', 'pacificalliance.com',
'2POINT.COM', 'audieswansonesq.com', 'solancemicroelectronicscom') into clean lexical
and compact representations to recover true matches.
"""

from typing import List, Optional, Set
import re
import unicodedata

from src.representations.name import strip_legal_suffixes, LEGAL_SUFFIXES
from src.representations.normalizer import standard_clean

# Common top-level and second-level domain extensions
DOMAIN_EXTENSIONS: List[str] = [
    ".co.in", ".com.au", ".co.uk", ".com", ".org", ".net", ".in", ".io",
    ".co", ".biz", ".info", ".us", ".tv", ".cc", ".fr", ".de", ".eu",
    ".online", ".tech", ".store", ".site", ".group", ".global", ".cloud"
]

# Extended legal/corporate noise tokens including 'pc', 'esq', etc.
EXTENDED_LEGAL_TOKENS: Set[str] = LEGAL_SUFFIXES.union({
    "pc", "p.c", "esq", "esquire", "dr", "md", "phd", "llp", "pllc"
})


def clean_domain_name(raw_name: str) -> str:
    """Normalize web domain names into spaced words.

    Examples:
        'foo-bar.com' -> 'foo bar'
        'indutrust.com' -> 'indutrust'
        'https://www.pacificalliance.com' -> 'pacificalliance'
        'solancemicroelectronicscom' -> 'solancemicroelectronics'
    """
    if not raw_name:
        return ""

    text = raw_name.lower().strip()
    # Strip URL schemes and subdomains
    text = re.sub(r"^https?://", "", text)
    text = re.sub(r"^www\.", "", text)

    # Strip explicit domain extensions
    for ext in DOMAIN_EXTENSIONS:
        if text.endswith(ext):
            text = text[:-len(ext)]
            break

    # Strip concatenated trailing 'com' if preceded by at least 6 letters
    if text.endswith("com") and len(text) > 7:
        text = text[:-3]

    # Replace punctuation (hyphens, dots, underscores, slashes) with spaces
    text = re.sub(r"[\-_\.\/]+", " ", text)
    return " ".join(text.split())


def get_compact_domain_signature(raw_name: str, min_length: int = 5) -> str:
    """Produce compact alphanumeric signature without legal suffixes or spaces.

    Safeguard: Returns empty string if signature is shorter than min_length
    to prevent distractor collisions on short names.

    Examples:
        'Indu Trust' -> 'indutrust'
        'Indutrust.Com' -> 'indutrust'
        'Pacific Alliance PC' -> 'pacificalliance'
        'pacificalliance.com' -> 'pacificalliance'
        '2-Point' -> '2point'
        '2POINT.COM' -> '2point'
    """
    cleaned = clean_domain_name(raw_name)
    if not cleaned:
        return ""

    # Normalize Unicode accents (NFKD decompose and strip combining marks)
    norm = unicodedata.normalize("NFKD", cleaned)
    no_accents = "".join(c for c in norm if not unicodedata.combining(c))

    # Tokenize and filter extended legal forms
    tokens = [t for t in re.findall(r"[a-z0-9]+", no_accents.lower()) if t not in EXTENDED_LEGAL_TOKENS]
    signature = "".join(tokens)

    if len(signature) < min_length:
        return ""
    return signature
