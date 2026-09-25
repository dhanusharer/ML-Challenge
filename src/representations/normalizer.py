"""Core text normalization primitives for Phase 1 candidate generation.

Supports Unicode normalization, case folding, punctuation handling,
alphanumeric filtering, and whitespace collapsing without altering raw data.
"""

import re
import unicodedata
from typing import List, Pattern

# Compiled regex patterns for speed
RE_WHITESPACE: Pattern = re.compile(r"\s+")
RE_PUNCTUATION: Pattern = re.compile(r"[^\w\s]", re.UNICODE)
RE_NON_ALPHANUMERIC: Pattern = re.compile(r"[^\w\s]", re.UNICODE)
RE_DIGITS: Pattern = re.compile(r"\b\d+\b")
RE_POSTAL_CODE: Pattern = re.compile(r"\b\d{5,6}\b")  # US 5-digit, India 6-digit, France 5-digit


def normalize_unicode(text: str, form: str = "NFKC") -> str:
    """Normalize Unicode characters (e.g., compatibility characters, accents)."""
    if not text:
        return ""
    return unicodedata.normalize(form, text)


def normalize_whitespace(text: str) -> str:
    """Strip leading/trailing whitespace and collapse internal whitespace."""
    if not text:
        return ""
    return RE_WHITESPACE.sub(" ", text).strip()


def normalize_case(text: str) -> str:
    """Convert text to lowercase."""
    if not text:
        return ""
    return text.lower()


def remove_punctuation(text: str) -> str:
    """Replace punctuation characters with spaces."""
    if not text:
        return ""
    return RE_PUNCTUATION.sub(" ", text)


def normalize_alphanumeric(text: str) -> str:
    """Retain only alphanumeric characters and spaces."""
    if not text:
        return ""
    cleaned = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return normalize_whitespace(cleaned)


def standard_clean(text: str) -> str:
    """Standard baseline cleaning pipeline: Unicode NFKC -> Lowercase -> Possessives stripped -> No Punctuation -> Collapsed Whitespace."""
    if not text:
        return ""
    norm = normalize_unicode(text, "NFKC")
    lowered = normalize_case(norm)
    no_possessive = re.sub(r"['’]s\b", "", lowered)
    no_punct = remove_punctuation(no_possessive)
    return normalize_whitespace(no_punct)

