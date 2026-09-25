"""Business address representation generators for Phase 1 candidate generation.

Extracts address features and structural tokens:
1. Standard cleaned address
2. Numeric tokens (house numbers, suite numbers, street numbers)
3. Postal / PIN code tokens (US 5-digit, India 6-digit, France 5-digit)
4. Address word tokens
5. Missing address detection
"""

import re
from typing import List, Optional
from src.representations.normalizer import standard_clean

# Regex matching 5 or 6 digit standalone postal / PIN codes
RE_POSTAL_CODE = re.compile(r"\b\d{5,6}\b")
# Regex matching any integer numeric token (house numbers, plot numbers, etc.)
RE_NUMERIC = re.compile(r"\b\d+\b")


def is_address_missing(raw_address: Optional[str]) -> bool:
    """Return True if raw_address is None, empty, or whitespace-only."""
    if raw_address is None:
        return True
    return len(raw_address.strip()) == 0


def get_address_tokens(raw_address: Optional[str]) -> List[str]:
    """Tokenize cleaned address."""
    if is_address_missing(raw_address):
        return []
    cleaned = standard_clean(raw_address)
    return [t for t in cleaned.split() if t]


def extract_postal_codes(raw_address: Optional[str]) -> List[str]:
    """Extract 5 or 6 digit postal codes (e.g. US 5-digit zip, Indian 6-digit PIN, France 5-digit)."""
    if is_address_missing(raw_address):
        return []
    return RE_POSTAL_CODE.findall(raw_address)


def extract_numeric_tokens(raw_address: Optional[str]) -> List[str]:
    """Extract all numeric components (e.g. house numbers, building numbers)."""
    if is_address_missing(raw_address):
        return []
    return RE_NUMERIC.findall(raw_address)


def get_sorted_tokens_address(raw_address: Optional[str]) -> str:
    """Produce sorted address tokens invariant to address component reordering."""
    if is_address_missing(raw_address):
        return ""
    tokens = sorted(get_address_tokens(raw_address))
    return " ".join(tokens)
