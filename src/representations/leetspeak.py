"""Controlled leetspeak and character substitution normalization (Phase 1.7 Exp E).

Normalizes character substitutions:
- 0 <-> o
- 1 <-> l / i
- 5 <-> s
- Accent / diacritic stripping (ó -> o, é -> e, etc.)

Safeguard: Evaluated strictly as a bounded candidate generator with length
thresholds, rather than universal uncontrolled replacement that causes candidate explosion.
"""

from typing import List, Optional, Set
import re
import unicodedata

from src.representations.normalizer import standard_clean


def strip_accents(text: str) -> str:
    """Strip Unicode accents/diacritics (e.g. 'Póint' -> 'Point', 'Héalth' -> 'Health')."""
    if not text:
        return ""
    norm = unicodedata.normalize("NFKD", text)
    return "".join(c for c in norm if not unicodedata.combining(c))


def normalize_leetspeak_text(text: str) -> str:
    """Normalize internal digit substitutions within words.

    Examples:
        '2-P0int' -> '2-Point'
        'S0lutions' -> 'Solutions'
        'Fa5t' -> 'Fast'
    """
    if not text:
        return ""

    lowered = text.lower()
    # Strip accents
    cleaned = strip_accents(lowered)

    # Substitute 0 with o only when flanked by letters or at end of a word of letters
    cleaned = re.sub(r"(?<=[a-z])0(?=[a-z])", "o", cleaned)
    cleaned = re.sub(r"(?<=[a-z])0\b", "o", cleaned)
    cleaned = re.sub(r"\b0(?=[a-z]{2,})", "o", cleaned)

    # Substitute 5 with s only when flanked by letters or in letter words
    cleaned = re.sub(r"(?<=[a-z])5(?=[a-z])", "s", cleaned)
    cleaned = re.sub(r"(?<=[a-z])5\b", "s", cleaned)
    cleaned = re.sub(r"\b5(?=[a-z]{2,})", "s", cleaned)

    # Collapse hyphens, dots, and whitespace
    cleaned = re.sub(r"[\-_\.]+", " ", cleaned)
    return " ".join(cleaned.split())


def get_leetspeak_signature(raw_name: str, min_length: int = 4) -> str:
    """Generate compact leetspeak signature for candidate blocking.

    Enforces min_length threshold to prevent distractor explosions.
    """
    norm = normalize_leetspeak_text(raw_name)
    tokens = re.findall(r"[a-z0-9]+", norm)
    compact = "".join(tokens)
    if len(compact) < min_length:
        return ""
    return compact
