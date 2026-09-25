"""Cross-script transliteration and Indic script detection (Phase 1.7 Exp C).

Provides zero-dependency, rule-based phonetic romanization for Indic scripts
(Devanagari, Bengali, Gurmukhi, Gujarati, Odia, Tamil, Telugu, Kannada, Malayalam).
Does NOT use external business or address databases, complying strictly with
Amazon ML Challenge fair-play guidelines.
"""

from typing import Dict, List, Optional
import re
import unicodedata

# Unicode block ranges for Indic scripts
INDIC_RANGES = [
    (0x0900, 0x097F),  # Devanagari (Hindi, Marathi, Sanskrit, Nepali)
    (0x0980, 0x09FF),  # Bengali, Assamese
    (0x0A00, 0x0A7F),  # Gurmukhi (Punjabi)
    (0x0A80, 0x0AFF),  # Gujarati
    (0x0B00, 0x0B7F),  # Odia
    (0x0B80, 0x0BFF),  # Tamil
    (0x0C00, 0x0C7F),  # Telugu
    (0x0C80, 0x0CFF),  # Kannada
    (0x0D00, 0x0D7F),  # Malayalam
]

# Canonical Brahmic phonetic offset map to Latin phonemes
# All Brahmic scripts share standardized Unicode block offset positions:
# 0x05-0x14: independent vowels
# 0x15-0x39: consonants (velar, palatal, retroflex, dental, labial, liquids, sibilants)
# 0x3E-0x4C: dependent vowel signs (matras)
# 0x4D: virama (vowel killer)
BRAHMIC_OFFSET_MAP: Dict[int, str] = {
    # Independent vowels
    0x05: "a", 0x06: "aa", 0x07: "i", 0x08: "ee", 0x09: "u", 0x0A: "oo",
    0x0B: "ri", 0x0E: "e", 0x0F: "ai", 0x10: "ai", 0x12: "o", 0x13: "au", 0x14: "au",
    # Consonants - Velars
    0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "ng",
    # Palatals
    0x1A: "c", 0x1B: "ch", 0x1C: "j", 0x1D: "jh", 0x1E: "ny",
    # Retroflexes
    0x1F: "t", 0x20: "th", 0x21: "d", 0x22: "dh", 0x23: "n",
    # Dentals
    0x24: "t", 0x25: "th", 0x26: "d", 0x27: "dh", 0x28: "n", 0x29: "nn",
    # Labials
    0x2A: "p", 0x2B: "ph", 0x2C: "b", 0x2D: "bh", 0x2E: "m",
    # Semivowels / Liquids
    0x2F: "y", 0x30: "r", 0x31: "rr", 0x32: "l", 0x33: "l", 0x34: "ll", 0x35: "v",
    # Sibilants / Fricatives
    0x36: "sh", 0x37: "sh", 0x38: "s", 0x39: "h",
    # Dependent vowel signs (matras)
    0x3E: "a", 0x3F: "i", 0x40: "ee", 0x41: "u", 0x42: "oo",
    0x46: "e", 0x47: "e", 0x48: "ai", 0x4A: "o", 0x4B: "o", 0x4C: "au",
    # Virama (vowel suppression)
    0x4D: "",
    # Anusvara & Visarga
    0x02: "n", 0x03: "h",
}


def has_indic_script(text: Optional[str]) -> bool:
    """Return True if text contains characters from any Indic Unicode block."""
    if not text:
        return False
    for char in text:
        cp = ord(char)
        for start, end in INDIC_RANGES:
            if start <= cp <= end:
                return True
    return False


def transliterate_indic_to_latin(text: Optional[str]) -> str:
    """Romanize Indic script text into approximate Latin phonemes.

    Handles consonants, vowels, matras, virama, and inherent 'a' vowels.
    Runs fully offline with zero external network or database dependencies.
    """
    if not text:
        return ""

    out: List[str] = []
    block_bases = [start for start, _ in INDIC_RANGES]
    chars = list(text)
    n = len(chars)

    for i, char in enumerate(chars):
        cp = ord(char)
        matched = False
        for base in block_bases:
            if base <= cp < base + 0x80:
                offset = cp - base
                if offset in BRAHMIC_OFFSET_MAP:
                    mapped = BRAHMIC_OFFSET_MAP[offset]
                    out.append(mapped)
                    # If this character is a consonant (0x15 - 0x39)
                    if 0x15 <= offset <= 0x39:
                        # Check if next char is virama or matra
                        has_modifier = False
                        if i + 1 < n:
                            next_cp = ord(chars[i + 1])
                            for next_base in block_bases:
                                if next_base <= next_cp < next_base + 0x80:
                                    next_offset = next_cp - next_base
                                    # Matras: 0x3E-0x4C, Virama: 0x4D
                                    if (0x3E <= next_offset <= 0x4D) or next_offset in (0x02, 0x03):
                                        has_modifier = True
                                        break
                        # If no matra or virama follows, append inherent 'a'
                        if not has_modifier:
                            out.append("a")
                    matched = True
                    break
        if not matched:
            out.append(char)

    romanized = "".join(out).lower()
    cleaned = re.sub(r"[^\w\s]", " ", romanized)
    return " ".join(cleaned.split())


def get_indic_transliterated_signature(raw_name: Optional[str], min_length: int = 4) -> str:
    """Produce compact signature of transliterated Indic name."""
    if not raw_name or not has_indic_script(raw_name):
        return ""

    latin = transliterate_indic_to_latin(raw_name)
    # Remove common Indic corporate words (pvt, ltd, private, limited, enterprises, etc.)
    # In Indic transliteration: "praaivet limitted" -> "pvt ltd"
    cleaned = re.sub(r"\b(praaivet|praivet|limitted|limited|intrpraaijhis|enterprises|kampanee|company)\b", "", latin)
    compact = "".join(re.findall(r"[a-z0-9]+", cleaned))
    if len(compact) < min_length:
        return ""
    return compact
