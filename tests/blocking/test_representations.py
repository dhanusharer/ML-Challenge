"""Unit tests for text and entity representation generators."""

import pytest
from src.representations.normalizer import (
    normalize_unicode,
    normalize_whitespace,
    normalize_case,
    remove_punctuation,
    normalize_alphanumeric,
    standard_clean,
)
from src.representations.name import (
    get_name_tokens,
    strip_legal_suffixes,
    get_sorted_tokens_name,
    get_char_ngrams,
    get_token_ngrams,
)
from src.representations.address import (
    is_address_missing,
    get_address_tokens,
    extract_postal_codes,
    extract_numeric_tokens,
    get_sorted_tokens_address,
)


def test_standard_clean():
    raw = "  Acme,   Inc. & Sons (Trading) \t"
    cleaned = standard_clean(raw)
    assert cleaned == "acme inc sons trading"


def test_legal_suffix_stripping_us_india_france():
    # US
    assert strip_legal_suffixes("Google Inc.") == "google"
    assert strip_legal_suffixes("Amazon Corporation") == "amazon"
    assert strip_legal_suffixes("Meta Platforms LLC") == "meta platforms"
    # India
    assert strip_legal_suffixes("Reliance Industries Private Limited") == "reliance"
    assert strip_legal_suffixes("Tata Motors Pvt Ltd") == "tata motors"
    # France
    assert strip_legal_suffixes("Boulangerie Paris SARL") == "boulangerie paris"
    assert strip_legal_suffixes("Renault SAS") == "renault"
    assert strip_legal_suffixes("Airbus SA") == "airbus"


def test_sorted_tokens_name_permutation_invariance():
    name1 = "National State Bank of India"
    name2 = "Bank of India State National"
    sig1 = get_sorted_tokens_name(name1)
    sig2 = get_sorted_tokens_name(name2)
    assert sig1 == sig2
    assert sig1 == "bank india national of state"


def test_char_ngrams():
    ngrams = get_char_ngrams("Acme", n=3)
    assert ngrams == ["acm", "cme"]

    # Short string
    assert get_char_ngrams("Ab", n=3) == ["ab"]


def test_address_missing_handling():
    assert is_address_missing(None) is True
    assert is_address_missing("") is True
    assert is_address_missing("   \t\n") is True
    assert is_address_missing("123 Main St") is False

    assert get_address_tokens("") == []
    assert extract_postal_codes(None) == []
    assert extract_numeric_tokens("") == []
    assert get_sorted_tokens_address("") == ""


def test_postal_code_extraction():
    # US 5-digit
    addr_us = "1795 Westchester Drive, High Point, NC 27262"
    assert extract_postal_codes(addr_us) == ["27262"]

    # India 6-digit
    addr_in = "KH NO. -570/13, NEW DELHI 110059"
    assert extract_postal_codes(addr_in) == ["110059"]

    # France 5-digit
    addr_fr = "63 R. DE DIEPPE, LILLE 59000"
    assert extract_postal_codes(addr_fr) == ["59000"]


def test_numeric_token_extraction():
    addr = "KH NO. -570/13, NEW DELHI 110059, Plot 42"
    nums = extract_numeric_tokens(addr)
    assert nums == ["570", "13", "110059", "42"]
