"""Unit tests for Phase 1.7 representations: domain, transliteration, leetspeak."""

import pytest
from src.representations.domain import clean_domain_name, get_compact_domain_signature
from src.representations.transliteration import (
    has_indic_script,
    transliterate_indic_to_latin,
    get_indic_transliterated_signature,
)
from src.representations.leetspeak import (
    strip_accents,
    normalize_leetspeak_text,
    get_leetspeak_signature,
)


def test_clean_domain_name():
    assert clean_domain_name("foo-bar.com") == "foo bar"
    assert clean_domain_name("indutrust.com") == "indutrust"
    assert clean_domain_name("https://www.pacificalliance.com") == "pacificalliance"
    assert clean_domain_name("2POINT.COM") == "2point"
    assert clean_domain_name("solancemicroelectronicscom") == "solancemicroelectronics"


def test_get_compact_domain_signature():
    assert get_compact_domain_signature("Indu Trust") == "indutrust"
    assert get_compact_domain_signature("Indutrust.Com") == "indutrust"
    assert get_compact_domain_signature("Indu Trust") == get_compact_domain_signature("Indutrust.Com")
    assert get_compact_domain_signature("Pacific Alliance PC") == "pacificalliance"
    assert get_compact_domain_signature("pacificalliance.com") == "pacificalliance"
    # Short signature safeguard
    assert get_compact_domain_signature("A B", min_length=5) == ""


def test_has_indic_script():
    assert not has_indic_script("Eastern International Private Limited")
    assert has_indic_script("ईस्टर्न इंटरनेशनल प्राइवेट लिमिटेड")  # Devanagari
    assert has_indic_script("হাইটেক বিজনেস প্রাইভেট লিমিটেড")      # Bengali
    assert has_indic_script("நார்த் இம்பெக்ஸ்")                    # Tamil
    assert has_indic_script("ಮಾಡರ್ನ್ ಮಾರ್ಕೆಟಿಂಗ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್")      # Kannada
    assert has_indic_script("આણંદ કન્સલ્ટન્સી પ્રાઇવેટ લિમિટેડ")    # Gujarati
    assert has_indic_script("ശിവ് కన్‌స్ట్రక్షన్")                # Telugu


def test_transliterate_indic_to_latin():
    devanagari_translit = transliterate_indic_to_latin("सनराइज")
    assert "san" in devanagari_translit
    bengali_translit = transliterate_indic_to_latin("হাইটেক")
    assert "hai" in bengali_translit or "tek" in bengali_translit


def test_strip_accents():
    assert strip_accents("2-Póint") == "2-Point"
    assert strip_accents("Behavioral Héalth") == "Behavioral Health"
    assert strip_accents("Régius-Decorators") == "Regius-Decorators"


def test_normalize_leetspeak_text():
    assert normalize_leetspeak_text("2-P0int") == "2 point"
    assert normalize_leetspeak_text("2-Póint") == "2 point"
    assert normalize_leetspeak_text("Fa5t Solutions") == "fast solutions"


def test_get_leetspeak_signature():
    sig1 = get_leetspeak_signature("2-Point")
    sig2 = get_leetspeak_signature("2-P0int")
    sig3 = get_leetspeak_signature("2-Póint")
    assert sig1 == sig2 == sig3 == "2point"
