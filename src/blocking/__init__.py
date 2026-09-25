"""Candidate generation, blocking strategies, and union analyzers."""

from src.blocking.base import BaseBlocker, BlockingMetrics
from src.blocking.exact_name import ExactNameBlocker
from src.blocking.token_buckets import SortedTokenNameBlocker, RareTokenNameBlocker
from src.blocking.char_ngrams import CharNgramBoundaryBlocker
from src.blocking.address_blocks import ExactAddressBlocker, PostalCodeBlocker
from src.blocking.hybrids import NamePostalHybridBlocker, NameNumericHybridBlocker
from src.blocking.sparse_tfidf import SparseTfidfBlocker
from src.blocking.union import analyze_complementarity, evaluate_union, merge_candidate_dicts

__all__ = [
    "BaseBlocker",
    "BlockingMetrics",
    "ExactNameBlocker",
    "SortedTokenNameBlocker",
    "RareTokenNameBlocker",
    "CharNgramBoundaryBlocker",
    "ExactAddressBlocker",
    "PostalCodeBlocker",
    "NamePostalHybridBlocker",
    "NameNumericHybridBlocker",
    "SparseTfidfBlocker",
    "analyze_complementarity",
    "evaluate_union",
    "merge_candidate_dicts",
]
