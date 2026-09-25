"""Feature engineering package for Phase 2 candidate matching."""

from src.features.schema import (
    FeatureSpec,
    FEATURE_REGISTRY,
    get_feature_names,
    get_feature_by_name,
    export_feature_inventory_csv,
)
from src.features.extractor import (
    PairwiseFeatureExtractor,
    compute_levenshtein_ratio,
    compute_ngram_cosine,
)

__all__ = [
    "FeatureSpec",
    "FEATURE_REGISTRY",
    "get_feature_names",
    "get_feature_by_name",
    "export_feature_inventory_csv",
    "PairwiseFeatureExtractor",
    "compute_levenshtein_ratio",
    "compute_ngram_cosine",
]
