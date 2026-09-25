"""Feature schema, inventory, and quality validation specifications for Phase 2.

Every pairwise feature must be formally defined with provenance, range, missingness
handling, computational cost, and strict leakage verification.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
import numpy as np


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    group: str
    description: str
    data_source: str
    raw_or_derived: str
    missing_behavior: str
    allowed_range: str
    leakage_assessment: str
    computation_cost: str
    enabled: bool = True


# Master feature registry defining all Phase 2 pairwise features
FEATURE_REGISTRY: List[FeatureSpec] = [
    # -------------------------------------------------------------
    # Group A: Business Name (G1)
    # -------------------------------------------------------------
    FeatureSpec(
        name="name_exact_clean",
        group="G1_name",
        description="Exact equality of normalized legal-suffix-stripped business names",
        data_source="business_name",
        raw_or_derived="derived",
        missing_behavior="0.0 if either name missing",
        allowed_range="[0.0, 1.0]",
        leakage_assessment="Zero leakage; computed purely from query and target strings",
        computation_cost="O(1)",
    ),
    FeatureSpec(
        name="name_compact_exact",
        group="G1_name",
        description="Exact equality of compact alphanumeric names (no whitespace/punctuation)",
        data_source="business_name",
        raw_or_derived="derived",
        missing_behavior="0.0 if either name missing",
        allowed_range="[0.0, 1.0]",
        leakage_assessment="Zero leakage",
        computation_cost="O(1)",
    ),
    FeatureSpec(
        name="name_token_jaccard",
        group="G1_name",
        description="Jaccard similarity over word tokens of stripped business names",
        data_source="business_name",
        raw_or_derived="derived",
        missing_behavior="0.0 if empty token sets",
        allowed_range="[0.0, 1.0]",
        leakage_assessment="Zero leakage",
        computation_cost="O(N + M)",
    ),
    FeatureSpec(
        name="name_token_overlap_coeff",
        group="G1_name",
        description="Overlap coefficient: intersection / min(|tokens_q|, |tokens_t|)",
        data_source="business_name",
        raw_or_derived="derived",
        missing_behavior="0.0 if empty token sets",
        allowed_range="[0.0, 1.0]",
        leakage_assessment="Zero leakage",
        computation_cost="O(N + M)",
    ),
    FeatureSpec(
        name="name_char_ngram_cosine",
        group="G1_name",
        description="Cosine similarity over character 3-gram multiset vectors",
        data_source="business_name",
        raw_or_derived="derived",
        missing_behavior="0.0 if empty string",
        allowed_range="[0.0, 1.0]",
        leakage_assessment="Zero leakage",
        computation_cost="O(L1 + L2)",
    ),
    FeatureSpec(
        name="name_edit_similarity",
        group="G1_name",
        description="Normalized Levenshtein ratio: 1 - distance / max(len1, len2)",
        data_source="business_name",
        raw_or_derived="derived",
        missing_behavior="0.0 if empty string",
        allowed_range="[0.0, 1.0]",
        leakage_assessment="Zero leakage",
        computation_cost="O(L1 * L2) - bounded by max len 60",
    ),
    FeatureSpec(
        name="name_prefix_match_len",
        group="G1_name",
        description="Length of longest common prefix between normalized names normalized by min len",
        data_source="business_name",
        raw_or_derived="derived",
        missing_behavior="0.0 if empty string",
        allowed_range="[0.0, 1.0]",
        leakage_assessment="Zero leakage",
        computation_cost="O(min(L1, L2))",
    ),
    FeatureSpec(
        name="name_length_ratio",
        group="G1_name",
        description="Ratio of min(len1, len2) / max(len1, len2)",
        data_source="business_name",
        raw_or_derived="derived",
        missing_behavior="0.0 if max len 0",
        allowed_range="[0.0, 1.0]",
        leakage_assessment="Zero leakage",
        computation_cost="O(1)",
    ),
    FeatureSpec(
        name="name_first_token_match",
        group="G1_name",
        description="Binary indicator if the first informative name token matches exactly",
        data_source="business_name",
        raw_or_derived="derived",
        missing_behavior="0.0 if tokens missing",
        allowed_range="[0.0, 1.0]",
        leakage_assessment="Zero leakage",
        computation_cost="O(1)",
    ),
    FeatureSpec(
        name="name_sorted_token_jaccard",
        group="G1_name",
        description="Jaccard similarity on word order-invariant sorted tokens",
        data_source="business_name",
        raw_or_derived="derived",
        missing_behavior="0.0 if tokens missing",
        allowed_range="[0.0, 1.0]",
        leakage_assessment="Zero leakage",
        computation_cost="O(N log N)",
    ),

    # -------------------------------------------------------------
    # Group B: Address (G2)
    # -------------------------------------------------------------
    FeatureSpec(
        name="addr_exact_clean",
        group="G2_address",
        description="Exact equality of standard cleaned address strings",
        data_source="business_address",
        raw_or_derived="derived",
        missing_behavior="0.0 if either address missing",
        allowed_range="[0.0, 1.0]",
        leakage_assessment="Zero leakage",
        computation_cost="O(1)",
    ),
    FeatureSpec(
        name="addr_token_jaccard",
        group="G2_address",
        description="Jaccard similarity over word tokens of cleaned addresses",
        data_source="business_address",
        raw_or_derived="derived",
        missing_behavior="0.0 if empty address tokens",
        allowed_range="[0.0, 1.0]",
        leakage_assessment="Zero leakage",
        computation_cost="O(N + M)",
    ),
    FeatureSpec(
        name="addr_token_overlap_coeff",
        group="G2_address",
        description="Overlap coefficient: intersection / min(|addr_q|, |addr_t|)",
        data_source="business_address",
        raw_or_derived="derived",
        missing_behavior="0.0 if empty address tokens",
        allowed_range="[0.0, 1.0]",
        leakage_assessment="Zero leakage",
        computation_cost="O(N + M)",
    ),
    FeatureSpec(
        name="addr_char_ngram_cosine",
        group="G2_address",
        description="Cosine similarity over character 3-grams of address strings",
        data_source="business_address",
        raw_or_derived="derived",
        missing_behavior="0.0 if empty string",
        allowed_range="[0.0, 1.0]",
        leakage_assessment="Zero leakage",
        computation_cost="O(L1 + L2)",
    ),
    FeatureSpec(
        name="addr_numeric_exact_match",
        group="G2_address",
        description="Binary indicator if all numeric plot/street numbers match exactly",
        data_source="business_address",
        raw_or_derived="derived",
        missing_behavior="0.0 if numeric tokens missing",
        allowed_range="[0.0, 1.0]",
        leakage_assessment="Zero leakage",
        computation_cost="O(K1 + K2)",
    ),
    FeatureSpec(
        name="addr_numeric_jaccard",
        group="G2_address",
        description="Jaccard similarity over numeric street and house numbers",
        data_source="business_address",
        raw_or_derived="derived",
        missing_behavior="0.0 if numeric tokens missing",
        allowed_range="[0.0, 1.0]",
        leakage_assessment="Zero leakage",
        computation_cost="O(K1 + K2)",
    ),
    FeatureSpec(
        name="addr_postal_match",
        group="G2_address",
        description="Exact match of 5 or 6 digit postal/PIN codes extracted from address",
        data_source="business_address",
        raw_or_derived="derived",
        missing_behavior="-1.0 (imputed as missing flag) or 0.0",
        allowed_range="[-1.0, 1.0]",
        leakage_assessment="Zero leakage",
        computation_cost="O(1)",
    ),
    FeatureSpec(
        name="addr_is_missing_either",
        group="G2_address",
        description="Binary flag if either query or target address is empty/missing",
        data_source="business_address",
        raw_or_derived="derived",
        missing_behavior="1.0 if missing, 0.0 otherwise",
        allowed_range="[0.0, 1.0]",
        leakage_assessment="Zero leakage",
        computation_cost="O(1)",
    ),

    # -------------------------------------------------------------
    # Group C: Country & Source (G3)
    # -------------------------------------------------------------
    FeatureSpec(
        name="country_exact_match",
        group="G3_country_source",
        description="Open-set exact match indicator between query and candidate country strings",
        data_source="country",
        raw_or_derived="derived",
        missing_behavior="0.0 if missing",
        allowed_range="[0.0, 1.0]",
        leakage_assessment="Zero leakage; completely open-set (US, India, France handled generically)",
        computation_cost="O(1)",
    ),
    FeatureSpec(
        name="source_is_s2",
        group="G3_country_source",
        description="Binary indicator if candidate originates from Source 2",
        data_source="candidate_source",
        raw_or_derived="derived",
        missing_behavior="0.0",
        allowed_range="[0.0, 1.0]",
        leakage_assessment="Zero leakage; describes data origin only",
        computation_cost="O(1)",
    ),
    FeatureSpec(
        name="source_is_s3",
        group="G3_country_source",
        description="Binary indicator if candidate originates from Source 3",
        data_source="candidate_source",
        raw_or_derived="derived",
        missing_behavior="0.0",
        allowed_range="[0.0, 1.0]",
        leakage_assessment="Zero leakage",
        computation_cost="O(1)",
    ),

    # -------------------------------------------------------------
    # Group D: Cross-Script Transliteration (G6)
    # -------------------------------------------------------------
    FeatureSpec(
        name="is_cross_script_pair",
        group="G6_cross_script",
        description="Binary indicator if pair contains one Latin name and one Indic script name",
        data_source="business_name",
        raw_or_derived="derived",
        missing_behavior="0.0",
        allowed_range="[0.0, 1.0]",
        leakage_assessment="Zero leakage; rule-based Unicode character detection",
        computation_cost="O(L)",
    ),
    FeatureSpec(
        name="translit_name_token_jaccard",
        group="G6_cross_script",
        description="Token Jaccard similarity after offline Brahmic transliteration to Latin",
        data_source="business_name",
        raw_or_derived="derived",
        missing_behavior="0.0 if not cross-script or empty",
        allowed_range="[0.0, 1.0]",
        leakage_assessment="Zero leakage; strictly offline zero-dependency romanizer",
        computation_cost="O(L)",
    ),
    FeatureSpec(
        name="translit_name_char_ngram_cosine",
        group="G6_cross_script",
        description="Character 3-gram cosine after offline Brahmic transliteration to Latin",
        data_source="business_name",
        raw_or_derived="derived",
        missing_behavior="0.0 if not cross-script or empty",
        allowed_range="[0.0, 1.0]",
        leakage_assessment="Zero leakage",
        computation_cost="O(L)",
    ),

    # -------------------------------------------------------------
    # Group E: URL / Domain Normalization (G7)
    # -------------------------------------------------------------
    FeatureSpec(
        name="is_domain_name_pair",
        group="G7_domain",
        description="Binary indicator if either name resembles a web domain (.com, .org, .net, etc.)",
        data_source="business_name",
        raw_or_derived="derived",
        missing_behavior="0.0",
        allowed_range="[0.0, 1.0]",
        leakage_assessment="Zero leakage; regex domain pattern matching",
        computation_cost="O(1)",
    ),
    FeatureSpec(
        name="domain_compact_match",
        group="G7_domain",
        description="Exact match of compact signatures after stripping domain protocols/extensions",
        data_source="business_name",
        raw_or_derived="derived",
        missing_behavior="0.0 if no domain or <5 chars",
        allowed_range="[0.0, 1.0]",
        leakage_assessment="Zero leakage",
        computation_cost="O(1)",
    ),

    # -------------------------------------------------------------
    # Group F: Controlled Leetspeak / Accent (G8)
    # -------------------------------------------------------------
    FeatureSpec(
        name="leetspeak_exact_match",
        group="G8_leetspeak",
        description="Exact match of names after controlled internal digit-to-letter normalization",
        data_source="business_name",
        raw_or_derived="derived",
        missing_behavior="0.0 if <4 chars",
        allowed_range="[0.0, 1.0]",
        leakage_assessment="Zero leakage",
        computation_cost="O(L)",
    ),
    FeatureSpec(
        name="accent_stripped_match",
        group="G8_leetspeak",
        description="Exact match of names after stripping Unicode accents/diacritics (NFKD)",
        data_source="business_name",
        raw_or_derived="derived",
        missing_behavior="0.0",
        allowed_range="[0.0, 1.0]",
        leakage_assessment="Zero leakage",
        computation_cost="O(L)",
    ),

    # -------------------------------------------------------------
    # Group G: Frequency / Rarity (G4)
    # -------------------------------------------------------------
    FeatureSpec(
        name="shared_rare_token_count",
        group="G4_rarity",
        description="Count of shared word tokens that are rare in the unsupervised corpus",
        data_source="unsupervised_corpus",
        raw_or_derived="derived",
        missing_behavior="0.0",
        allowed_range="[0, inf)",
        leakage_assessment="Zero leakage; computed from unsupervised corpus token frequencies, no labels",
        computation_cost="O(N + M)",
    ),
    FeatureSpec(
        name="min_shared_token_idf",
        group="G4_rarity",
        description="Minimum corpus IDF among shared name tokens (rarity of shared evidence)",
        data_source="unsupervised_corpus",
        raw_or_derived="derived",
        missing_behavior="0.0 if no shared tokens",
        allowed_range="[0.0, 15.0]",
        leakage_assessment="Zero leakage; derived from unsupervised corpus vocabulary document counts",
        computation_cost="O(N + M)",
    ),

    # -------------------------------------------------------------
    # Group H: Retrieval Provenance (G5)
    # -------------------------------------------------------------
    FeatureSpec(
        name="retrieval_support_count",
        group="G5_provenance",
        description="Total number of independent retrieval blocks that surfaced this candidate",
        data_source="retrieval_pipeline",
        raw_or_derived="derived",
        missing_behavior="1.0 (at least one block)",
        allowed_range="[1, 10]",
        leakage_assessment="Zero leakage; describes candidate generation mechanism only",
        computation_cost="O(1)",
    ),
    FeatureSpec(
        name="retrieved_by_exact_name",
        group="G5_provenance",
        description="Binary indicator if candidate was surfaced by Exact Name block",
        data_source="retrieval_pipeline",
        raw_or_derived="derived",
        missing_behavior="0.0",
        allowed_range="[0.0, 1.0]",
        leakage_assessment="Zero leakage",
        computation_cost="O(1)",
    ),
    FeatureSpec(
        name="retrieved_by_sorted_name",
        group="G5_provenance",
        description="Binary indicator if candidate was surfaced by Sorted Tokens block",
        data_source="retrieval_pipeline",
        raw_or_derived="derived",
        missing_behavior="0.0",
        allowed_range="[0.0, 1.0]",
        leakage_assessment="Zero leakage",
        computation_cost="O(1)",
    ),
    FeatureSpec(
        name="retrieved_by_name_numeric",
        group="G5_provenance",
        description="Binary indicator if candidate was surfaced by Name + Numeric block",
        data_source="retrieval_pipeline",
        raw_or_derived="derived",
        missing_behavior="0.0",
        allowed_range="[0.0, 1.0]",
        leakage_assessment="Zero leakage",
        computation_cost="O(1)",
    ),
    FeatureSpec(
        name="retrieved_by_char_ngram",
        group="G5_provenance",
        description="Binary indicator if candidate was surfaced by Boundary Char N-gram block",
        data_source="retrieval_pipeline",
        raw_or_derived="derived",
        missing_behavior="0.0",
        allowed_range="[0.0, 1.0]",
        leakage_assessment="Zero leakage",
        computation_cost="O(1)",
    ),
    FeatureSpec(
        name="retrieved_by_domain",
        group="G5_provenance",
        description="Binary indicator if candidate was surfaced by Domain Normalization block",
        data_source="retrieval_pipeline",
        raw_or_derived="derived",
        missing_behavior="0.0",
        allowed_range="[0.0, 1.0]",
        leakage_assessment="Zero leakage",
        computation_cost="O(1)",
    ),
    FeatureSpec(
        name="retrieved_by_tfidf",
        group="G5_provenance",
        description="Binary indicator if candidate was surfaced by Sparse TF-IDF retrieval",
        data_source="retrieval_pipeline",
        raw_or_derived="derived",
        missing_behavior="0.0",
        allowed_range="[0.0, 1.0]",
        leakage_assessment="Zero leakage",
        computation_cost="O(1)",
    ),
    FeatureSpec(
        name="tfidf_similarity_score",
        group="G5_provenance",
        description="Cosine similarity score from TF-IDF sparse retrieval (0.0 if not retrieved via TF-IDF)",
        data_source="retrieval_pipeline",
        raw_or_derived="derived",
        missing_behavior="0.0",
        allowed_range="[0.0, 1.0]",
        leakage_assessment="Zero leakage",
        computation_cost="O(1)",
    ),
    FeatureSpec(
        name="tfidf_rank",
        group="G5_provenance",
        description="Rank position in query's top-k TF-IDF retrieval (1 to k; 1000 if not in top-k)",
        data_source="retrieval_pipeline",
        raw_or_derived="derived",
        missing_behavior="1000.0",
        allowed_range="[1.0, 1000.0]",
        leakage_assessment="Zero leakage",
        computation_cost="O(1)",
    ),

    # -------------------------------------------------------------
    # Group J: Meaningful Interaction Features
    # -------------------------------------------------------------
    FeatureSpec(
        name="inter_name_x_addr_jaccard",
        group="G9_interactions",
        description="Interaction product: name_token_jaccard * addr_token_jaccard",
        data_source="derived_interaction",
        raw_or_derived="derived",
        missing_behavior="0.0",
        allowed_range="[0.0, 1.0]",
        leakage_assessment="Zero leakage; joint similarity evidence",
        computation_cost="O(1)",
    ),
    FeatureSpec(
        name="inter_name_exact_x_addr_jaccard",
        group="G9_interactions",
        description="Interaction product: name_exact_clean * addr_token_jaccard",
        data_source="derived_interaction",
        raw_or_derived="derived",
        missing_behavior="0.0",
        allowed_range="[0.0, 1.0]",
        leakage_assessment="Zero leakage",
        computation_cost="O(1)",
    ),
    FeatureSpec(
        name="inter_name_jaccard_x_country",
        group="G9_interactions",
        description="Interaction product: name_token_jaccard * country_exact_match",
        data_source="derived_interaction",
        raw_or_derived="derived",
        missing_behavior="0.0",
        allowed_range="[0.0, 1.0]",
        leakage_assessment="Zero leakage",
        computation_cost="O(1)",
    ),
    FeatureSpec(
        name="inter_support_x_name_jaccard",
        group="G9_interactions",
        description="Interaction: log(1 + retrieval_support_count) * name_token_jaccard",
        data_source="derived_interaction",
        raw_or_derived="derived",
        missing_behavior="0.0",
        allowed_range="[0.0, 5.0]",
        leakage_assessment="Zero leakage",
        computation_cost="O(1)",
    ),
]


def get_feature_names(enabled_only: bool = True) -> List[str]:
    """Return ordered list of feature names."""
    if enabled_only:
        return [f.name for f in FEATURE_REGISTRY if f.enabled]
    return [f.name for f in FEATURE_REGISTRY]


def get_feature_by_name(name: str) -> Optional[FeatureSpec]:
    """Look up specification by feature name."""
    for f in FEATURE_REGISTRY:
        if f.name == name:
            return f
    return None


def export_feature_inventory_csv(output_path: str) -> None:
    """Export feature inventory table to CSV according to Phase 2 specification."""
    import csv
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "feature_name", "group", "description", "data_source",
            "raw_or_derived", "missing_behavior", "allowed_range",
            "leakage_risk", "computation_cost", "enabled"
        ])
        for f_spec in FEATURE_REGISTRY:
            writer.writerow([
                f_spec.name, f_spec.group, f_spec.description, f_spec.data_source,
                f_spec.raw_or_derived, f_spec.missing_behavior, f_spec.allowed_range,
                f_spec.leakage_assessment, f_spec.computation_cost, str(f_spec.enabled)
            ])
