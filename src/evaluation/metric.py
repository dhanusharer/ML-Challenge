"""Exact Amazon-Style Macro F_0.5 Evaluator.

Authoritative specification from Amazon ML Challenge 2026 problem statement:
- Metric: F_β Score with β = 0.5 (precision-heavy).
- Formula:
    F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
- Aggregation: Macro-averaged across ALL Source-1 entities in the evaluation set.
- Singleton Handling:
    - If true matches == [] and predicted == [] => Entity F_0.5 = 1.0 (Credit for correct singleton)
    - If true matches == [] and predicted != [] => Entity F_0.5 = 0.0 (Penalized for false merge)
- Non-Singleton Handling:
    - If true matches != [] and predicted == [] => Entity F_0.5 = 0.0
    - If true matches != [] and predicted != []:
        TP = len(set(true) & set(predicted))
        Precision = TP / len(predicted)
        Recall = TP / len(true)
        If TP == 0: F_0.5 = 0.0
        Else: F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
"""

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union
import json
import numpy as np


@dataclass(frozen=True)
class EvaluationResult:
    """Full diagnostic results for entity resolution evaluation."""
    macro_f05: float
    mean_precision: float
    mean_recall: float
    num_s1_entities: int
    num_true_singletons: int
    num_correct_singletons: int
    num_false_singleton_merges: int
    num_entities_with_predictions: int
    total_predicted_links: int
    total_true_links: int
    mean_predicted_links_per_s1: float
    mean_true_links_per_s1: float

    def to_dict(self) -> Dict[str, Union[float, int]]:
        return asdict(self)

    def summary(self) -> str:
        return (
            f"=== EVALUATION RESULTS (Macro F_0.5) ===\n"
            f"Macro F_0.5 Score:             {self.macro_f05:.6f}\n"
            f"Mean Precision:                {self.mean_precision:.6f}\n"
            f"Mean Recall:                   {self.mean_recall:.6f}\n"
            f"Total S1 Entities:             {self.num_s1_entities:,}\n"
            f"True Singletons:               {self.num_true_singletons:,}\n"
            f"Correct Singletons:            {self.num_correct_singletons:,}\n"
            f"False Singleton Merges:        {self.num_false_singleton_merges:,}\n"
            f"Entities with Predictions:     {self.num_entities_with_predictions:,}\n"
            f"Total True Links:              {self.total_true_links:,}\n"
            f"Total Predicted Links:         {self.total_predicted_links:,}\n"
            f"Mean True Links / S1:          {self.mean_true_links_per_s1:.4f}\n"
            f"Mean Predicted Links / S1:     {self.mean_predicted_links_per_s1:.4f}\n"
        )


def evaluate_entity_f05(
    true_ids: Sequence[str],
    predicted_ids: Sequence[str],
    check_duplicates: bool = True,
) -> Tuple[float, float, float]:
    """Evaluate F_0.5, Precision, and Recall for a single Source-1 entity.

    Args:
        true_ids: Ground truth matched entity IDs (S2/S3). Empty for singletons.
        predicted_ids: Model predicted matched entity IDs. Empty for singletons.
        check_duplicates: If True, raises ValueError if predicted_ids has duplicates.

    Returns:
        Tuple of (f05, precision, recall) as floats in [0.0, 1.0].

    Raises:
        ValueError: If duplicate IDs are found when check_duplicates is True.
    """
    if check_duplicates and len(predicted_ids) != len(set(predicted_ids)):
        raise ValueError(
            f"Duplicate predicted entity IDs detected in list: {predicted_ids}. "
            f"Duplicates are strictly forbidden by challenge rules."
        )

    num_true = len(true_ids)
    num_pred = len(predicted_ids)

    # Edge Case 1: True Singleton
    if num_true == 0:
        if num_pred == 0:
            # Correctly predicted singleton: full credit 1.0
            return 1.0, 1.0, 1.0
        else:
            # False merge on singleton: penalized 0.0
            return 0.0, 0.0, 0.0

    # Edge Case 2: True non-singleton, but prediction is empty
    if num_pred == 0:
        return 0.0, 0.0, 0.0

    # Edge Case 3: Both non-empty
    true_set = set(true_ids)
    pred_set = set(predicted_ids)
    true_positives = len(true_set & pred_set)

    precision = true_positives / num_pred
    recall = true_positives / num_true

    if true_positives == 0:
        return 0.0, precision, recall

    denominator = 0.25 * precision + recall
    if denominator == 0.0:
        return 0.0, precision, recall

    f05 = (1.25 * precision * recall) / denominator
    return f05, precision, recall


def evaluate_f05(
    ground_truth: Dict[str, Sequence[str]],
    predictions: Dict[str, Sequence[str]],
    strict_entity_coverage: bool = True,
    check_duplicates: bool = True,
) -> EvaluationResult:
    """Compute Macro-Averaged F_0.5 and comprehensive diagnostics across all Source-1 entities.

    Args:
        ground_truth: Mapping of {source1_entity_id: [matched_entity_ids]}.
        predictions: Mapping of {source1_entity_id: [matched_entity_ids]}.
        strict_entity_coverage: If True, raises KeyError if any ground_truth entity is missing from predictions.
        check_duplicates: If True, rejects predictions containing duplicate IDs.

    Returns:
        EvaluationResult containing macro F_0.5, mean precision, mean recall, and diagnostics.

    Raises:
        ValueError: If ground_truth is empty or predictions violate integrity.
        KeyError: If strict_entity_coverage is True and predictions are missing required S1 entities.
    """
    if not ground_truth:
        raise ValueError("ground_truth dictionary is empty. Cannot evaluate.")

    gt_entities = set(ground_truth.keys())
    pred_entities = set(predictions.keys())

    missing_entities = gt_entities - pred_entities
    if missing_entities and strict_entity_coverage:
        sample_missing = sorted(list(missing_entities))[:5]
        raise KeyError(
            f"Predictions missing {len(missing_entities)} required Source-1 entities. "
            f"Samples: {sample_missing}. Every S1 entity must appear (empty list for singletons)."
        )

    f05_scores: List[float] = []
    precisions: List[float] = []
    recalls: List[float] = []

    num_true_singletons = 0
    num_correct_singletons = 0
    num_false_singleton_merges = 0
    num_entities_with_predictions = 0
    total_predicted_links = 0
    total_true_links = 0

    for s1_id, true_matches in ground_truth.items():
        pred_matches = predictions.get(s1_id, [])

        num_t = len(true_matches)
        num_p = len(pred_matches)

        total_true_links += num_t
        total_predicted_links += num_p
        if num_p > 0:
            num_entities_with_predictions += 1

        if num_t == 0:
            num_true_singletons += 1
            if num_p == 0:
                num_correct_singletons += 1
            else:
                num_false_singleton_merges += 1

        f05, prec, rec = evaluate_entity_f05(
            true_matches,
            pred_matches,
            check_duplicates=check_duplicates,
        )

        f05_scores.append(f05)
        precisions.append(prec)
        recalls.append(rec)

    n_entities = len(ground_truth)
    macro_f05 = float(np.mean(f05_scores))
    mean_precision = float(np.mean(precisions))
    mean_recall = float(np.mean(recalls))

    return EvaluationResult(
        macro_f05=macro_f05,
        mean_precision=mean_precision,
        mean_recall=mean_recall,
        num_s1_entities=n_entities,
        num_true_singletons=num_true_singletons,
        num_correct_singletons=num_correct_singletons,
        num_false_singleton_merges=num_false_singleton_merges,
        num_entities_with_predictions=num_entities_with_predictions,
        total_predicted_links=total_predicted_links,
        total_true_links=total_true_links,
        mean_predicted_links_per_s1=total_predicted_links / n_entities if n_entities > 0 else 0.0,
        mean_true_links_per_s1=total_true_links / n_entities if n_entities > 0 else 0.0,
    )
