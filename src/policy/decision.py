"""Decision policies for entity-level set prediction in Phase 3.

Converts candidate match scores into the final set of S2/S3 IDs for each S1 entity,
maximizing exact Amazon Macro F0.5.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import numpy as np


class BaseDecisionPolicy(ABC):
    """Abstract base class for entity-level decision policies."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def predict_matches(
        self,
        s1_id: str,
        scored_candidates: List[Dict[str, Any]],
    ) -> List[str]:
        """Convert a list of candidate dicts {'target_id': str, 'score': float} to predicted target IDs."""
        pass

    def _sanitize_and_sort(
        self,
        scored_candidates: List[Dict[str, Any]],
    ) -> List[Tuple[str, float]]:
        """Filter for valid S2/S3 IDs, deduplicate by max score, and sort descending."""
        if not scored_candidates:
            return []

        best_scores: Dict[str, float] = {}
        for c in scored_candidates:
            tid = c.get("target_id") or c.get("target_record", {}).get("entity_id", "")
            # Rule 5: S2 and S3 only
            if not (tid.startswith("S2-") or tid.startswith("S3-")):
                continue
            s = float(c.get("score", 0.0))
            if np.isnan(s) or np.isinf(s):
                s = 0.0
            if tid not in best_scores or s > best_scores[tid]:
                best_scores[tid] = s

        # Deterministic sort: score descending, target_id ascending for exact tie-breaking
        sorted_pairs = sorted(best_scores.items(), key=lambda x: (-x[1], x[0]))
        return sorted_pairs

    def predict_cohort(
        self,
        scored_cohort: List[Dict[str, Any]],
    ) -> Dict[str, List[str]]:
        """Predict target IDs for an entire cohort of S1 queries."""
        predictions: Dict[str, List[str]] = {}
        for item in scored_cohort:
            s1_id = item["query_record"]["entity_id"]
            cands = item.get("candidates", [])
            predictions[s1_id] = self.predict_matches(s1_id, cands)
        return predictions


class GlobalThresholdPolicy(BaseDecisionPolicy):
    """Policy A: Accepts any candidate with score >= threshold."""

    def __init__(self, threshold: float = 0.70):
        super().__init__(name=f"GlobalThreshold(th={threshold:.2f})")
        self.threshold = threshold

    def predict_matches(
        self,
        s1_id: str,
        scored_candidates: List[Dict[str, Any]],
    ) -> List[str]:
        sorted_cands = self._sanitize_and_sort(scored_candidates)
        return [tid for tid, s in sorted_cands if s >= self.threshold]


class TopKThresholdPolicy(BaseDecisionPolicy):
    """Policy B: Accepts candidates with score >= threshold up to max_k."""

    def __init__(self, threshold: float = 0.70, max_k: int = 3):
        super().__init__(name=f"TopKThreshold(th={threshold:.2f}, k={max_k})")
        self.threshold = threshold
        self.max_k = max_k

    def predict_matches(
        self,
        s1_id: str,
        scored_candidates: List[Dict[str, Any]],
    ) -> List[str]:
        sorted_cands = self._sanitize_and_sort(scored_candidates)
        accepted = []
        for tid, s in sorted_cands:
            if s >= self.threshold:
                accepted.append(tid)
                if len(accepted) >= self.max_k:
                    break
        return accepted


class RelativeMarginPolicy(BaseDecisionPolicy):
    """Policy C: Top-1 margin rule for precision-preserving multi-match prediction.

    1. If top-1 candidate score < threshold_floor, predict empty set [] (protects singletons).
    2. Accept top-1 candidate.
    3. Accept subsequent candidates if score >= multi_threshold AND (top_1 - score) <= max_margin,
       up to max_k total matches.
    """

    def __init__(
        self,
        threshold_floor: float = 0.70,
        multi_threshold: float = 0.75,
        max_margin: float = 0.10,
        max_k: int = 5,
    ):
        super().__init__(
            name=f"RelativeMargin(floor={threshold_floor:.2f}, multi={multi_threshold:.2f}, margin={max_margin:.2f}, k={max_k})"
        )
        self.threshold_floor = threshold_floor
        self.multi_threshold = multi_threshold
        self.max_margin = max_margin
        self.max_k = max_k

    def predict_matches(
        self,
        s1_id: str,
        scored_candidates: List[Dict[str, Any]],
    ) -> List[str]:
        sorted_cands = self._sanitize_and_sort(scored_candidates)
        if not sorted_cands:
            return []

        top_id, top_score = sorted_cands[0]
        if top_score < self.threshold_floor:
            return []

        accepted = [top_id]
        if self.max_k <= 1:
            return accepted

        for tid, s in sorted_cands[1:]:
            if s >= self.multi_threshold and (top_score - s) <= self.max_margin:
                accepted.append(tid)
                if len(accepted) >= self.max_k:
                    break

        return accepted


class AdaptiveRatioPolicy(BaseDecisionPolicy):
    """Policy D: Ratio-based adaptive candidate selection.

    Accepts top candidate if score >= floor, and any additional candidate if
    score / top_score >= min_ratio.
    """

    def __init__(
        self,
        threshold_floor: float = 0.70,
        min_ratio: float = 0.90,
        max_k: int = 5,
    ):
        super().__init__(name=f"AdaptiveRatio(floor={threshold_floor:.2f}, ratio={min_ratio:.2f}, k={max_k})")
        self.threshold_floor = threshold_floor
        self.min_ratio = min_ratio
        self.max_k = max_k

    def predict_matches(
        self,
        s1_id: str,
        scored_candidates: List[Dict[str, Any]],
    ) -> List[str]:
        sorted_cands = self._sanitize_and_sort(scored_candidates)
        if not sorted_cands:
            return []

        top_id, top_score = sorted_cands[0]
        if top_score < self.threshold_floor:
            return []

        accepted = [top_id]
        if self.max_k <= 1 or top_score <= 0.0:
            return accepted

        for tid, s in sorted_cands[1:]:
            if s >= self.threshold_floor and (s / top_score) >= self.min_ratio:
                accepted.append(tid)
                if len(accepted) >= self.max_k:
                    break

        return accepted
