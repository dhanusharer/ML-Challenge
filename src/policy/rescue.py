"""Adaptive retrieval rescue pass coordinator (charter §15, §16).

Implements two-pass cascading candidate generation:
Pass 1: Fast, low-volume Candidate Arm B (~44 cands/query)
Trigger: If query has no high-confidence candidate (top_score < rescue_threshold)
Pass 2: Selective recovery via Candidate Arm C (top-k=50 + cross-script + leetspeak)
"""

from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import time
import numpy as np


class AdaptiveRescueCoordinator:
    """Manages adaptive cascading retrieval between Candidate Arm B and Arm C."""

    def __init__(
        self,
        rescue_threshold: float = 0.70,
        rescue_margin: float = 0.05,
    ):
        self.rescue_threshold = rescue_threshold
        self.rescue_margin = rescue_margin

    def should_rescue_query(
        self,
        scored_candidates: List[Dict[str, Any]],
    ) -> bool:
        """Determine whether an entity should trigger expensive secondary retrieval."""
        if not scored_candidates:
            return True

        scores = [c.get("score", 0.0) for c in scored_candidates]
        max_score = max(scores) if scores else 0.0

        # Trigger 1: No candidate meets the confidence floor
        if max_score < self.rescue_threshold:
            return True

        # Trigger 2: High ambiguity (top two candidates within tiny margin and moderate score)
        if len(scores) >= 2:
            sorted_scores = sorted(scores, reverse=True)
            if sorted_scores[0] < 0.85 and (sorted_scores[0] - sorted_scores[1]) <= self.rescue_margin:
                return True

        return False

    def merge_candidate_lists(
        self,
        primary_cands: List[Dict[str, Any]],
        rescue_cands: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Merge primary and rescue candidates, updating scores and provenance."""
        seen_tids: Dict[str, Dict[str, Any]] = {}

        for c in primary_cands:
            tid = c.get("target_id") or c.get("target_record", {}).get("entity_id", "")
            seen_tids[tid] = c

        for c in rescue_cands:
            tid = c.get("target_id") or c.get("target_record", {}).get("entity_id", "")
            if tid not in seen_tids:
                seen_tids[tid] = c
            else:
                # Keep higher score if re-scored
                if c.get("score", 0.0) > seen_tids[tid].get("score", 0.0):
                    seen_tids[tid]["score"] = c["score"]

        return list(seen_tids.values())
