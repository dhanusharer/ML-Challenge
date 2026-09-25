"""Base blocker interface, candidate storage, and candidate-evaluation metrics.

Authoritative metrics:
- Blocking Recall evaluated strictly at the TRUE LINK level:
    Recall = (Total true links captured in candidate sets) / (Total true links)
- Separate S2 and S3 recall tracking
- Candidate distribution: mean, median, P90, P95, P99, max, total pairs
- Reduction ratio: fraction of pairwise search space eliminated
"""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
import numpy as np
import time


@dataclass
class BlockingMetrics:
    """Quantitative performance and scale metrics for candidate generation."""
    block_name: str
    total_s1_evaluated: int
    total_true_links: int
    captured_true_links: int
    blocking_recall: float
    s2_true_links: int
    s2_captured_links: int
    s2_recall: float
    s3_true_links: int
    s3_captured_links: int
    s3_recall: float
    total_candidates: int
    mean_candidates_per_s1: float
    median_candidates_per_s1: float
    p90_candidates_per_s1: float
    p95_candidates_per_s1: float
    p99_candidates_per_s1: float
    max_candidates_per_s1: int
    reduction_ratio: float
    indexing_time_sec: float
    query_time_sec: float
    total_time_sec: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BaseBlocker(ABC):
    """Abstract base class for all candidate generation / blocking strategies."""

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self.indexing_time_sec: float = 0.0
        self.query_time_sec: float = 0.0

    @abstractmethod
    def fit(self, targets: Sequence[Dict[str, str]]) -> "BaseBlocker":
        """Index target records (from Source 2 and Source 3)."""
        pass

    @abstractmethod
    def retrieve(self, queries: Sequence[Dict[str, str]]) -> Dict[str, Set[str]]:
        """Retrieve candidate target IDs for each query (Source 1) record."""
        pass

    def evaluate(
        self,
        queries: Sequence[Dict[str, str]],
        ground_truth: Dict[str, Sequence[str]],
        total_target_records: int,
    ) -> Tuple[BlockingMetrics, Dict[str, Set[str]]]:
        """Run retrieval and evaluate candidate quality and scale at the true link level."""
        t0 = time.time()
        candidates = self.retrieve(queries)
        self.query_time_sec = round(time.time() - t0, 3)

        total_true = 0
        captured_true = 0
        s2_true = 0
        s2_captured = 0
        s3_true = 0
        s3_captured = 0

        candidate_counts: List[int] = []

        for q in queries:
            s1_id = q["entity_id"]
            true_mids = set(ground_truth.get(s1_id, []))
            pred_cands = candidates.get(s1_id, set())

            candidate_counts.append(len(pred_cands))

            for mid in true_mids:
                total_true += 1
                is_captured = mid in pred_cands
                if is_captured:
                    captured_true += 1

                if mid.startswith("S2-"):
                    s2_true += 1
                    if is_captured:
                        s2_captured += 1
                elif mid.startswith("S3-"):
                    s3_true += 1
                    if is_captured:
                        s3_captured += 1

        total_s1 = len(queries)
        counts_arr = np.array(candidate_counts) if candidate_counts else np.array([0])
        total_cands = int(np.sum(counts_arr))

        # Reduction ratio relative to Cartesian product (queries x total_target_records)
        cartesian_space = total_s1 * total_target_records if total_target_records > 0 else 1
        reduction_ratio = 1.0 - (total_cands / cartesian_space) if cartesian_space > 0 else 1.0

        metrics = BlockingMetrics(
            block_name=self.name,
            total_s1_evaluated=total_s1,
            total_true_links=total_true,
            captured_true_links=captured_true,
            blocking_recall=captured_true / total_true if total_true > 0 else 0.0,
            s2_true_links=s2_true,
            s2_captured_links=s2_captured,
            s2_recall=s2_captured / s2_true if s2_true > 0 else 0.0,
            s3_true_links=s3_true,
            s3_captured_links=s3_captured,
            s3_recall=s3_captured / s3_true if s3_true > 0 else 0.0,
            total_candidates=total_cands,
            mean_candidates_per_s1=float(np.mean(counts_arr)),
            median_candidates_per_s1=float(np.median(counts_arr)),
            p90_candidates_per_s1=float(np.percentile(counts_arr, 90)),
            p95_candidates_per_s1=float(np.percentile(counts_arr, 95)),
            p99_candidates_per_s1=float(np.percentile(counts_arr, 99)),
            max_candidates_per_s1=int(np.max(counts_arr)),
            reduction_ratio=float(reduction_ratio),
            indexing_time_sec=self.indexing_time_sec,
            query_time_sec=self.query_time_sec,
            total_time_sec=round(self.indexing_time_sec + self.query_time_sec, 3),
        )
        return metrics, candidates
