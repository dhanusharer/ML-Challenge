"""Phase 2 Matcher Evaluation Engine (charter §25, §27, §28, §29, §30, §31).

Evaluates:
A. Candidate-conditioned pairwise performance:
   - Pair Precision & Recall
   - PR-AUC (Average Precision)
   - ROC-AUC
   - Recall@K (K in {1, 3, 5, 10})
B. End-to-end performance:
   - True link recall
   - Diagnostic Macro F0.5 via official MacroF05Evaluator across threshold sweeps
   - Multi-match retention analysis
   - Singleton score analysis & false merge tracking
   - Rank-margin diagnostics (top score, second score, score margin)
"""

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score

from src.evaluation.metric import evaluate_f05, EvaluationResult


class MatcherEvaluator:
    """Computes candidate-conditioned and end-to-end entity resolution metrics."""

    def __init__(self, ground_truth: Dict[str, List[str]]):
        self.ground_truth = ground_truth

    def evaluate_candidate_conditioned(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        threshold: float = 0.5,
    ) -> Dict[str, float]:
        """Compute metrics exclusively among candidates that reached the matcher."""
        if len(y_true) == 0:
            return {
                "pair_precision": 0.0,
                "pair_recall": 0.0,
                "pr_auc": 0.0,
                "roc_auc": 0.0,
            }

        y_pred = (y_scores >= threshold).astype(np.int32)
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)

        pr_auc = average_precision_score(y_true, y_scores) if np.sum(y_true) > 0 else 0.0
        try:
            roc = roc_auc_score(y_true, y_scores) if (len(np.unique(y_true)) > 1) else 0.5
        except Exception:
            roc = 0.5

        return {
            "pair_precision": float(p),
            "pair_recall": float(r),
            "pr_auc": float(pr_auc),
            "roc_auc": float(roc),
        }

    def evaluate_ranking_quality(
        self,
        query_candidates_scored: List[Dict[str, Any]],
        k_list: Sequence[int] = (1, 3, 5, 10),
    ) -> Dict[str, float]:
        """Compute Recall@K and Precision@K across query entities."""
        recalls_at_k: Dict[int, List[float]] = {k: [] for k in k_list}
        precisions_at_k: Dict[int, List[float]] = {k: [] for k in k_list}

        for q_item in query_candidates_scored:
            gt_targets = q_item["ground_truth_targets"]
            if not gt_targets:
                continue

            cands = q_item["candidates"]  # List of {'target_record': dict, 'score': float, 'label': int}
            if not cands:
                for k in k_list:
                    recalls_at_k[k].append(0.0)
                    precisions_at_k[k].append(0.0)
                continue

            # Sort by score descending
            sorted_cands = sorted(cands, key=lambda c: c["score"], reverse=True)

            for k in k_list:
                top_k = sorted_cands[:k]
                top_k_ids = {c["target_record"]["entity_id"] for c in top_k}
                hits = len(top_k_ids.intersection(gt_targets))
                r_k = hits / len(gt_targets) if len(gt_targets) > 0 else 0.0
                p_k = hits / len(top_k) if len(top_k) > 0 else 0.0
                recalls_at_k[k].append(r_k)
                precisions_at_k[k].append(p_k)

        out = {}
        for k in k_list:
            out[f"recall_at_{k}"] = float(np.mean(recalls_at_k[k])) if recalls_at_k[k] else 0.0
            out[f"precision_at_{k}"] = float(np.mean(precisions_at_k[k])) if precisions_at_k[k] else 0.0
        return out

    def evaluate_end_to_end_sweep(
        self,
        query_candidates_scored: List[Dict[str, Any]],
        thresholds: Sequence[float] = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8),
    ) -> List[Dict[str, Any]]:
        """Run diagnostic threshold sweep computing official Macro F0.5 and link recall."""
        results = []

        all_q_ids = [item["query_record"]["entity_id"] for item in query_candidates_scored]
        total_true_links = sum(len(item["ground_truth_targets"]) for item in query_candidates_scored)

        for th in thresholds:
            predictions: Dict[str, List[str]] = {}
            total_pred_links = 0
            captured_links = 0

            for item in query_candidates_scored:
                s1_id = item["query_record"]["entity_id"]
                gt_set = item["ground_truth_targets"]
                matched_tids = []

                for cand in item["candidates"]:
                    if cand["score"] >= th:
                        tid = cand["target_record"]["entity_id"]
                        matched_tids.append(tid)
                        if tid in gt_set:
                            captured_links += 1

                predictions[s1_id] = matched_tids
                total_pred_links += len(matched_tids)

            # Evaluate with official Phase 0 evaluate_f05
            subset_gt = {sid: self.ground_truth.get(sid, []) for sid in all_q_ids}
            eval_res = evaluate_f05(ground_truth=subset_gt, predictions=predictions, strict_entity_coverage=True)

            results.append({
                "threshold": float(th),
                "macro_f05": eval_res.macro_f05,
                "mean_precision": eval_res.mean_precision,
                "mean_recall": eval_res.mean_recall,
                "end_to_end_link_recall": captured_links / total_true_links if total_true_links > 0 else 0.0,
                "total_predicted_links": total_pred_links,
                "total_true_links": total_true_links,
                "num_true_singletons": eval_res.num_true_singletons,
                "num_correct_singletons": eval_res.num_correct_singletons,
                "num_false_singleton_merges": eval_res.num_false_singleton_merges,
            })

        return results

    def compute_rank_margin_diagnostics(
        self,
        query_candidates_scored: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Compute diagnostic top score, second score, and score margin."""
        margins = []
        top_scores = []
        second_scores = []

        for item in query_candidates_scored:
            cands = item["candidates"]
            if not cands:
                continue
            scores = sorted([c["score"] for c in cands], reverse=True)
            top_s = scores[0]
            sec_s = scores[1] if len(scores) > 1 else 0.0
            top_scores.append(top_s)
            second_scores.append(sec_s)
            margins.append(top_s - sec_s)

        return {
            "mean_top_score": float(np.mean(top_scores)) if top_scores else 0.0,
            "mean_second_score": float(np.mean(second_scores)) if second_scores else 0.0,
            "mean_score_margin": float(np.mean(margins)) if margins else 0.0,
        }
