"""Policy evaluation, error decomposition, slice analysis, and bootstrap inference."""

from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple
import numpy as np
from src.evaluation.metric import evaluate_f05, evaluate_entity_f05, EvaluationResult


class PolicyEvaluator:
    """Evaluates entity-level decision policies with Amazon Macro F0.5 and slice diagnostics."""

    def __init__(self, ground_truth: Dict[str, List[str]]):
        self.ground_truth = ground_truth

    def evaluate_predictions(
        self,
        predictions: Dict[str, List[str]],
        query_ids: Sequence[str],
    ) -> Dict[str, Any]:
        """Compute exact Amazon Macro F0.5 and detailed entity-level metrics."""
        subset_gt = {sid: self.ground_truth.get(sid, []) for sid in query_ids}
        subset_preds = {sid: predictions.get(sid, []) for sid in query_ids}

        eval_res: EvaluationResult = evaluate_f05(
            ground_truth=subset_gt,
            predictions=subset_preds,
            strict_entity_coverage=True,
        )

        # Entity-level F0.5 scores
        entity_f05_scores: Dict[str, float] = {}
        for sid in query_ids:
            g = subset_gt.get(sid, [])
            p = subset_preds.get(sid, [])
            f05_val, _, _ = evaluate_entity_f05(g, p, check_duplicates=False)
            entity_f05_scores[sid] = f05_val

        # Entity-level breakdown
        singleton_ids = [sid for sid, g in subset_gt.items() if len(g) == 0]
        one_match_ids = [sid for sid, g in subset_gt.items() if len(g) == 1]
        multi_match_ids = [sid for sid, g in subset_gt.items() if len(g) >= 2]

        singleton_gt = {sid: subset_gt[sid] for sid in singleton_ids}
        singleton_preds = {sid: subset_preds[sid] for sid in singleton_ids}
        singleton_res = evaluate_f05(singleton_gt, singleton_preds) if singleton_ids else None

        one_gt = {sid: subset_gt[sid] for sid in one_match_ids}
        one_preds = {sid: subset_preds[sid] for sid in one_match_ids}
        one_res = evaluate_f05(one_gt, one_preds) if one_match_ids else None

        multi_gt = {sid: subset_gt[sid] for sid in multi_match_ids}
        multi_preds = {sid: subset_preds[sid] for sid in multi_match_ids}
        multi_res = evaluate_f05(multi_gt, multi_preds) if multi_match_ids else None

        # Exact link recall
        total_true_links = sum(len(g) for g in subset_gt.values())
        captured_links = 0
        for sid, g in subset_gt.items():
            g_set = set(g)
            p_list = subset_preds.get(sid, [])
            captured_links += len(g_set.intersection(set(p_list)))

        link_recall = captured_links / total_true_links if total_true_links > 0 else 0.0

        # Predicted count distribution
        pred_counts = [len(p) for p in subset_preds.values()]
        pct_0 = float(np.mean([1 if c == 0 else 0 for c in pred_counts]))
        pct_1 = float(np.mean([1 if c == 1 else 0 for c in pred_counts]))
        pct_2 = float(np.mean([1 if c == 2 else 0 for c in pred_counts]))
        pct_3plus = float(np.mean([1 if c >= 3 else 0 for c in pred_counts]))

        return {
            "macro_f05": eval_res.macro_f05,
            "mean_precision": eval_res.mean_precision,
            "mean_recall": eval_res.mean_recall,
            "link_recall": link_recall,
            "total_true_links": total_true_links,
            "captured_true_links": captured_links,
            "singleton_f05": singleton_res.macro_f05 if singleton_res else 1.0,
            "one_match_f05": one_res.macro_f05 if one_res else 0.0,
            "multi_match_f05": multi_res.macro_f05 if multi_res else 0.0,
            "pct_predicted_0": pct_0,
            "pct_predicted_1": pct_1,
            "pct_predicted_2": pct_2,
            "pct_predicted_3plus": pct_3plus,
            "entity_f05_scores": entity_f05_scores,
        }

    def paired_bootstrap_comparison(
        self,
        preds_a: Dict[str, List[str]],
        preds_b: Dict[str, List[str]],
        query_ids: Sequence[str],
        n_bootstraps: int = 1000,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Run paired entity-level bootstrap test comparing Policy A vs Policy B."""
        res_a = self.evaluate_predictions(preds_a, query_ids)
        res_b = self.evaluate_predictions(preds_b, query_ids)

        scores_a = np.array([res_a["entity_f05_scores"].get(sid, 0.0) for sid in query_ids], dtype=np.float32)
        scores_b = np.array([res_b["entity_f05_scores"].get(sid, 0.0) for sid in query_ids], dtype=np.float32)

        diffs = scores_a - scores_b
        mean_diff = float(np.mean(diffs))

        pct_improved = float(np.mean(diffs > 1e-6))
        pct_worsened = float(np.mean(diffs < -1e-6))
        pct_tied = float(np.mean(np.abs(diffs) <= 1e-6))

        rng = np.random.RandomState(seed)
        n = len(query_ids)
        boot_diffs = []
        for _ in range(n_bootstraps):
            idx = rng.randint(0, n, size=n)
            boot_diffs.append(float(np.mean(diffs[idx])))

        ci_lower = float(np.percentile(boot_diffs, 2.5))
        ci_upper = float(np.percentile(boot_diffs, 97.5))

        # Two-sided empirical p-value for mean_diff != 0
        p_val = float(np.mean(np.array(boot_diffs) <= 0.0) if mean_diff > 0 else np.mean(np.array(boot_diffs) >= 0.0)) * 2.0
        p_val = min(1.0, max(0.0, p_val))

        return {
            "macro_f05_a": res_a["macro_f05"],
            "macro_f05_b": res_b["macro_f05"],
            "mean_delta_f05": mean_diff,
            "ci_95_lower": ci_lower,
            "ci_95_upper": ci_upper,
            "pct_improved": pct_improved,
            "pct_worsened": pct_worsened,
            "pct_tied": pct_tied,
            "p_value": p_val,
            "statistically_significant": bool(ci_lower > 0.0 or ci_upper < 0.0),
        }

    def decompose_errors(
        self,
        scored_cohort: List[Dict[str, Any]],
        predictions: Dict[str, List[str]],
        threshold: float = 0.70,
    ) -> Dict[str, Any]:
        """Decompose missed true links into Type 1 (blocking), Type 2 (matcher), Type 3 (policy)."""
        type1_blocking_loss = 0
        type2_matcher_loss = 0
        type3_policy_loss = 0
        total_true_links = 0
        captured_links = 0

        for item in scored_cohort:
            s1_id = item["query_record"]["entity_id"]
            gt_set = item["ground_truth_targets"]
            total_true_links += len(gt_set)

            cands = item.get("candidates", [])
            cand_id_to_score = {}
            for c in cands:
                tid = c.get("target_id") or c.get("target_record", {}).get("entity_id", "")
                cand_id_to_score[tid] = float(c.get("score", 0.0))

            pred_set = set(predictions.get(s1_id, []))

            for true_tid in gt_set:
                if true_tid in pred_set:
                    captured_links += 1
                elif true_tid not in cand_id_to_score:
                    type1_blocking_loss += 1
                elif cand_id_to_score[true_tid] < threshold:
                    type2_matcher_loss += 1
                else:
                    type3_policy_loss += 1

        denom = total_true_links if total_true_links > 0 else 1
        return {
            "total_true_links": total_true_links,
            "captured_true_links": captured_links,
            "type1_blocking_loss_count": type1_blocking_loss,
            "type1_blocking_loss_pct": type1_blocking_loss / denom,
            "type2_matcher_loss_count": type2_matcher_loss,
            "type2_matcher_loss_pct": type2_matcher_loss / denom,
            "type3_policy_loss_count": type3_policy_loss,
            "type3_policy_loss_pct": type3_policy_loss / denom,
        }
