"""Phase 2 End-to-End Pairwise Matching Pipeline (charter §3, §4, §34, §43).

Orchestrates:
1. Partitioning (strict adherence to frozen Phase 0 80/20 Source-1 split)
2. Streaming candidate generation for development queries
3. Stratified hard-negative sampling
4. Feature matrix extraction with feature quality checks (NaN, range, leakage)
5. Model training across model families
6. Streaming evaluation on frozen validation queries under ARM B and ARM C
7. Diagnostic reporting of candidate-conditioned and end-to-end metrics
"""

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import json
import numpy as np
import time
import psutil

from src.utils.env import PROJECT_ROOT, resolve_train_dir
from src.data.parser import parse_ground_truth_file
from src.features.schema import get_feature_names
from src.features.extractor import PairwiseFeatureExtractor
from src.matching.negative_sampler import StratifiedNegativeSampler
from src.matching.candidate_loader import CandidateArmLoader
from src.matching.evaluator import MatcherEvaluator
from src.models.base import BaseMatcher
from src.models.deterministic import DeterministicWeightedMatcher
from src.models.logistic import LogisticRegressionMatcher
from src.models.tree_matcher import LightGBMMatcher
from src.models.ltr_matcher import LightGBMRankerMatcher


class MatchingPipelineRunner:
    """Manages Phase 2 dataset construction, training, ablation, and evaluation."""

    def __init__(
        self,
        seed: int = 42,
    ):
        self.seed = seed
        self.train_dir = resolve_train_dir()
        self.ground_truth = parse_ground_truth_file(self.train_dir / "train_ground_truth.tsv")

        # Load frozen Phase 0 80/20 Source-1 split
        split_summary = PROJECT_ROOT / "reports" / "phase0" / "validation_split_summary.json"
        with open(split_summary, "r", encoding="utf-8") as f:
            meta = json.load(f)
        from src.validation.split import create_entity_validation_split
        s1_all = list(self.ground_truth.keys())
        split = create_entity_validation_split(
            s1_all,
            val_fraction=meta["split_metadata"]["validation_fraction"],
            seed=meta["split_metadata"]["validation_seed"],
        )
        self.train_s1_ids = split.train_s1_ids
        self.val_s1_ids = split.val_s1_ids

        # Strict leakage assertion
        assert len(set(self.train_s1_ids).intersection(set(self.val_s1_ids))) == 0, (
            "CRITICAL: Train and validation S1 entities overlap!"
        )

        self.evaluator = MatcherEvaluator(ground_truth=self.ground_truth)
        self.feature_extractor = PairwiseFeatureExtractor()
        self.feature_names = get_feature_names(enabled_only=True)

    def load_entity_records(
        self,
        s1_ids: Sequence[str],
        target_ids: Optional[Set[str]] = None,
        max_distractors_per_source: int = 250000,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Load specific S1 queries and target records from raw TSVs."""
        s1_id_set = set(s1_ids)
        query_records = []

        s1_file = self.train_dir / "train_source1.tsv"
        with open(s1_file, "r", encoding="utf-8") as f:
            next(f)
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) >= 4 and parts[0] in s1_id_set:
                    query_records.append({
                        "entity_id": parts[0],
                        "business_name": parts[1],
                        "business_address": parts[2],
                        "country": parts[3],
                    })
                    if len(query_records) == len(s1_id_set):
                        break

        target_records_dict: Dict[str, Dict[str, Any]] = {}
        target_ids_set = set(target_ids) if target_ids is not None else None

        for filename in ["train_source2.tsv", "train_source3.tsv"]:
            src_prefix = "S2-" if "source2" in filename else "S3-"
            needed_in_src = {tid for tid in target_ids_set if tid.startswith(src_prefix)} if target_ids_set else None
            found_needed = 0

            p = self.train_dir / filename
            with open(p, "r", encoding="utf-8") as f:
                next(f)
                distractors = 0
                for line in f:
                    parts = line.rstrip("\r\n").split("\t")
                    if len(parts) < 4:
                        continue
                    eid = parts[0]
                    rec = {
                        "entity_id": eid,
                        "business_name": parts[1],
                        "business_address": parts[2],
                        "country": parts[3],
                    }
                    if target_ids_set is not None and eid in target_ids_set:
                        target_records_dict[eid] = rec
                        found_needed += 1
                    elif distractors < max_distractors_per_source:
                        target_records_dict[eid] = rec
                        distractors += 1

                    if needed_in_src is not None:
                        if found_needed >= len(needed_in_src) and distractors >= max_distractors_per_source:
                            break
                    elif distractors >= max_distractors_per_source:
                        break

        return query_records, list(target_records_dict.values())

    def prepare_training_pairs(
        self,
        candidate_arm_loader: CandidateArmLoader,
        train_queries: List[Dict[str, Any]],
        sampler: StratifiedNegativeSampler,
    ) -> Tuple[np.ndarray, np.ndarray, List[int], Dict[str, Any]]:
        """Construct training feature matrix X and labels y using stratified sampling."""
        t0 = time.time()
        res = candidate_arm_loader.generate_candidates_for_queries(train_queries, self.ground_truth)
        cohort = res["structured_cohort"]

        pairs_to_extract = []
        labels = []
        query_groups = []

        total_pos = 0
        total_neg = 0

        for q_item in cohort:
            q_rec = q_item["query_record"]
            s1_id = q_rec["entity_id"]
            gt_set = q_item["ground_truth_targets"]
            all_cands = q_item["candidates"]

            # Positives that survived candidate generation
            pos_cands = [c for c in all_cands if c["target_record"]["entity_id"] in gt_set]
            # Stratified negative sample
            sampled_neg_cands = sampler.sample_query_negatives(s1_id, all_cands, gt_set)

            q_group_size = len(pos_cands) + len(sampled_neg_cands)
            if q_group_size == 0:
                continue

            query_groups.append(q_group_size)

            for c in pos_cands:
                pairs_to_extract.append((q_rec, c["target_record"], c["provenance"]))
                labels.append(1)
                total_pos += 1

            for c in sampled_neg_cands:
                pairs_to_extract.append((q_rec, c["target_record"], c["provenance"]))
                labels.append(0)
                total_neg += 1

        X = self.feature_extractor.extract_matrix(pairs_to_extract)
        y = np.array(labels, dtype=np.int32)

        meta = {
            "total_pairs": len(labels),
            "positives": total_pos,
            "negatives": total_neg,
            "pos_neg_ratio": total_pos / total_neg if total_neg > 0 else 0.0,
            "query_count": len(query_groups),
            "feature_count": X.shape[1],
            "feature_extraction_seconds": time.time() - t0,
        }
        return X, y, query_groups, meta

    def evaluate_model_on_cohort(
        self,
        model: BaseMatcher,
        candidate_arm_loader: CandidateArmLoader,
        val_queries: List[Dict[str, Any]],
        batch_size: int = 500,
    ) -> Dict[str, Any]:
        """Run unbiased evaluation of model on a validation cohort under a specific candidate arm."""
        t0 = time.time()
        process = psutil.Process()
        m0 = process.memory_info().rss / (1024 * 1024)

        # 1. Candidate generation
        t_cand_start = time.time()
        cand_res = candidate_arm_loader.generate_candidates_for_queries(val_queries, self.ground_truth)
        cohort = cand_res["structured_cohort"]
        cand_time = time.time() - t_cand_start

        # 2. Extract features and predict scores for all candidates in batches
        t_score_start = time.time()
        all_y_true = []
        all_y_scores = []
        scored_cohort = []

        for q_item in cohort:
            q_rec = q_item["query_record"]
            gt_set = q_item["ground_truth_targets"]
            cands = q_item["candidates"]

            if not cands:
                scored_cohort.append({
                    "query_record": q_rec,
                    "ground_truth_targets": gt_set,
                    "candidates": [],
                })
                continue

            pairs_to_extract = [(q_rec, c["target_record"], c["provenance"]) for c in cands]
            X_q = self.feature_extractor.extract_matrix(pairs_to_extract)
            scores_q = model.predict_proba(X_q)

            q_scored_cands = []
            for i, c in enumerate(cands):
                tid = c["target_record"]["entity_id"]
                lbl = 1 if tid in gt_set else 0
                s = float(scores_q[i])
                all_y_true.append(lbl)
                all_y_scores.append(s)
                q_scored_cands.append({
                    "target_record": c["target_record"],
                    "provenance": c["provenance"],
                    "score": s,
                    "label": lbl,
                })

            scored_cohort.append({
                "query_record": q_rec,
                "ground_truth_targets": gt_set,
                "candidates": q_scored_cands,
            })

        score_time = time.time() - t_score_start

        # 3. Compute metrics
        y_true_arr = np.array(all_y_true, dtype=np.int32)
        y_scores_arr = np.array(all_y_scores, dtype=np.float32)

        cand_metrics = self.evaluator.evaluate_candidate_conditioned(y_true_arr, y_scores_arr, threshold=0.50)
        rank_metrics = self.evaluator.evaluate_ranking_quality(scored_cohort, k_list=(1, 3, 5, 10))
        e2e_sweeps = self.evaluator.evaluate_end_to_end_sweep(scored_cohort, thresholds=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8))
        margin_metrics = self.evaluator.compute_rank_margin_diagnostics(scored_cohort)

        # Pick best diagnostic threshold for reporting summary
        best_sweep = max(e2e_sweeps, key=lambda s: s["macro_f05"])

        peak_ram = process.memory_info().rss / (1024 * 1024)
        total_time = time.time() - t0

        return {
            "model_name": model.name,
            "candidate_arm": cand_res["arm"],
            "query_count": len(val_queries),
            "target_count": cand_res["target_count"],
            "total_candidate_pairs": len(all_y_true),
            "mean_candidates_per_query": len(all_y_true) / len(val_queries) if val_queries else 0.0,
            "blocking_recall": cand_res["blocking_recall"],
            "blocking_miss_count": cand_res["blocking_miss_count"],
            "candidate_conditioned": cand_metrics,
            "ranking_quality": rank_metrics,
            "best_diagnostic_threshold": best_sweep["threshold"],
            "best_diagnostic_macro_f05": best_sweep["macro_f05"],
            "best_diagnostic_precision": best_sweep["mean_precision"],
            "best_diagnostic_recall": best_sweep["mean_recall"],
            "end_to_end_link_recall": best_sweep["end_to_end_link_recall"],
            "end_to_end_sweeps": e2e_sweeps,
            "rank_margin": margin_metrics,
            "cand_generation_seconds": cand_time,
            "scoring_seconds": score_time,
            "total_evaluation_seconds": total_time,
            "peak_ram_mb": peak_ram,
        }
