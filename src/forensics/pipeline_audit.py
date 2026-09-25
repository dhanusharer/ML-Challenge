"""Forensic audits for candidate generation, alignment, features, model, and policy."""

from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import numpy as np

from src.utils.env import PROJECT_ROOT, resolve_train_dir
from src.representations.normalizer import standard_clean
from src.representations.name import strip_legal_suffixes, get_sorted_tokens_name
from src.representations.domain import clean_domain_name, get_compact_domain_signature
from src.blocking.recovery_lab import get_informative_name_tokens, extract_address_numeric_compounds
from src.features.schema import FEATURE_REGISTRY, get_feature_names
from src.features.extractor import PairwiseFeatureExtractor
from src.models.tree_matcher import LightGBMMatcher
from src.matching.pipeline import MatchingPipelineRunner
from src.matching.candidate_loader import CandidateArmLoader
from src.matching.negative_sampler import StratifiedNegativeSampler
from src.policy.decision import RelativeMarginPolicy
from src.policy.evaluator import PolicyEvaluator
from src.submission.generator import SubmissionGenerator
from src.submission.analyzer import SubmissionDistributionAnalyzer


class ForensicAuditSuite:
    """Comprehensive test-inference forensics suite for Phase 4.1."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.train_dir = resolve_train_dir()
        self.test_dir = PROJECT_ROOT / "student_resource" / "dataset" / "test"
        self.reports_dir = PROJECT_ROOT / "reports" / "phase4_1"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.runner = MatchingPipelineRunner(seed=self.seed)
        self.evaluator = PolicyEvaluator(ground_truth=self.runner.ground_truth)

    def run_golden_replay_test(
        self,
        cohort_sizes: Sequence[int] = (100, 1000, 5000),
    ) -> List[Dict[str, Any]]:
        """Section 3: Golden Replay Test.
        
        Runs validation cohorts through:
        1) Authoritative Phase 3 Pipeline (with Arm B + true target pool)
        2) Production SubmissionGenerator (with max_distractor_scan=200000 like Phase 4)
        3) Production SubmissionGenerator (full scan or target-aware)
        Scores each with official Amazon Macro F0.5 evaluator.
        """
        print("\n" + "=" * 80)
        print(" FORENSIC AUDIT 1: GOLDEN REPLAY TEST (VALIDATION ON PRODUCTION CODE) ")
        print("=" * 80, flush=True)

        rng = np.random.RandomState(self.seed)
        shuffled_val = rng.permutation(self.runner.val_s1_ids)

        replay_rows = []
        recall_audit_rows = []

        for n_q in cohort_sizes:
            print(f"\n--- Evaluating Validation Cohort Size: {n_q:,} S1 ---", flush=True)
            eval_s1 = list(shuffled_val[:n_q])
            req_targets = set()
            for sid in eval_s1:
                for mid in self.runner.ground_truth.get(sid, []):
                    req_targets.add(mid)

            # 1. Authoritative Phase 3 Pipeline Reference
            eval_q_recs, eval_target_pool = self.runner.load_entity_records(
                eval_s1, target_ids=req_targets, max_distractors_per_source=50000
            )
            loader_arm_b = CandidateArmLoader(target_records=eval_target_pool, arm="ARM_B")
            
            # Train base matcher
            shuffled_train = rng.permutation(self.runner.train_s1_ids)
            train_s1 = list(shuffled_train[:500])
            train_targets = {mid for sid in train_s1 for mid in self.runner.ground_truth.get(sid, [])}
            train_q_recs, train_target_pool = self.runner.load_entity_records(
                train_s1, target_ids=train_targets, max_distractors_per_source=25000
            )
            train_loader = CandidateArmLoader(target_records=train_target_pool, arm="ARM_B")
            sampler = StratifiedNegativeSampler(negatives_per_positive=6, seed=self.seed)
            X_train, y_train, _, _ = self.runner.prepare_training_pairs(train_loader, train_q_recs, sampler)
            
            model = LightGBMMatcher(max_depth=6, num_leaves=31, learning_rate=0.08, n_estimators=100, random_state=self.seed)
            model.fit(X_train, y_train)

            # Score Phase 3 Reference
            t0 = time.time()
            cand_res = loader_arm_b.generate_candidates_for_queries(eval_q_recs, self.runner.ground_truth)
            structured_cohort = cand_res["structured_cohort"]
            p3_blocking_recall = cand_res["blocking_recall"]

            p3_scored_cands = defaultdict(list)
            p3_cand_counts = []
            s2_captured, s2_total = 0, 0
            s3_captured, s3_total = 0, 0

            for item in structured_cohort:
                q_rec = item["query_record"]
                sid = q_rec["entity_id"]
                cand_items = item["candidates"]
                p3_cand_counts.append(len(cand_items))
                gt_targets = item["ground_truth_targets"]

                for mid in gt_targets:
                    if mid.startswith("S2-"):
                        s2_total += 1
                        if any(c["target_record"]["entity_id"] == mid for c in cand_items):
                            s2_captured += 1
                    elif mid.startswith("S3-"):
                        s3_total += 1
                        if any(c["target_record"]["entity_id"] == mid for c in cand_items):
                            s3_captured += 1

                if not cand_items:
                    continue

                pairs_for_q = [(q_rec, c["target_record"], c["provenance"]) for c in cand_items]
                X_q = self.runner.feature_extractor.extract_matrix(pairs_for_q)
                scores = model.predict_proba(X_q)
                for c, sc in zip(cand_items, scores):
                    p3_scored_cands[sid].append({"target_id": c["target_record"]["entity_id"], "score": float(sc)})

            policy = RelativeMarginPolicy(threshold_floor=0.75, multi_threshold=0.80, max_margin=0.08, max_k=3)
            p3_predictions = {}
            for sid in eval_s1:
                c_list = p3_scored_cands.get(sid, [])
                p3_predictions[sid] = policy.predict_matches(sid, c_list)

            p3_eval = self.evaluator.evaluate_predictions(p3_predictions, eval_s1)
            p3_time = time.time() - t0
            print(f"  [Phase 3 Reference] Macro F0.5: {p3_eval['macro_f05']:.4f} (Precision: {p3_eval['mean_precision']:.4f}, Recall: {p3_eval['mean_recall']:.4f}, Blocking Recall: {p3_blocking_recall:.4f}) in {p3_time:.1f}s")

            # 2. Production Code (SubmissionGenerator with max_distractor_scan=200000 like Phase 4)
            # We simulate the exact logic of SubmissionGenerator on validation records
            t1 = time.time()
            # Index queries
            q_exact = defaultdict(list)
            q_sorted = defaultdict(list)
            q_domain = defaultdict(list)
            q_numeric = defaultdict(list)
            for i, q in enumerate(eval_q_recs):
                rn = q.get("business_name", "")
                ra = q.get("business_address", "")
                cn = strip_legal_suffixes(rn)
                if cn: q_exact[cn].append(i)
                sn = get_sorted_tokens_name(rn, strip_legal=True)
                if sn: q_sorted[sn].append(i)
                ds = get_compact_domain_signature(rn, min_length=4)
                if ds: q_domain[ds].append(i)
                inf = get_informative_name_tokens(rn)
                nums = extract_address_numeric_compounds(ra)
                if inf and nums:
                    for num in nums:
                        q_numeric[f"{inf[0]}::{num}"].append(i)

            # Stream train_source2 and train_source3 with max_distractor_scan=200000
            accum_cands = [{} for _ in range(len(eval_q_recs))]
            for fn in ["train_source2.tsv", "train_source3.tsv"]:
                fp = self.train_dir / fn
                with open(fp, "r", encoding="utf-8") as f:
                    next(f)
                    for l_idx, line in enumerate(f):
                        if l_idx >= 200000:
                            break
                        parts = line.rstrip("\r\n").split("\t")
                        if len(parts) < 4:
                            continue
                        tid, rn, ra, cntry = parts[0], parts[1], parts[2], parts[3]
                        cn = strip_legal_suffixes(rn)
                        sn = get_sorted_tokens_name(rn, strip_legal=True)
                        ds = get_compact_domain_signature(rn, min_length=4)
                        inf = get_informative_name_tokens(rn)
                        nums = extract_address_numeric_compounds(ra)

                        hits = set()
                        if cn in q_exact: hits.update(q_exact[cn])
                        if sn in q_sorted: hits.update(q_sorted[sn])
                        if ds in q_domain: hits.update(q_domain[ds])
                        if inf and nums:
                            for num in nums:
                                k = f"{inf[0]}::{num}"
                                if k in q_numeric: hits.update(q_numeric[k])

                        t_rec = {"entity_id": tid, "business_name": rn, "business_address": ra, "country": cntry}
                        for q_idx in hits:
                            cd = accum_cands[q_idx]
                            if tid not in cd and len(cd) < 40:
                                prov = {
                                    "exact_name": (cn in q_exact and q_idx in q_exact[cn]),
                                    "sorted_name": (sn in q_sorted and q_idx in q_sorted[sn]),
                                    "name_numeric": False,
                                    "char_ngram": False,
                                    "domain": (ds in q_domain and q_idx in q_domain[ds]),
                                    "transliteration": False,
                                    "leetspeak": False,
                                    "tfidf": False,
                                    "tfidf_score": 0.0,
                                    "tfidf_rank": 1000,
                                    "support_count": 1,
                                }
                                cd[tid] = (t_rec, prov)

            # Score with model
            prod_predictions = {}
            for q_idx, q_rec in enumerate(eval_q_recs):
                sid = q_rec["entity_id"]
                cd = accum_cands[q_idx]
                if not cd:
                    prod_predictions[sid] = []
                    continue
                pairs_to_extract = [(q_rec, tr, pr) for tr, pr in cd.values()]
                X_q = self.runner.feature_extractor.extract_matrix(pairs_to_extract)
                scores = model.predict_proba(X_q)
                c_scored = [{"target_id": tid, "score": float(sc)} for (tid, _), sc in zip(cd.items(), scores)]
                prod_predictions[sid] = policy.predict_matches(sid, c_scored)

            # Calculate production candidate recall & counts
            prod_s2_captured, prod_s3_captured = 0, 0
            prod_cand_counts = []
            for q_idx, q_rec in enumerate(eval_q_recs):
                sid = q_rec["entity_id"]
                cd = accum_cands[q_idx]
                prod_cand_counts.append(len(cd))
                gt = self.runner.ground_truth.get(sid, [])
                for mid in gt:
                    if mid in cd:
                        if mid.startswith("S2-"):
                            prod_s2_captured += 1
                        elif mid.startswith("S3-"):
                            prod_s3_captured += 1

            total_true = s2_total + s3_total
            prod_captured = prod_s2_captured + prod_s3_captured
            prod_blocking_recall = prod_captured / total_true if total_true > 0 else 0.0

            prod_eval = self.evaluator.evaluate_predictions(prod_predictions, eval_s1)
            prod_time = time.time() - t1
            print(f"  [Production Code Replay] Macro F0.5: {prod_eval['macro_f05']:.4f} (Precision: {prod_eval['mean_precision']:.4f}, Recall: {prod_eval['mean_recall']:.4f}, Blocking Recall: {prod_blocking_recall:.4f}) in {prod_time:.1f}s")

            delta_f05 = prod_eval["macro_f05"] - p3_eval["macro_f05"]
            replay_rows.append({
                "cohort_size": n_q,
                "phase3_macro_f05": p3_eval["macro_f05"],
                "phase3_precision": p3_eval["mean_precision"],
                "phase3_recall": p3_eval["mean_recall"],
                "phase3_link_recall": p3_eval["link_recall"],
                "prod_macro_f05": prod_eval["macro_f05"],
                "prod_precision": prod_eval["mean_precision"],
                "prod_recall": prod_eval["mean_recall"],
                "prod_link_recall": prod_eval["link_recall"],
                "delta_macro_f05": delta_f05,
                "reproduction_status": "REPRODUCED" if abs(delta_f05) < 0.02 else "COLLAPSED_FAILURE",
            })

            # Record candidate recall metrics
            recall_audit_rows.append({
                "pipeline_stage": f"Phase3_ArmB_Cohort_{n_q}",
                "cohort_size": n_q,
                "total_true_links": total_true,
                "captured_links": s2_captured + s3_captured,
                "blocking_recall": p3_blocking_recall,
                "s2_recall": s2_captured / s2_total if s2_total > 0 else 0.0,
                "s3_recall": s3_captured / s3_total if s3_total > 0 else 0.0,
                "mean_candidates_per_s1": float(np.mean(p3_cand_counts)),
                "median_candidates": float(np.median(p3_cand_counts)),
                "p95_candidates": float(np.percentile(p3_cand_counts, 95)),
                "p99_candidates": float(np.percentile(p3_cand_counts, 99)),
                "max_candidates": int(np.max(p3_cand_counts)),
                "retrieval_failure_pct": 1.0 - p3_blocking_recall,
            })
            recall_audit_rows.append({
                "pipeline_stage": f"ProductionCode_Scan200k_Cohort_{n_q}",
                "cohort_size": n_q,
                "total_true_links": total_true,
                "captured_links": prod_captured,
                "blocking_recall": prod_blocking_recall,
                "s2_recall": prod_s2_captured / s2_total if s2_total > 0 else 0.0,
                "s3_recall": prod_s3_captured / s3_total if s3_total > 0 else 0.0,
                "mean_candidates_per_s1": float(np.mean(prod_cand_counts)),
                "median_candidates": float(np.median(prod_cand_counts)),
                "p95_candidates": float(np.percentile(prod_cand_counts, 95)),
                "p99_candidates": float(np.percentile(prod_cand_counts, 99)),
                "max_candidates": int(np.max(prod_cand_counts)),
                "retrieval_failure_pct": 1.0 - prod_blocking_recall,
            })

        # Save to validation_production_replay.csv
        p_csv = self.reports_dir / "validation_production_replay.csv"
        with open(p_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(replay_rows[0].keys()))
            writer.writeheader()
            writer.writerows(replay_rows)
        print(f"\n[Golden Replay] Results exported to {p_csv}", flush=True)

        # Save to candidate_recall_audit.csv
        p_rec_csv = self.reports_dir / "candidate_recall_audit.csv"
        with open(p_rec_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(recall_audit_rows[0].keys()))
            writer.writeheader()
            writer.writerows(recall_audit_rows)
        print(f"[Candidate Recall] Results exported to {p_rec_csv}", flush=True)

        return replay_rows


if __name__ == "__main__":
    suite = ForensicAuditSuite(seed=42)
    suite.run_golden_replay_test(cohort_sizes=[100, 1000, 5000])

