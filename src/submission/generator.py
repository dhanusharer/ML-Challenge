"""Submission generator and streaming inference engine for Phase 4."""

from collections import defaultdict
import csv
import json
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple
import numpy as np

from src.utils.env import PROJECT_ROOT, resolve_train_dir
from src.representations.normalizer import standard_clean
from src.representations.name import strip_legal_suffixes, get_sorted_tokens_name
from src.representations.domain import clean_domain_name, get_compact_domain_signature
from src.blocking.recovery_lab import get_informative_name_tokens, extract_address_numeric_compounds
from src.features.extractor import PairwiseFeatureExtractor
from src.models.tree_matcher import LightGBMMatcher
from src.matching.pipeline import MatchingPipelineRunner
from src.matching.candidate_loader import CandidateArmLoader
from src.matching.negative_sampler import StratifiedNegativeSampler
from src.policy.decision import BaseDecisionPolicy, RelativeMarginPolicy


class SubmissionGenerator:
    """Manages model training, candidate generation, caching, and TSV emission for test entities."""

    def __init__(
        self,
        test_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        seed: int = 42,
    ):
        self.seed = seed
        self.test_dir = Path(test_dir or (PROJECT_ROOT / "student_resource" / "dataset" / "test"))
        self.output_dir = Path(output_dir or (PROJECT_ROOT / "output" / "leaderboard"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model: Optional[LightGBMMatcher] = None
        self.feature_extractor = PairwiseFeatureExtractor()
        self.cached_scored_candidates: Dict[str, List[Tuple[str, float]]] = {}
        self.test_s1_records: List[Dict[str, str]] = []
        self.test_s1_order: List[str] = []

    def train_production_model(self, sample_size: int = 25000) -> LightGBMMatcher:
        """Train authoritative Phase 3 LightGBM matcher on training partition."""
        print(f"[SubmissionGenerator] Training production LightGBM matcher ({sample_size} train S1 entities)...", flush=True)
        runner = MatchingPipelineRunner(seed=self.seed)
        rng = np.random.RandomState(self.seed)
        shuffled_train = rng.permutation(runner.train_s1_ids)
        selected_train_s1 = list(shuffled_train[:sample_size])

        req_targets = set()
        for sid in selected_train_s1:
            for mid in runner.ground_truth.get(sid, []):
                req_targets.add(mid)

        train_q_recs, dev_target_pool = runner.load_entity_records(
            selected_train_s1, target_ids=req_targets, max_distractors_per_source=50000
        )
        arm_b_loader = CandidateArmLoader(target_records=dev_target_pool, arm="ARM_B")
        sampler = StratifiedNegativeSampler(negatives_per_positive=6, seed=self.seed)

        X_train, y_train, q_groups, meta = runner.prepare_training_pairs(arm_b_loader, train_q_recs, sampler)
        print(f"  Training matrix: {len(y_train):,} pairs ({meta['positives']} pos, {meta['negatives']} neg)", flush=True)

        self.model = LightGBMMatcher(
            max_depth=6,
            num_leaves=31,
            learning_rate=0.08,
            n_estimators=100,
            random_state=self.seed,
        )
        self.model.fit(X_train, y_train)
        print("  Model fit complete.", flush=True)
        return self.model

    def load_test_queries(self, max_queries: Optional[int] = None) -> List[str]:
        """Load ordered test Source-1 entities from test_source1.tsv."""
        s1_file = self.test_dir / "test_source1.tsv"
        print(f"[SubmissionGenerator] Reading test Source-1 queries from {s1_file}...", flush=True)
        self.test_s1_records.clear()
        self.test_s1_order.clear()

        with open(s1_file, "r", encoding="utf-8") as f:
            header = f.readline().rstrip("\r\n").split("\t")
            for i, line in enumerate(f):
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
                self.test_s1_order.append(eid)
                if max_queries is None or i < max_queries:
                    self.test_s1_records.append(rec)

        print(f"  Loaded {len(self.test_s1_order):,} total test S1 IDs ({len(self.test_s1_records):,} evaluated).", flush=True)
        return self.test_s1_order

    def scan_and_score_test_candidates(
        self,
        max_distractor_scan: Optional[int] = None,
    ) -> Dict[str, List[Tuple[str, float]]]:
        """Stream target records against indexed test queries and score with LightGBM."""
        if not self.test_s1_records:
            raise RuntimeError("Test queries not loaded. Call load_test_queries() first.")
        if self.model is None:
            self.train_production_model()

        print(f"[SubmissionGenerator] Building inverted indexes for {len(self.test_s1_records):,} test queries...", flush=True)
        t0 = time.time()
        q_exact = defaultdict(list)
        q_sorted = defaultdict(list)
        q_domain = defaultdict(list)
        q_numeric = defaultdict(list)
        q_map = {r["entity_id"]: r for r in self.test_s1_records}

        for i, q in enumerate(self.test_s1_records):
            raw_name = q.get("business_name", "")
            raw_addr = q.get("business_address", "")
            cn = strip_legal_suffixes(raw_name)
            if cn:
                q_exact[cn].append(i)
            sn = get_sorted_tokens_name(raw_name, strip_legal=True)
            if sn:
                q_sorted[sn].append(i)
            dom_sig = get_compact_domain_signature(raw_name, min_length=4)
            if dom_sig:
                q_domain[dom_sig].append(i)
            inf = get_informative_name_tokens(raw_name)
            nums = extract_address_numeric_compounds(raw_addr)
            if inf and nums:
                for num in nums:
                    q_numeric[f"{inf[0]}::{num}"].append(i)

        print(f"  Indexes built in {time.time()-t0:.2f}s.", flush=True)

        # Candidate accumulation map: q_idx -> dict(target_id -> (target_record, provenance))
        accumulated_candidates: List[Dict[str, Tuple[Dict[str, Any], Dict[str, Any]]]] = [
            {} for _ in range(len(self.test_s1_records))
        ]

        # Scan test_source2.tsv and test_source3.tsv
        t_scan = time.time()
        for filename in ["test_source2.tsv", "test_source3.tsv"]:
            p = self.test_dir / filename
            if not p.is_file():
                continue
            print(f"  Streaming {filename}...", flush=True)
            with open(p, "r", encoding="utf-8") as f:
                next(f)
                for line_idx, line in enumerate(f):
                    if max_distractor_scan and line_idx >= max_distractor_scan:
                        break
                    parts = line.rstrip("\r\n").split("\t")
                    if len(parts) < 4:
                        continue
                    tid = parts[0]
                    raw_name = parts[1]
                    raw_addr = parts[2]
                    country = parts[3]
                    t_rec = {
                        "entity_id": tid,
                        "business_name": raw_name,
                        "business_address": raw_addr,
                        "country": country,
                    }

                    cn = strip_legal_suffixes(raw_name)
                    sn = get_sorted_tokens_name(raw_name, strip_legal=True)
                    dom_sig = get_compact_domain_signature(raw_name, min_length=4)
                    inf = get_informative_name_tokens(raw_name)
                    nums = extract_address_numeric_compounds(raw_addr)

                    hit_q_indices = set()
                    if cn in q_exact:
                        hit_q_indices.update(q_exact[cn])
                    if sn in q_sorted:
                        hit_q_indices.update(q_sorted[sn])
                    if dom_sig in q_domain:
                        hit_q_indices.update(q_domain[dom_sig])
                    if inf and nums:
                        for num in nums:
                            k = f"{inf[0]}::{num}"
                            if k in q_numeric:
                                hit_q_indices.update(q_numeric[k])

                    for q_idx in hit_q_indices:
                        c_dict = accumulated_candidates[q_idx]
                        if tid not in c_dict and len(c_dict) < 40:
                            prov = {
                                "exact_name": (cn in q_exact and q_idx in q_exact[cn]),
                                "sorted_name": (sn in q_sorted and q_idx in q_sorted[sn]),
                                "name_numeric": False,
                                "char_ngram": False,
                                "domain": (dom_sig in q_domain and q_idx in q_domain[dom_sig]),
                                "transliteration": False,
                                "leetspeak": False,
                                "tfidf": False,
                                "tfidf_score": 0.0,
                                "tfidf_rank": 1000,
                                "support_count": 1,
                            }
                            c_dict[tid] = (t_rec, prov)

        print(f"  Scanning complete in {time.time()-t_scan:.2f}s.", flush=True)

        # Score candidates with LightGBM
        t_score = time.time()
        self.cached_scored_candidates.clear()
        total_scored_pairs = 0

        for q_idx, q_rec in enumerate(self.test_s1_records):
            sid = q_rec["entity_id"]
            cand_dict = accumulated_candidates[q_idx]
            if not cand_dict:
                self.cached_scored_candidates[sid] = []
                continue

            pairs_to_extract = [(q_rec, t_rec, prov) for t_rec, prov in cand_dict.values()]
            X_q = self.feature_extractor.extract_matrix(pairs_to_extract)
            scores = self.model.predict_proba(X_q)

            scored_list = []
            for (tid, _), sc in zip(cand_dict.items(), scores):
                scored_list.append((tid, float(sc)))

            # Sort descending by score
            scored_list.sort(key=lambda x: x[1], reverse=True)
            self.cached_scored_candidates[sid] = scored_list
            total_scored_pairs += len(scored_list)

        print(f"  Scoring complete: {total_scored_pairs:,} candidate pairs scored in {time.time()-t_score:.2f}s.", flush=True)
        return self.cached_scored_candidates

    def generate_submission_tsv(
        self,
        policy: BaseDecisionPolicy,
        output_tsv_path: Path,
    ) -> Path:
        """Apply decision policy and emit complete Amazon-compliant matching_results.tsv."""
        output_tsv_path = Path(output_tsv_path)
        output_tsv_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"[SubmissionGenerator] Applying {policy.name} to generate {output_tsv_path.name}...", flush=True)
        t0 = time.time()

        # Build set predictions from cached scores
        predictions_map: Dict[str, List[str]] = {}
        for sid, scored_list in self.cached_scored_candidates.items():
            if not scored_list:
                predictions_map[sid] = []
                continue
            # Format candidate dicts for policy
            cands_for_policy = [{"target_id": tid, "score": sc} for tid, sc in scored_list]
            preds = policy.predict_matches(sid, cands_for_policy)
            # Enforce Rule 5: only S2/S3 IDs allowed, no duplicates
            clean_preds = []
            seen = set()
            for pid in preds:
                if (pid.startswith("S2-") or pid.startswith("S3-")) and pid not in seen:
                    clean_preds.append(pid)
                    seen.add(pid)
            predictions_map[sid] = clean_preds

        # Emit TSV ensuring EVERY test S1 entity has exactly one row
        with open(output_tsv_path, "w", newline="", encoding="utf-8") as f:
            f.write("source1_entity_id\tmatched_entity_ids\n")
            for sid in self.test_s1_order:
                matches = predictions_map.get(sid, [])
                match_str = ",".join(matches)
                f.write(f"{sid}\t{match_str}\n")

        print(f"  Wrote {len(self.test_s1_order):,} rows to {output_tsv_path} in {time.time()-t0:.2f}s.", flush=True)
        return output_tsv_path
