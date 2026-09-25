"""Prediction distribution analyzer and sanity checks for Phase 4 submissions."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np


class SubmissionDistributionAnalyzer:
    """Analyzes test submission distributions and detects anomalous distribution shifts."""

    @staticmethod
    def analyze_submission_file(submission_path: Path) -> Dict[str, Any]:
        """Parse a submission TSV and compute entity-level match statistics."""
        submission_path = Path(submission_path)
        if not submission_path.is_file():
            raise FileNotFoundError(f"Submission file not found: {submission_path}")

        total_s1 = 0
        match_counts = []
        s2_count = 0
        s3_count = 0
        invalid_prefix_count = 0

        with open(submission_path, "r", encoding="utf-8") as f:
            header = f.readline().rstrip("\r\n").split("\t")
            if len(header) < 2 or header[0].lower() != "source1_entity_id":
                raise ValueError(f"Malformed header in {submission_path}: {header}")

            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                if not parts or not parts[0].strip():
                    continue

                total_s1 += 1
                matched_str = parts[1].strip() if len(parts) > 1 else ""
                if not matched_str:
                    match_counts.append(0)
                    continue

                mids = [m.strip() for m in matched_str.split(",") if m.strip()]
                match_counts.append(len(mids))

                for mid in mids:
                    if mid.startswith("S2-"):
                        s2_count += 1
                    elif mid.startswith("S3-"):
                        s3_count += 1
                    else:
                        invalid_prefix_count += 1

        counts = np.array(match_counts, dtype=np.int32)
        n = len(counts)
        if n == 0:
            return {"total_s1_entities": 0, "error": "empty_file"}

        empty_cnt = int(np.sum(counts == 0))
        one_cnt = int(np.sum(counts == 1))
        two_cnt = int(np.sum(counts == 2))
        three_cnt = int(np.sum(counts == 3))
        three_plus_cnt = int(np.sum(counts > 3))
        total_matches = int(np.sum(counts))
        total_targets = s2_count + s3_count

        s2_pct = s2_count / total_targets if total_targets > 0 else 0.0
        s3_pct = s3_count / total_targets if total_targets > 0 else 0.0

        # Assess distribution shift relative to validated training distribution
        # Expected: ~5-20% empty, avg matches ~1.2 - 2.8, S2/S3 roughly balanced
        avg_matches = float(np.mean(counts))
        empty_pct = empty_cnt / n

        is_anomalous = False
        notes = []
        if empty_pct > 0.40:
            is_anomalous = True
            notes.append(f"High empty rate ({empty_pct:.1%})")
        elif empty_pct < 0.01:
            is_anomalous = True
            notes.append(f"Extremely low empty rate ({empty_pct:.1%})")

        if avg_matches > 4.5:
            is_anomalous = True
            notes.append(f"Excessive average matches per S1 ({avg_matches:.2f})")
        elif avg_matches < 0.5:
            is_anomalous = True
            notes.append(f"Depressed average matches per S1 ({avg_matches:.2f})")

        verdict = "NORMAL" if not is_anomalous else f"WARNING: {', '.join(notes)}"

        return {
            "total_s1_entities": n,
            "empty_prediction_count": empty_cnt,
            "empty_prediction_pct": empty_pct,
            "one_match_count": one_cnt,
            "one_match_pct": one_cnt / n,
            "two_match_count": two_cnt,
            "two_match_pct": two_cnt / n,
            "three_match_count": three_cnt,
            "three_match_pct": three_cnt / n,
            "three_plus_match_count": three_plus_cnt,
            "three_plus_match_pct": three_plus_cnt / n,
            "total_predicted_matches": total_matches,
            "avg_matches_per_s1": avg_matches,
            "s2_target_count": s2_count,
            "s2_target_pct": s2_pct,
            "s3_target_count": s3_count,
            "s3_target_pct": s3_pct,
            "invalid_prefix_count": invalid_prefix_count,
            "distribution_shift_verdict": verdict,
        }
