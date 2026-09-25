"""Submission Registry and Validation Manager for Phase 4."""

import csv
import hashlib
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from src.utils.env import PROJECT_ROOT


class SubmissionRegistryManager:
    """Tracks submission lifecycle, validation gate execution, and leaderboard audit matrix."""

    def __init__(self, registry_dir: Optional[Path] = None):
        self.registry_dir = Path(registry_dir or (PROJECT_ROOT / "reports" / "phase4"))
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.registry_csv = self.registry_dir / "submission_registry.csv"
        self.matrix_csv = self.registry_dir / "leaderboard_matrix.csv"
        self.hypotheses_csv = self.registry_dir / "experiment_hypotheses.csv"

    @staticmethod
    def compute_sha256(file_path: Path) -> str:
        """Compute SHA256 checksum of a file."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    def run_official_validator(
        self,
        matching_path: Path,
        test_dir: Optional[Path] = None,
        check_ids: bool = False,
    ) -> Tuple[bool, str]:
        """Execute student_resource/utils/validate_submission.py."""
        validator_script = PROJECT_ROOT / "student_resource" / "utils" / "validate_submission.py"
        test_dir = test_dir or (PROJECT_ROOT / "student_resource" / "dataset" / "test")

        if not validator_script.is_file():
            return False, f"Validator script not found at {validator_script}"

        cmd = [
            sys.executable,
            str(validator_script),
            "--matching", str(matching_path),
            "--test-dir", str(test_dir),
        ]
        if check_ids:
            cmd.append("--check-ids")

        res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
        is_valid = (res.returncode == 0)
        output_log = f"Return code: {res.returncode}\n--- STDOUT ---\n{res.stdout}\n--- STDERR ---\n{res.stderr}"
        return is_valid, output_log

    def register_submission(
        self,
        submission_entry: Dict[str, Any],
    ) -> None:
        """Append an audited submission entry to submission_registry.csv."""
        registry_fields = [
            "submission_id",
            "timestamp",
            "day",
            "hypothesis_id",
            "candidate_arm",
            "matcher",
            "threshold_floor",
            "multi_threshold",
            "margin",
            "max_k",
            "rescue_trigger",
            "variant_description",
            "local_dev_f05",
            "local_holdout_f05",
            "public_score",
            "score_delta_vs_baseline",
            "score_delta_vs_previous",
            "file_sha256",
            "validator_status",
            "decision",
            "notes",
        ]

        file_exists = self.registry_csv.is_file()
        with open(self.registry_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=registry_fields)
            if not file_exists:
                writer.writeheader()
            row = {k: submission_entry.get(k, "") for k in registry_fields}
            writer.writerow(row)

    def update_leaderboard_matrix(
        self,
        matrix_rows: List[Dict[str, Any]],
    ) -> None:
        """Export current leaderboard matrix sorted by performance."""
        matrix_fields = [
            "submission_id",
            "hypothesis_id",
            "local_dev_f05",
            "local_holdout_f05",
            "public_score",
            "delta_vs_baseline",
            "candidate_pairs_count",
            "predicted_empty_pct",
            "avg_predicted_matches_per_s1",
            "runtime_seconds",
            "file_sha256",
            "status_verdict",
        ]
        with open(self.matrix_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=matrix_fields)
            writer.writeheader()
            for r in matrix_rows:
                writer.writerow({k: r.get(k, "") for k in matrix_fields})

    def export_hypotheses(
        self,
        hypotheses: List[Dict[str, Any]],
    ) -> None:
        """Export structured hypothesis documentation to experiment_hypotheses.csv."""
        hyp_fields = [
            "hypothesis_id",
            "variant_name",
            "rationale",
            "proposed_change",
            "expected_effect",
            "local_validation_evidence",
            "risk_assessment",
        ]
        with open(self.hypotheses_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=hyp_fields)
            writer.writeheader()
            for h in hypotheses:
                writer.writerow({k: h.get(k, "") for k in hyp_fields})
