"""Experiment registry for tracking reproducible ML experiments.

Tracks:
- experiment_id
- timestamp
- git_commit_if_available
- data_version
- validation_seed
- validation_fraction
- code_version
- notes
- metrics (optional)
"""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import subprocess

from src.utils.env import PROJECT_ROOT


def get_git_commit() -> str:
    """Safely obtain current git commit hash, or return 'unavailable'."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if res.returncode == 0:
            commit = res.stdout.strip()
            return commit if commit else "unavailable"
    except Exception:
        pass
    return "unavailable"


@dataclass
class ExperimentRecord:
    """Container for experiment metadata."""
    experiment_id: str
    timestamp: str
    git_commit_if_available: str
    data_version: str
    validation_seed: int
    validation_fraction: float
    code_version: str
    notes: str
    metrics: Optional[Dict[str, Any]] = None
    extra_metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ExperimentRegistry:
    """Manages the lifecycle and serialization of experiments."""

    def __init__(self, registry_dir: Optional[Path] = None):
        self.registry_dir = registry_dir or (PROJECT_ROOT / "experiments")
        self.registry_dir.mkdir(parents=True, exist_ok=True)

    def register(self, record: ExperimentRecord) -> Path:
        """Save experiment record to JSON under experiments/<experiment_id>/metadata.json."""
        exp_folder = self.registry_dir / record.experiment_id
        exp_folder.mkdir(parents=True, exist_ok=True)

        target_file = exp_folder / "metadata.json"
        with open(target_file, "w", encoding="utf-8") as f:
            json.dump(record.to_dict(), f, indent=2, ensure_ascii=False)

        # Also update global index
        index_file = self.registry_dir / "experiments_index.json"
        index: List[Dict[str, Any]] = []
        if index_file.is_file():
            try:
                with open(index_file, "r", encoding="utf-8") as f:
                    index = json.load(f)
            except Exception:
                index = []

        # Avoid duplicate entries in index
        index = [e for e in index if e.get("experiment_id") != record.experiment_id]
        index.append(record.to_dict())

        with open(index_file, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2, ensure_ascii=False)

        return target_file


def register_experiment(
    experiment_id: str,
    notes: str,
    validation_seed: int = 42,
    validation_fraction: float = 0.20,
    code_version: str = "0.1.0",
    data_version: str = "amazon-ml-challenge-2026-v1",
    metrics: Optional[Dict[str, Any]] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> ExperimentRecord:
    """Create and register a new experiment record."""
    timestamp = datetime.now(timezone.utc).isoformat()
    git_commit = get_git_commit()

    record = ExperimentRecord(
        experiment_id=experiment_id,
        timestamp=timestamp,
        git_commit_if_available=git_commit,
        data_version=data_version,
        validation_seed=validation_seed,
        validation_fraction=validation_fraction,
        code_version=code_version,
        notes=notes,
        metrics=metrics,
        extra_metadata=extra_metadata,
    )

    registry = ExperimentRegistry()
    registry.register(record)
    return record
