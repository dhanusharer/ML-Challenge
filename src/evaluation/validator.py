"""Validation module for competition submission files.

Wraps and mirrors the official challenge validation rules from validate_submission.py.
Checks matching_results.tsv and candidate_pairs.tsv formatting prior to any leaderboard submission.
"""

from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
import os

from src.utils.env import (
    DELIMITER,
    EXPECTED_SUBMISSION_COLUMNS,
    EXPECTED_CANDIDATE_COLUMNS,
    VALID_MATCH_PREFIXES,
)


class SubmissionValidationError(ValueError):
    """Raised when a submission file violates competition formatting rules."""
    pass


def validate_results_tsv(
    path: Union[str, Path],
    expected_header: List[str],
    col_label: str,
    required_s1_ids: Optional[Set[str]] = None,
) -> Tuple[Dict[str, Set[str]], List[str], List[str]]:
    """Validate a results TSV file (matching_results or candidate_pairs).

    Returns:
        (mapping of {s1_id: set_of_ids}, list_of_errors, list_of_warnings)
    """
    file_path = Path(path)
    errors: List[str] = []
    warnings: List[str] = []

    if not file_path.is_file():
        errors.append(f"File not found: {file_path}")
        return {}, errors, warnings

    mapping: Dict[str, Set[str]] = {}
    seen_s1: Set[str] = set()
    dup_rows: Set[str] = set()
    intra_dupes: Set[str] = set()
    self_matches: Set[str] = set()
    wrong_prefix: Set[str] = set()

    with open(file_path, "r", encoding="utf-8") as f:
        header_line = f.readline()
        if not header_line:
            errors.append(f"{file_path.name} is empty.")
            return {}, errors, warnings

        header_clean = header_line.rstrip("\r\n")
        if DELIMITER not in header_clean and "," in header_clean:
            errors.append(
                f"{file_path.name}: Header has no TAB but contains commas. "
                f"Submissions must be TAB-separated (.tsv). Write with df.to_csv(sep='\\t', index=False)."
            )
            return {}, errors, warnings

        cols = [c.strip() for c in header_clean.split(DELIMITER)]
        if cols != expected_header:
            errors.append(
                f"{file_path.name}: Unexpected header {cols}. Expected exactly {expected_header}."
            )
            return {}, errors, warnings

        for line_num, line in enumerate(f, start=2):
            parts = line.rstrip("\r\n").split(DELIMITER)
            if len(parts) == 1:
                parts = [parts[0], ""]
            elif len(parts) != 2:
                errors.append(
                    f"{file_path.name}: Malformed row at line {line_num}: {line!r}"
                )
                continue

            s1 = parts[0].strip()
            if not s1:
                errors.append(f"{file_path.name}: Empty source1_entity_id at line {line_num}.")
                continue

            if s1 in seen_s1:
                dup_rows.add(s1)
            seen_s1.add(s1)

            raw_ids = parts[1].strip()
            ids = [i.strip() for i in raw_ids.split(",") if i.strip()] if raw_ids else []

            if len(ids) != len(set(ids)):
                intra_dupes.add(s1)

            id_set = set(ids)
            mapping[s1] = id_set

            for mid in id_set:
                if mid.startswith("S1-"):
                    self_matches.add(mid)
                elif not mid.startswith(VALID_MATCH_PREFIXES):
                    wrong_prefix.add(mid)

    if dup_rows:
        sample = sorted(list(dup_rows))[:5]
        errors.append(f"{file_path.name}: Duplicate source1_entity_id rows found: {sample}")

    if intra_dupes:
        sample = sorted(list(intra_dupes))[:5]
        errors.append(f"{file_path.name}: Repeated IDs inside a {col_label} list for: {sample}")

    if self_matches:
        sample = sorted(list(self_matches))[:5]
        errors.append(f"{file_path.name}: Self-matches (S1- prefix) found in {col_label}: {sample}")

    if wrong_prefix:
        sample = sorted(list(wrong_prefix))[:5]
        errors.append(f"{file_path.name}: IDs with invalid prefix found in {col_label}: {sample}")

    if required_s1_ids is not None:
        missing = required_s1_ids - seen_s1
        if missing:
            sample = sorted(list(missing))[:5]
            errors.append(
                f"{file_path.name}: Required S1 entities missing ({len(missing)} total, e.g. {sample}). "
                f"Every entity in test_source1.tsv needs a row."
            )

        extra = seen_s1 - required_s1_ids
        if extra:
            sample = sorted(list(extra))[:5]
            errors.append(f"{file_path.name}: Unexpected S1 IDs not in test set: {sample}")

    return mapping, errors, warnings
