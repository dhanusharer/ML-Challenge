"""Deterministic ground truth parser.

Converts raw ground truth records into structured entity representations:
{
    "source1_entity_id": str,
    "matched_entity_ids": List[str]
}

Enforces challenge rules:
- Empty matched_entity_ids => empty list []
- Comma-separated IDs => list of stripped strings
- Preserve IDs exactly
- Detect duplicate matched IDs within an entity's list (reject / raise error)
- Ensure matched IDs strictly belong to Source 2 or Source 3 (prefixes S2-, S3-)
- Reject malformed structures and self-matches (S1- in matched_entity_ids)
"""

from pathlib import Path
from typing import Dict, Generator, List, Optional, Set, Tuple, Union
import re

from src.utils.env import DELIMITER, VALID_MATCH_PREFIXES


class GroundTruthParseError(ValueError):
    """Raised when a ground truth record violates formatting or domain rules."""
    pass


def parse_ground_truth_row(
    source1_entity_id: str,
    matched_entity_ids_str: str,
    line_number: Optional[int] = None,
) -> Dict[str, Union[str, List[str]]]:
    """Parse and validate a single ground truth record.

    Args:
        source1_entity_id: Raw Source-1 entity ID.
        matched_entity_ids_str: Raw comma-separated string of matched IDs.
        line_number: Optional line number for clear error reporting.

    Returns:
        Dictionary with keys:
            "source1_entity_id": str,
            "matched_entity_ids": List[str]

    Raises:
        GroundTruthParseError: If any formatting or semantic constraint is violated.
    """
    prefix_info = f"Line {line_number}: " if line_number is not None else ""

    s1_id = source1_entity_id.strip()
    if not s1_id:
        raise GroundTruthParseError(f"{prefix_info}source1_entity_id is empty.")

    if not s1_id.startswith("S1-"):
        raise GroundTruthParseError(
            f"{prefix_info}source1_entity_id '{s1_id}' does not start with expected prefix 'S1-'."
        )

    raw_matches = matched_entity_ids_str.strip()
    if not raw_matches:
        return {
            "source1_entity_id": s1_id,
            "matched_entity_ids": [],
        }

    # Split on comma
    tokens = raw_matches.split(",")
    matched_ids: List[str] = []
    seen: Set[str] = set()

    for token in tokens:
        mid = token.strip()
        if not mid:
            raise GroundTruthParseError(
                f"{prefix_info}Malformed empty ID token in matched_entity_ids: '{raw_matches}'"
            )

        # Disallow self-matches
        if mid.startswith("S1-"):
            raise GroundTruthParseError(
                f"{prefix_info}Forbidden self-match: matched ID '{mid}' has S1- prefix. "
                f"Source-1 entities can only match Source 2 or Source 3 entities."
            )

        # Ensure valid target source prefix
        if not mid.startswith(VALID_MATCH_PREFIXES):
            raise GroundTruthParseError(
                f"{prefix_info}Invalid target prefix for matched ID '{mid}'. "
                f"Expected prefix in {VALID_MATCH_PREFIXES}."
            )

        # Detect duplicate matched IDs within an entity's list
        if mid in seen:
            raise GroundTruthParseError(
                f"{prefix_info}Duplicate matched ID '{mid}' detected for entity '{s1_id}'."
            )

        seen.add(mid)
        matched_ids.append(mid)

    return {
        "source1_entity_id": s1_id,
        "matched_entity_ids": matched_ids,
    }


def parse_ground_truth_file(
    filepath: Union[str, Path],
    max_rows: Optional[int] = None,
) -> Dict[str, List[str]]:
    """Parse a full ground truth TSV file into a dictionary of {s1_id: [matched_ids]}.

    Args:
        filepath: Path to train_ground_truth.tsv.
        max_rows: Optional limit on number of rows to parse.

    Returns:
        Dict mapping source1_entity_id -> list of matched entity IDs.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Ground truth file not found: {path}")

    ground_truth_map: Dict[str, List[str]] = {}

    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\r\n").split(DELIMITER)
        if header != ["source1_entity_id", "matched_entity_ids"]:
            raise GroundTruthParseError(
                f"{path.name}: Expected header ['source1_entity_id', 'matched_entity_ids'], got {header}"
            )

        for line_num, line in enumerate(f, start=2):
            if max_rows and (line_num - 2) >= max_rows:
                break
            parts = line.rstrip("\r\n").split(DELIMITER)
            if len(parts) != 2:
                # Handle empty second column when line ends with tab
                if len(parts) == 1 and line.endswith("\t\n") or line.endswith("\t"):
                    parts = [parts[0], ""]
                else:
                    raise GroundTruthParseError(
                        f"Line {line_num}: Malformed TSV row (expected 2 tab-separated fields, got {len(parts)}): {line!r}"
                    )

            parsed = parse_ground_truth_row(parts[0], parts[1], line_number=line_num)
            s1_id = parsed["source1_entity_id"]
            if s1_id in ground_truth_map:
                raise GroundTruthParseError(
                    f"Line {line_num}: Duplicate source1_entity_id '{s1_id}' in ground truth file."
                )
            ground_truth_map[s1_id] = parsed["matched_entity_ids"]

    return ground_truth_map


def iter_ground_truth_file(
    filepath: Union[str, Path],
) -> Generator[Tuple[str, List[str]], None, None]:
    """Memory-efficient streaming generator yielding (source1_id, matched_ids).

    Yields:
        (source1_entity_id, list_of_matched_ids)
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Ground truth file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\r\n").split(DELIMITER)
        if header != ["source1_entity_id", "matched_entity_ids"]:
            raise GroundTruthParseError(
                f"{path.name}: Expected header ['source1_entity_id', 'matched_entity_ids'], got {header}"
            )

        for line_num, line in enumerate(f, start=2):
            parts = line.rstrip("\r\n").split(DELIMITER)
            if len(parts) == 1:
                parts = [parts[0], ""]
            parsed = parse_ground_truth_row(parts[0], parts[1], line_number=line_num)
            yield parsed["source1_entity_id"], parsed["matched_entity_ids"]
