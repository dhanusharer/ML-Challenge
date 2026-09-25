"""Authoritative TSV data loaders with strict schema and integrity validation.

Enforces the Amazon ML Challenge 2026 data contract:
- Tab-separated files (.tsv)
- Required columns:
    Source files: entity_id, business_name, business_address, country
    Ground truth: source1_entity_id, matched_entity_ids
- Delimiter verification (detect accidental CSV or single-column loading)
- Source prefix consistency (S1-, S2-, S3-)
- Entity ID uniqueness and non-nullity
"""

from pathlib import Path
from typing import Generator, List, Optional, Union
import os
import pandas as pd

from src.utils.env import (
    DELIMITER,
    EXPECTED_SOURCE_COLUMNS,
    EXPECTED_GROUND_TRUTH_COLUMNS,
    SOURCE_PREFIXES,
)


class TSVSchemaValidationError(ValueError):
    """Raised when a TSV file violates the authoritative schema or data contract."""
    pass


def validate_tsv_header(
    filepath: Union[str, Path],
    expected_columns: List[str],
) -> List[str]:
    """Inspect the first line of a TSV file to verify delimiter and column schema.

    Args:
        filepath: Path to the TSV file.
        expected_columns: Exact list of column names expected.

    Returns:
        List of column names parsed from header.

    Raises:
        FileNotFoundError: If the file does not exist.
        TSVSchemaValidationError: If header is missing, malformed, or has incorrect delimiter.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        first_line = f.readline()

    if not first_line:
        raise TSVSchemaValidationError(f"File is empty: {path}")

    first_line_clean = first_line.rstrip("\r\n")

    # Detect accidental CSV loading (comma without tab)
    if DELIMITER not in first_line_clean and "," in first_line_clean:
        raise TSVSchemaValidationError(
            f"{path.name}: Header contains commas but NO tab delimiter. "
            f"The file appears to be comma-separated. Amazon ML Challenge requires tab-separated (.tsv)."
        )

    columns = [col.strip() for col in first_line_clean.split(DELIMITER)]

    # Detect accidental single-column loading
    if len(columns) == 1 and len(expected_columns) > 1:
        raise TSVSchemaValidationError(
            f"{path.name}: Loaded as a single column '{columns[0]}'. "
            f"Expected {len(expected_columns)} tab-separated columns: {expected_columns}."
        )

    if columns != expected_columns:
        raise TSVSchemaValidationError(
            f"{path.name}: Schema mismatch. Found columns {columns}, expected exactly {expected_columns}."
        )

    return columns


def load_source_tsv(
    filepath: Union[str, Path],
    expected_prefix: Optional[str] = None,
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """Load a Source TSV file into a pandas DataFrame with full schema verification.

    Args:
        filepath: Path to *_source1.tsv, *_source2.tsv, or *_source3.tsv.
        expected_prefix: Expected ID prefix (e.g. 'S1-', 'S2-', 'S3-').
        nrows: Optional row limit for fast sampling or tests.

    Returns:
        pd.DataFrame containing the validated data with string types.

    Raises:
        TSVSchemaValidationError: If any schema rule or ID constraint is violated.
    """
    path = Path(filepath)
    validate_tsv_header(path, EXPECTED_SOURCE_COLUMNS)

    df = pd.read_csv(
        path,
        sep=DELIMITER,
        dtype=str,
        keep_default_na=False,
        nrows=nrows,
        encoding="utf-8",
    )

    # Verification: check column names
    if list(df.columns) != EXPECTED_SOURCE_COLUMNS:
        raise TSVSchemaValidationError(
            f"{path.name}: DataFrame columns {list(df.columns)} do not match {EXPECTED_SOURCE_COLUMNS}"
        )

    # Verification: null or empty entity_id
    null_mask = df["entity_id"].str.strip() == ""
    if null_mask.any():
        num_null = int(null_mask.sum())
        raise TSVSchemaValidationError(
            f"{path.name}: Found {num_null} empty or whitespace-only entity_id entries."
        )

    # Verification: duplicate entity_id
    dup_mask = df["entity_id"].duplicated(keep=False)
    if dup_mask.any():
        sample_dups = df.loc[dup_mask, "entity_id"].head(5).tolist()
        num_dups = int(dup_mask.sum())
        raise TSVSchemaValidationError(
            f"{path.name}: Found {num_dups} duplicate entity_id rows. Samples: {sample_dups}"
        )

    # Verification: prefix consistency
    if expected_prefix:
        bad_prefix_mask = ~df["entity_id"].str.startswith(expected_prefix)
        if bad_prefix_mask.any():
            sample_bad = df.loc[bad_prefix_mask, "entity_id"].head(5).tolist()
            num_bad = int(bad_prefix_mask.sum())
            raise TSVSchemaValidationError(
                f"{path.name}: Found {num_bad} entity IDs not starting with '{expected_prefix}'. "
                f"Samples: {sample_bad}"
            )

    return df


def iter_source_tsv_chunks(
    filepath: Union[str, Path],
    chunksize: int = 100000,
    expected_prefix: Optional[str] = None,
) -> Generator[pd.DataFrame, None, None]:
    """Stream a large source TSV file in memory-efficient chunks.

    Yields:
        Validated pd.DataFrame chunks with string types.
    """
    path = Path(filepath)
    validate_tsv_header(path, EXPECTED_SOURCE_COLUMNS)

    for chunk in pd.read_csv(
        path,
        sep=DELIMITER,
        dtype=str,
        keep_default_na=False,
        chunksize=chunksize,
        encoding="utf-8",
    ):
        # Check null entity_id in chunk
        null_mask = chunk["entity_id"].str.strip() == ""
        if null_mask.any():
            raise TSVSchemaValidationError(
                f"{path.name}: Found empty entity_id in chunk."
            )
        if expected_prefix:
            bad_prefix_mask = ~chunk["entity_id"].str.startswith(expected_prefix)
            if bad_prefix_mask.any():
                sample_bad = chunk.loc[bad_prefix_mask, "entity_id"].head(5).tolist()
                raise TSVSchemaValidationError(
                    f"{path.name}: Found invalid prefix in chunk. Samples: {sample_bad}"
                )
        yield chunk


def load_ground_truth_tsv(
    filepath: Union[str, Path],
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """Load train_ground_truth.tsv into a pandas DataFrame with schema validation.

    Args:
        filepath: Path to train_ground_truth.tsv.
        nrows: Optional row limit.

    Returns:
        pd.DataFrame with columns ['source1_entity_id', 'matched_entity_ids'].
    """
    path = Path(filepath)
    validate_tsv_header(path, EXPECTED_GROUND_TRUTH_COLUMNS)

    df = pd.read_csv(
        path,
        sep=DELIMITER,
        dtype=str,
        keep_default_na=False,
        nrows=nrows,
        encoding="utf-8",
    )

    if list(df.columns) != EXPECTED_GROUND_TRUTH_COLUMNS:
        raise TSVSchemaValidationError(
            f"{path.name}: Columns {list(df.columns)} do not match {EXPECTED_GROUND_TRUTH_COLUMNS}"
        )

    # Validate source1_entity_id non-nullity
    null_s1 = df["source1_entity_id"].str.strip() == ""
    if null_s1.any():
        num_null = int(null_s1.sum())
        raise TSVSchemaValidationError(
            f"{path.name}: Found {num_null} empty source1_entity_id values."
        )

    # Validate source1_entity_id uniqueness
    dup_s1 = df["source1_entity_id"].duplicated(keep=False)
    if dup_s1.any():
        sample_dups = df.loc[dup_s1, "source1_entity_id"].head(5).tolist()
        raise TSVSchemaValidationError(
            f"{path.name}: Duplicate source1_entity_id rows found. Samples: {sample_dups}"
        )

    # Validate S1- prefix
    bad_prefix = ~df["source1_entity_id"].str.startswith("S1-")
    if bad_prefix.any():
        sample_bad = df.loc[bad_prefix, "source1_entity_id"].head(5).tolist()
        raise TSVSchemaValidationError(
            f"{path.name}: source1_entity_id entries missing 'S1-' prefix: {sample_bad}"
        )

    return df
