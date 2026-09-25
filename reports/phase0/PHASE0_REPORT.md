# Phase 0 Report: The Experiment & Evaluation Foundation
**Amazon ML Challenge 2026 — Business Entity Resolution**  
**Experiment Identifier:** `EXP-P0-001`  
**Date:** September 2026  
**Status:** Complete & Verified  

---

## A. Scope of Phase 0

Phase 0 establishes the **Experiment and Evaluation Foundation** for the Amazon ML Challenge 2026 Business Entity Resolution challenge. Its singular goal is to provide a rigorous, mathematically exact, and leak-free foundation to govern all future experiments before any machine learning modeling begins.

### Key Objectives Completed:
1. Immutable ingestion and strict schema validation for all official TSV datasets.
2. Comprehensive, fact-based data integrity audit across train and test partitions.
3. Deterministic parsing of ground truth records enforcing all challenge constraints.
4. Exact implementation of the official Amazon-style Macro $F_{0.5}$ evaluation metric with complete diagnostic tracking.
5. Verification of the evaluation metric against 10 explicit test cases (A through J) and exact reproduction of the numerical example in the challenge specification.
6. Deterministic, entity-level validation splitting (Source 1 disjoint) with cryptographic hash tracking.
7. Verification of zero partition leakage between training and validation data.
8. Creation of an experiment registry (`EXP-P0-001`) tracking data version, code version, and parameters.
9. Delivery of automated tests, a verification runner, and a visualization notebook.

---

## B. Authoritative Challenge Constraints Used

The foundation strictly enforces the authoritative rules provided in the Amazon ML Challenge 2026 documentation (`student_resource/README.md`, `student_resource/utils/validate_submission.py`):

1. **Tab-Separated Format:** All input and output files must be tab-separated (`.tsv`). Comma-separated files are rejected.
2. **Entity ID Prefixes:** 
   - Source 1: `S1-` (Reference deduplicated source)
   - Source 2: `S2-`
   - Source 3: `S3-`
3. **Valid Matching Targets:** Source 1 entities may only match entities with prefixes `S2-` or `S3-`. Self-matches (`S1-` in predictions or ground truth) are prohibited.
4. **Duplicate Prevention:** 
   - No duplicate Source 1 rows are permitted.
   - No duplicate entity IDs are permitted within any matched ID list.
5. **Open-Set Country Distribution:** The training set covers `US` and `India`. The test set introduces a third country, `France`, which does not appear in training. No hardcoded country filters or one-hot vectors restricted to `{US, India}` are permitted.
6. **Macro $F_{0.5}$ Metric:** Precision-heavy evaluation weighting precision 2× over recall:
   $$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$
   Macro-averaged across all Source 1 entities in the evaluation set.
7. **Singleton Semantics:** Source 1 entities with zero true matches score $1.0$ when predicted empty, and $0.0$ when any match is predicted.
8. **Fair-Play Rule:** No external business data lookup, geocoding APIs, web search, or commercial ER APIs are allowed.

*Any parameter not explicitly mandated by Amazon (such as the 80/20 train/val ratio and random seed 42) is designated as: **PROJECT DESIGN CHOICE**.*

---

## C. Dataset Integrity Results & Record Count Reconciliation

A complete, non-modifying audit was executed directly against physical files on disk across all training and test files.

### 1. Independent File Inventory

| Dataset File | File Size (Bytes) | File Size (MB) | Total Lines (incl. Header) | Data Rows (excl. Header) | Unique IDs | Duplicate IDs | Null IDs | Schema Valid | Source Prefix Consistency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `train_source1.tsv` | 210,069,713 | 200.34 MB | 2,206,822 | 2,206,821 | 2,206,821 | 0 | 0 | PASS | 100% `S1-` |
| `train_source2.tsv` | 489,301,488 | 466.63 MB | 5,034,617 | 5,034,616 | 5,034,616 | 0 | 0 | PASS | 100% `S2-` |
| `train_source3.tsv` | 503,705,637 | 480.37 MB | 5,285,604 | 5,285,603 | 5,285,603 | 0 | 0 | PASS | 100% `S3-` |
| `train_ground_truth.tsv` | 127,015,583 | 121.13 MB | 2,206,822 | 2,206,821 | 2,206,821 | 0 | 0 | PASS | Valid `S1-` references |
| `test_source1.tsv` | 175,022,086 | 166.91 MB | 1,732,545 | 1,732,544 | 1,732,544 | 0 | 0 | PASS | 100% `S1-` |
| `test_source2.tsv` | 509,456,422 | 485.86 MB | 4,887,274 | 4,887,273 | 4,887,273 | 0 | 0 | PASS | 100% `S2-` |
| `test_source3.tsv` | 506,002,772 | 482.56 MB | 5,082,317 | 5,082,316 | 5,082,316 | 0 | 0 | PASS | 100% `S3-` |

### 2. Authoritative Dataset Totals

To eliminate ambiguity, two distinct, formally defined totals govern all challenge documentation:

1. **Total of Six Entity Source Files (Excluding Ground Truth)**:
   $$\text{Total}_{\text{6-sources}} = 2,206,821 + 5,034,616 + 5,285,603 + 1,732,544 + 4,887,273 + 5,082,316 = \mathbf{24,229,173} \text{ data rows}$$
   - **Total Lines (including headers):** $24,229,179$ lines
   - **Total Size:** $2,393,558,118$ bytes ($2.229\text{ GiB} \approx 2.394\text{ GB}$)
   - **Train Sources Subtotal:** $12,527,040$ rows
   - **Test Sources Subtotal:** $11,702,133$ rows

2. **Total of All Seven Files (Including Ground Truth)**:
   $$\text{Total}_{\text{7-files}} = 24,229,173 + 2,206,821 = \mathbf{26,435,994} \text{ data rows}$$
   - **Total Lines (including headers):** $26,436,001$ lines
   - **Total Size:** $2,520,573,701$ bytes ($2.347\text{ GiB} \approx 2.521\text{ GB}$)

### 3. Forensic Analysis of the Prior `24,146,873` Figure

A previous summary cited the figure `24,146,873`. The origin and nature of this number have been rigorously audited:

- **Formula that Produced It:**
  $$\sum (\text{train\_s1, train\_s2, train\_s3, test\_s1, test\_s2}) + \mathbf{5,000,016}$$
  $$= 2,206,821 + 5,034,616 + 5,285,603 + 1,732,544 + 4,887,273 + 5,000,016 = \mathbf{24,146,873}$$
- **Exact Deficit:**
  $$24,229,173 - 24,146,873 = \mathbf{82,300} \text{ rows}$$
  $$5,082,316 (\text{actual test\_source3}) - 5,000,016 = 82,300$$
- **Root Cause Classification:**
  1. **Reporting-Definition Issue (YES):** Ground truth ($2,206,821$ rows) was intentionally excluded because it represents label associations rather than raw business entities, but this distinction was not made clear in the top-level label.
  2. **Excluded File (YES):** `train_ground_truth.tsv` was not part of the six-source sum.
  3. **Arithmetic / Transcription Slip (YES):** In `test_source3.tsv`, the row count was measured correctly as $5,082,316$ in `data_audit.json`, but when computing the chat summary sum, the thousands digits `82` were dropped as $5,000,016$, creating an exact $82,300$ row deficit.

**Integrity Findings:**
- Zero malformed TSV lines.
- Zero duplicate entity IDs across all source files.
- Zero null entity IDs.
- Total raw files remain 100% immutable and untouched.

---

## D. Dataset Descriptive Statistics

### 1. Missingness Analysis

| File | Missing `business_name` | Missing `business_address` | Missing `country` |
| :--- | :--- | :--- | :--- |
| `train_source1.tsv` | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| `train_source2.tsv` | 0 (0.00%) | 168,967 (3.36%) | 0 (0.00%) |
| `train_source3.tsv` | 0 (0.00%) | 175,916 (3.33%) | 0 (0.00%) |
| `test_source1.tsv` | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| `test_source2.tsv` | 0 (0.00%) | 129,408 (2.65%) | 0 (0.00%) |
| `test_source3.tsv` | 0 (0.00%) | 136,098 (2.68%) | 0 (0.00%) |

*Key finding:* Source 1 always has complete names and addresses. Sources 2 and 3 contain ~2.6% to 3.4% missing addresses. Name and country are never missing.

### 2. Country Distribution (Open Set)

| Country | `train_source1` | `train_source2` | `train_source3` | `test_source1` | `test_source2` | `test_source3` |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **US** | 1,323,633 (59.98%) | 3,016,817 (59.92%) | 3,170,056 (59.97%) | 663,106 (38.27%) | 1,871,330 (38.29%) | 1,945,701 (38.28%) |
| **India** | 883,188 (40.02%) | 2,017,799 (40.08%) | 2,115,547 (40.03%) | 809,986 (46.75%) | 2,312,565 (47.32%) | 2,405,000 (47.32%) |
| **France** | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 259,452 (14.98%) | 703,378 (14.39%) | 731,615 (14.39%) |

*Key finding:* As dictated by the problem statement, France appears strictly in the test set (~14.4% to 15.0% of records). The training set consists of 60% US and 40% India records.

---

## E. Ground-Truth Statistics

Analysis of all 2,206,821 records in `train_ground_truth.tsv`:

- **Total Source 1 Reference Entities:** 2,206,821
- **Total Ground-Truth True Links:** 7,638,365
- **Mean True Links per Source 1 Entity:** 3.4613
- **True Singletons (0 matches):** 123,247 (5.58%)
- **One-Match Entities:** 119,157 (5.40%)
- **Multi-Match Entities (>1 match):** 1,964,417 (89.02%)
- **Source Breakdown of Matches:**
  - S2 links: 3,693,619 (48.36%)
  - S3 links: 3,944,746 (51.64%)
  - Matching only S2: 143,029 entities (6.48%)
  - Matching only S3: 164,498 entities (7.45%)
  - Matching both S2 and S3: 1,776,047 entities (80.48%)
- **Top Match Count Distribution:**
  - 3 matches: 530,841 (24.05%)
  - 4 matches: 484,115 (21.94%)
  - 2 matches: 375,212 (17.00%)
  - 5 matches: 321,957 (14.59%)
  - 6 matches: 164,868 (7.47%)
  - 0 matches (singletons): 123,247 (5.58%)
  - 1 match: 119,157 (5.40%)
  - 7 matches: 63,968 (2.90%)
  - 8 matches: 18,680 (0.85%)
  - 9 matches: 4,205 (0.19%)

### Raw Exact Agreement Diagnostics (Audit Only)
Evaluated on a random sample of 50,000 true links from `train_ground_truth.tsv`:
- **Exact Name Agreement:** 4.53% (2,265 pairs)
- **Exact Address Agreement:** 2.34% (1,172 pairs)
- **Exact (Name + Address) Agreement:** **0.00%** (0 pairs out of 50,000)

*Crucial takeaway:* Over 95% of true matches exhibit variations, typos, legal suffix changes, or transliterations. A naive exact string match strategy would fail almost completely.

---

## F. Validation Design

### Strategy
Evaluation is macro-averaged across Source 1 entities. Consequently, splitting must be performed strictly at the **Source 1 ENTITY level**, never at the pair level.

### Configuration
- **Validation Fraction:** 0.20 (20% validation, 80% train) — *PROJECT DESIGN CHOICE*
- **Random Seed:** 42 — *PROJECT DESIGN CHOICE*
- **Split Determinism:** IDs are sorted alphabetically before running NumPy's deterministic random permutation.

### Measured Split Counts & Hashes
- **Total Entities:** 2,206,821
- **Train Entities:** 1,765,457 (80.00%)
- **Validation Entities:** 441,364 (20.00%)
- **Train Set SHA256:** `7a83b93a4a935193c851c084d2436adaddaf0196beb6ce3a629fe97b9c7d470f`
- **Validation Set SHA256:** `af242d8b4cf0f30d771e109b499e0a34e0ffb51d216810ee421ae8133c3929ab`

---

## G. $F_{0.5}$ Implementation & Edge Cases

The evaluator (`src/evaluation/metric.py`) implements:
$$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$

### Formal Edge Case Handling:
1. **True Singleton ($|T| = 0$):**
   - If $|P| = 0$: Score = $1.0$ (credit for singleton discovery). Precision = 1.0, Recall = 1.0.
   - If $|P| > 0$: Score = $0.0$ (penalized false merge). Precision = 0.0, Recall = 0.0.
2. **True Non-Singleton ($|T| > 0$):**
   - If $|P| = 0$: Score = $0.0$ (missed entity). Precision = 0.0, Recall = 0.0.
   - If $|P| > 0$:
     - $TP = |T \cap P|$
     - $\text{Precision} = TP / |P|$
     - $\text{Recall} = TP / |T|$
     - If $TP = 0$: Score = $0.0$
     - Else: $F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$
3. **Macro Aggregation:**
   $$\text{Macro } F_{0.5} = \frac{1}{N} \sum_{i=1}^{N} F_{0.5}(e_i)$$
   where $N$ is the total count of Source 1 entities in the evaluation set.

---

## H. $F_{0.5}$ Verification Tests

The test suite (`tests/test_evaluator.py`) verifies all 10 canonical test cases:

| Case | Scenario | True IDs | Predicted IDs | Expected Precision | Expected Recall | Expected $F_{0.5}$ | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **PDF Ex.** | Challenge Problem Statement | `[S2-47, S3-812]` | `[S2-47, S2-193, S3-812]` | $2/3 \approx 0.667$ | $1.000$ | $5/7 \approx 0.714$ | PASS |
| **A** | Perfect One-Match | `[S2-1]` | `[S2-1]` | $1.000$ | $1.000$ | $1.000$ | PASS |
| **B** | Completely Empty | `[S2-1]` | `[]` | $0.000$ | $0.000$ | $0.000$ | PASS |
| **C** | One Missed Match | `[S2-1, S3-1]` | `[S2-1]` | $1.000$ | $0.500$ | $5/6 \approx 0.833$ | PASS |
| **D** | One False Positive | `[S2-1]` | `[S2-1, S3-2]` | $0.500$ | $1.000$ | $5/9 \approx 0.556$ | PASS |
| **E** | Partial Multi-Match | `[S2-1, S2-2, S3-1]` | `[S2-1, S3-99]` | $0.500$ | $1/3 \approx 0.333$ | $5/11 \approx 0.455$ | PASS |
| **F** | Completely Incorrect | `[S2-1, S2-2]` | `[S3-1, S3-2]` | $0.000$ | $0.000$ | $0.000$ | PASS |
| **G** | Correct Singleton | `[]` | `[]` | $1.000$ | $1.000$ | $1.000$ | PASS |
| **H** | False Match on Singleton | `[]` | `[S2-1]` | $0.000$ | $0.000$ | $0.000$ | PASS |
| **I** | Duplicate Predicted ID | `[S2-1]` | `[S2-1, S2-1]` | N/A | N/A | ValueError (Rejected) | PASS |
| **J** | Macro Average (4 entities) | Mixed | Mixed | Mixed | Mixed | Exact Mean = $0.500$ | PASS |

---

## I. Leakage Safeguards

1. **Partition Disjointness:** Automated check `verify_no_leakage()` strictly asserts that `len(set(train_ids) & set(val_ids)) == 0`.
2. **Ground Truth Isolation:** Ground truth mapping is partitioned via `partition_ground_truth()`. Validation ground truth keys and values are sequestered in `val_ground_truth`.
3. **Feature Calculation Protocol:** Future feature encoders (Phase 1+) must fit statistics exclusively on the training partition. No test set or validation set labels may be accessed during feature fitting.

---

## J. Reproducibility Status

The entire Phase 0 verification is 100% reproducible via a single CLI invocation:

```bash
# Run all automated tests
python run_phase0.py --test

# Run full data integrity audit
python run_phase0.py --audit

# Run validation split and experiment registration
python run_phase0.py --split

# Or execute everything sequentially
python run_phase0.py --all
```

All 44 automated tests pass deterministically in under 1 second.

---

## K. Known Limitations

1. **Memory Budgeting:** The combined raw training and test data exceeds 2.4 GB uncompressed (24M+ records). Loading all files into memory at once as uncompressed pandas DataFrames requires 6–8 GB RAM. The streaming and chunked readers in `src/data/loader.py` and `src/data/integrity.py` prevent out-of-memory errors by operating line-by-line or in configurable chunks.
2. **Missing Addresses in Target Sources:** Sources 2 and 3 have ~3% missing addresses. Future candidate generation cannot rely exclusively on address-based blocking keys (such as PIN codes) without a fallback strategy.

---

## L. Manual Actions Required

- **None for Phase 0.** The dataset was successfully discovered and verified at `student_resource/dataset`.
- All automated tests, audits, split generation, and experiment registration completed successfully.

---

## M. Explicit Phase Boundary Statement

> **CONFIRMATION:**  
> **No blocking strategies, candidate generation, similarity models, ML matchers, embeddings, ANN, vector indexes, threshold optimization, external lookups, or leaderboard submissions were implemented in Phase 0.**  
> Phase 0 concludes strictly with the verified evaluation and experiment foundation.
