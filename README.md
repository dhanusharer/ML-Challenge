# Amazon ML Challenge 2026 — Business Entity Resolution
## Phase 0: The Experiment & Evaluation Foundation

This repository contains the trustworthy, reproducible foundation for the **Amazon ML Challenge 2026 (Business Entity Resolution)**.

---

### 1. What Phase 0 Does
- **Data Ingestion Contract:** Strict TSV loaders (`src/data/loader.py`) enforcing tab separation, schema headers, entity ID uniqueness, non-nullity, and source prefix integrity (`S1-`, `S2-`, `S3-`).
- **Ground Truth Parser:** Deterministic parser (`src/data/parser.py`) converting ground truth rows into verified entity dictionaries, enforcing zero self-matches and zero intra-list duplicates.
- **Official Amazon Macro $F_{0.5}$ Evaluator:** Exact implementation of the challenge metric (`src/evaluation/metric.py`) weighting precision 2× over recall, with full singleton support (1.0 for true singleton predicted empty, 0.0 for false merge) and comprehensive diagnostics.
- **Entity-Level Validation Split:** Deterministic disjoint split on Source-1 entities (`src/validation/split.py`) ensuring zero entity overlap and zero label leakage between train (80%) and validation (20%).
- **Leakage Safeguards:** Rigorous assertion checks (`src/validation/leakage.py`) confirming partition boundaries.
- **Factual Data Audit:** Streaming integrity analysis (`src/data/integrity.py`) of all 24+ million records across train and test sets without modifying raw files.
- **Experiment Registry:** Structured tracking (`src/experiments/registry.py`) recording experiment `EXP-P0-001`.

---

### 2. What Phase 0 Deliberately Does NOT Do
In strict accordance with the Phase 0 specification, the following are **strictly out of scope**:
- ❌ No blocking strategies or bucket generation
- ❌ No candidate pair generation
- ❌ No TF-IDF, ANN, embeddings, or vector databases
- ❌ No string similarity matchers (Levenshtein, Jaccard)
- ❌ No ML models (Logistic Regression, LightGBM, XGBoost, CatBoost, neural ER)
- ❌ No threshold optimization
- ❌ No test-set pseudo-labeling
- ❌ No external data lookups, geocoding APIs, Google Maps, or web search

---

### 3. Required Dataset Layout
Raw datasets must remain completely immutable. The pipeline automatically locates files at `student_resource/dataset` (or optionally `data/raw` and `data/test`):

```
student_resource/dataset/
├── train/
│   ├── train_source1.tsv           # Deduplicated reference source (S1-)
│   ├── train_source2.tsv           # Source 2 (S2-)
│   ├── train_source3.tsv           # Source 3 (S3-)
│   └── train_ground_truth.tsv      # S1 to S2/S3 true matches
└── test/
    ├── test_source1.tsv            # Test S1 reference entities
    ├── test_source2.tsv            # Test S2 entities (includes France)
    └── test_source3.tsv            # Test S3 entities (includes France)
```

---

### 4. Installation & Environment Setup

Using Python 3.8+ (tested with Python 3.11):

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate       # On Linux/macOS
.venv\Scripts\activate          # On Windows

# Install pinned dependencies
pip install -r requirements.txt
```

---

### 5. How to Run the Data Audit

To execute the streaming, memory-efficient data audit across all raw files:

```bash
python run_phase0.py --audit
```
Or directly:
```bash
python -m src.data.integrity
```
The audit report is saved to `reports/phase0/data_audit.json`.

---

### 6. How to Run Evaluator Tests

To verify the Amazon-style Macro $F_{0.5}$ evaluator against all canonical test cases (A through J) and the challenge numerical example:

```bash
python run_phase0.py --test
```
Or via pytest:
```bash
pytest tests/ -v
```

---

### 7. How the Validation Split is Generated

To generate the deterministic 80/20 Source-1 entity split, partition ground truth, verify zero leakage, and register experiment `EXP-P0-001`:

```bash
python run_phase0.py --split
```
Or directly:
```bash
python -m src.validation.split
```
The split summary and cryptographic checksums are written to `reports/phase0/validation_split_summary.json` and registered under `experiments/EXP-P0-001/metadata.json`.

---

### 8. Generated Reports and Artifacts

- **Phase 0 Audit Report:** [reports/phase0/PHASE0_REPORT.md](file:///c:/Users/DHANUSH%20A%20G/Desktop/ML-Challenge/reports/phase0/PHASE0_REPORT.md)
- **Data Audit JSON:** [reports/phase0/data_audit.json](file:///c:/Users/DHANUSH%20A%20G/Desktop/ML-Challenge/reports/phase0/data_audit.json)
- **Validation Split Summary:** [reports/phase0/validation_split_summary.json](file:///c:/Users/DHANUSH%20A%20G/Desktop/ML-Challenge/reports/phase0/validation_split_summary.json)
- **Experiment Registry:** [experiments/EXP-P0-001/metadata.json](file:///c:/Users/DHANUSH%20A%20G/Desktop/ML-Challenge/experiments/EXP-P0-001/metadata.json)
- **Interactive Exploration Notebook:** [notebooks/01_phase0_data_audit.ipynb](file:///c:/Users/DHANUSH%20A%20G/Desktop/ML-Challenge/notebooks/01_phase0_data_audit.ipynb)

---

### 9. What Remains for Phase 1
- Design and evaluation of blocking / candidate generation strategies (targeting high recall on the validation set).
- Generation and validation of `candidate_pairs.tsv`.
- Construction of feature extraction pipelines (name and address similarity, token overlaps).
- Pipelining models within the 8B parameter / Apache 2.0 / MIT constraints.

---

### 10. Manual Actions Required
- **None.** All dependencies, datasets, scripts, and validation checks are verified and passing.
