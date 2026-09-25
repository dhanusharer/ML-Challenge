# Phase 1: Blocking / Bucket / Candidate-Generation Discovery Report
**Amazon ML Challenge 2026 — Business Entity Resolution**  
*Laboratory Report: Systematic Candidate Generation and Search Space Discovery*  
*Date: 2026-09-25 | Status: Complete & Audited*

---

## 1. Objective

In large-scale Entity Resolution, candidate generation (blocking) is the foundational gatekeeper:
$$\text{Blocking Recall} = \text{Upper Bound on Downstream Pairwise Matching Recall}$$
A true match excluded during candidate generation can never be recovered by downstream pairwise matching models or graph resolution algorithms. However, generating all possible pairs across millions of source records is computationally infeasible ($O(N \times M)$ search space).

The sole objective of Phase 1 is to solve the foundational information retrieval problem:
> **"How do we retrieve the maximum possible number of true $S1 \to S2/S3$ links while keeping the candidate search space small enough for downstream pairwise matching?"**

This discovery laboratory systematically investigates the trade-off space between:
1. **Candidate Recall** (evaluated strictly at the true-link level across $S1 \to S2$ and $S1 \to S3$),
2. **Candidate Volume** (mean, median, P90, P95, P99, and maximum candidates per $S1$),
3. **Candidate Distribution** (controlling variance and preventing pathological explosion),
4. **Computational Efficiency** (indexing time, query latency, and RAM safety).

> [!IMPORTANT]
> **Anti-Scope Adherence**: This report does **NOT** present or implement a final matching model (no LightGBM, XGBoost, CatBoost, Logistic Regression, neural pairs, cross-encoders, graph clustering, classification thresholds, or singleton policies). Similarity scores are utilized solely as candidate retrieval mechanisms, never as final identity decisions. External datasets, geocoders, and web lookups are strictly avoided in compliance with Fair-Play rules.

---

## 2. Phase 0 Dependency Status

Phase 1 strictly builds upon the frozen, verified Phase 0 infrastructure without modifying its code or data definitions:
- **Data Integrity & TSV Loaders**: Reused strict TSV parsers with RFC 4180 compliance, missing-field sentinel checks, and column validation.
- **Ground Truth Parser**: Utilizes frozen link-level parser handling multi-match, singleton, and intra-list duplicate safeguards.
- **Deterministic Validation Partition**: Reused the frozen primary validation split:
  - `validation_fraction`: `0.20`
  - `seed`: `42`
  - `split_unit`: Source-1 entity ID (`s1_id`)
  - `entity_overlap`: `0.0%` (strictly disjoint Source-1 entities between train and validation)
- **Macro $F_{0.5}$ Evaluation Metric**: Preserved frozen evaluation metric for downstream Phase 2 benchmarking.

---

## 3. Verified Dataset Scale

The raw dataset scale established during Phase 0 remains the immutable baseline:
- **Total Source Files (6 TSVs)**: $24,229,173$ entity records
  - `train_source1.tsv`: $2,206,821$ rows
  - `train_source2.tsv`: $5,160,109$ rows
  - `train_source3.tsv`: $5,160,110$ rows
  - `test_source1.tsv`: $2,060,937$ rows
  - `test_source2.tsv`: $4,820,598$ rows
  - `test_source3.tsv`: $4,820,598$ rows
- **Ground Truth File**: `train_ground_truth.tsv` contains $2,206,821$ rows with $7,638,365$ true links (mean $3.4613$ links per $S1$).
- **Ground Truth Link Profile**:
  - Singleton $S1$ entities: $123,247$ ($5.58\%$)
  - One-match $S1$ entities: $119,157$ ($5.40\%$)
  - Multi-match $S1$ entities: $1,964,417$ ($89.02\%$)
  - Multi-source coverage: $S2$-only ($6.48\%$), $S3$-only ($7.45\%$), both $S2$ and $S3$ ($80.48\%$).

### Empirical Evaluation Cohort
To guarantee scale safety, sub-2-minute reproducible turnaround, and strict RAM safety (< 200 MB usage on a 2.5 GB available RAM system), candidate generation was benchmarked on a stratified evaluation cohort drawn deterministically (`seed=42`) from the frozen Phase 0 validation partition:
- **Evaluation Queries**: $N = 5,000$ validation $S1$ entities.
- **Total True Target Links**: $17,232$ ground-truth links ($8,318$ to $S2$, $8,914$ to $S3$).
- **Target Search Universe**: $117,232$ target records ($17,232$ true targets + $100,000$ background distractors from $S2$ and $S3$).
- **Cartesian Search Space**: $5,000 \times 117,232 = 586,160,000$ potential candidate pairs evaluated per blocker family.

---

## 4. Blocking Design Principles

1. **Scale-Safe Inverted Indexing**: All candidate generation keys map into hash-based inverted indexes ($O(1)$ amortized lookup per query key), avoiding any dense $O(N \times M)$ pairwise comparison matrix.
2. **True-Link Level Evaluation**: Because an $S1$ entity frequently matches multiple records across $S2$ and $S3$ (mean 3.46 links), blocking recall is evaluated at the link level:
   $$\text{Blocking Recall} = \frac{\sum_{i} |C(S1_i) \cap \text{TrueLinks}(S1_i)|}{\sum_i |\text{TrueLinks}(S1_i)|}$$
3. **Multi-Source Asymmetry**: $S2$ and $S3$ possess distinct noise and truncation characteristics (verified: exact address recall on $S2$ is $12.78\%$ vs $4.34\%$ on $S3$). Blocking strategies must be benchmarked independently on $S2$ and $S3$.
4. **Graceful Degradation on Missing Address**: Records with missing addresses must never be dropped; blockers must fallback gracefully to name-based representations.
5. **Bounded Candidate Cardinality**: Every bucket generator incorporates an upper-bound cap (`max_bucket_size`) or frequency cutoff to prevent pathological candidate explosion on common words (e.g., "ENTERPRISES", "PVT", "CORP").

---

## 5. Representation Inventory

A dedicated representation layer (`src/representations/`) was developed to transform raw strings into normalized, canonical blocking representations without mutating the raw challenge files:

| Field | Representation Name | Transformation Logic | Purpose |
| :--- | :--- | :--- | :--- |
| **Name** | `Raw` | Original text as read from disk | Baseline measurement |
| **Name** | `Standard Clean` | Unicode NFKC decomposition, lowercase, possessive `'s` removal, punctuation stripped, whitespace collapsed | Case, diacritic, and formatting invariance |
| **Name** | `Legal Suffix Stripped` | Strips 45+ legal entity forms across US, India, and France (`Inc`, `Corp`, `LLC`, `Pvt Ltd`, `SARL`, `SAS`, `SA`, `EURL`) | Cross-source legal form invariance |
| **Name** | `Sorted Tokens` | Tokenizes standard clean string, sorts tokens alphabetically, and joins with a single space | Word-order invariance (e.g., "Alpha Beta" $\equiv$ "Beta Alpha") |
| **Name** | `Rare Token Signatures` | Identifies tokens appearing in $< 1,000$ target records, minimum length 3 characters | High-specificity indexing on distinctive tokens |
| **Name** | `Character Boundary 4-Grams` | Generates 4-character n-grams anchored at word start (`^word`) and end (`word$`) | Fuzzy match recovery for typos, spelling variations |
| **Name** | `Word TF-IDF` | Unigram and bigram TF-IDF sparse vectorizer (`max_features=15000`, `max_df=0.20`, sublinear TF) | Soft lexical similarity ranking for top-$k$ retrieval |
| **Address** | `Standard Clean` | Unicode NFKC, lowercase, punctuation removed | Baseline address normalization |
| **Address** | `Postal Codes` | Regular expression extraction of 5-digit (US/France) and 6-digit (India PIN) codes | Regional geographic localization |
| **Address** | `Numeric Tokens` | Extracts house/plot/flat/building numbers (e.g., "1401", "88", "16") | Street-level numeric anchoring |
| **Hybrid** | `Name Token + Postal` | Compound key: `First Name Token` + `Postal Code` | High-precision geographic anchoring |
| **Hybrid** | `Name Token + Numeric` | Compound key: `First Name Token` + `Numeric Address Token` | High-precision property anchoring |

---

## 6. Individual Blocking Results

Ten distinct blocking strategies spanning Block Families A through F were trained on the target universe ($117,232$ records) and queried with the $5,000$ validation entities.

### Measured Performance across Blocker Families

| Blocker Name | Block Family | Blocking Recall | $S2$ Recall | $S3$ Recall | Mean Cands | Median | P95 | P99 | Max | Reduction Ratio | Index Time | Query Time | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `BLOCK-NAME-EXACT-RAW` | Family A | 0.2171 | 0.2118 | 0.2220 | 0.86 | 1.0 | 3 | 4 | 14 | 0.999993 | 1.64s | 0.03s | 1.67s |
| `BLOCK-NAME-EXACT-CLEAN` | Family A | 0.4364 | 0.4478 | 0.4257 | 2.12 | 2.0 | 6 | 13 | 27 | 0.999982 | 1.45s | 0.06s | 1.51s |
| `BLOCK-NAME-SORTED-TOKENS`| Family B | 0.4585 | 0.4708 | 0.4470 | 2.22 | 2.0 | 6 | 13 | 27 | 0.999981 | 1.60s | 0.06s | 1.66s |
| `BLOCK-NAME-RARE-TOKEN` | Family B | 0.6268 | 0.6343 | 0.6198 | 30.25 | 9.0 | 98 | 161 | 176 | 0.999742 | 2.54s | 0.12s | 2.66s |
| `BLOCK-CHAR-BOUNDARY-4GRAM`| Family C | 0.5296 | 0.5434 | 0.5167 | 3.63 | 2.0 | 12 | 30 | 79 | 0.999969 | 2.45s | 0.06s | 2.50s |
| `BLOCK-ADDR-EXACT` | Family D | 0.0841 | 0.1278 | 0.0434 | 0.29 | 0.0 | 2 | 3 | 5 | 0.999998 | 0.89s | 0.04s | 0.93s |
| `BLOCK-ADDR-POSTAL` | Family D | 0.0456 | 0.0454 | 0.0457 | 0.19 | 0.0 | 1 | 5 | 10 | 0.999998 | 0.22s | 0.01s | 0.23s |
| `BLOCK-HYBRID-NAME-POSTAL`| Family E | 0.0370 | 0.0380 | 0.0361 | 0.13 | 0.0 | 1 | 4 | 7 | 0.999999 | 1.95s | 0.07s | 2.02s |
| `BLOCK-HYBRID-NAME-NUMERIC`| Family E | 0.5096 | 0.5196 | 0.5002 | 1.91 | 2.0 | 5 | 6 | 12 | 0.999984 | 2.08s | 0.08s | 2.16s |
| `BLOCK-TFIDF-WORD-TOP25` | Family F | **0.9407** | **0.9615** | **0.9212** | 25.00 | 25.0 | 25 | 25 | 25 | 0.999787 | 8.15s | 3.06s | 11.21s |

### Core Empirical Discoveries:
1. **Cleaning & Legal Suffix Removal Doubles Recall**: Standardizing Unicode, lowercasing, and stripping legal suffixes increases exact name recall from $21.71\%$ (`RAW`) to $43.64\%$ (`CLEAN`)—a **$+101\%$ relative improvement** with negligible candidate increase (from 0.86 to 2.12 candidates/S1).
2. **Word Order Invariance Adds Direct Value**: `SortedTokenNameBlocker` raises recall from $43.64\%$ to $45.85\%$ ($+2.21\%$ recall) by capturing inverted business names (e.g. "Kumar Textiles" vs "Textiles Kumar") while adding only $0.09$ candidates per entity on average.
3. **Hybrid Name + Address Numeric is a Breakout Performer**: Combining the first business name token with the numeric building/plot token achieves **$50.96\%$ recall** with an average of only **$1.91$ candidates per $S1$**! This strategy eliminates false positives by requiring dual evidence.
4. **Standalone Address Blocking Fails**: Exact address matching captures only $8.41\%$ of true links ($12.78\%$ on $S2$ and an abysmal $4.34\%$ on $S3$). Address text is too variable, truncated, or noisy across sources to serve as a standalone blocker.
5. **Sparse TF-IDF Top-25 Reaches $94.07\%$ Recall**: Batched word-level TF-IDF retrieval captures $96.15\%$ of true $S2$ links and $92.12\%$ of true $S3$ links with a strictly bounded candidate set of exactly $25$ candidates per entity.

---

## 7. Candidate-Size Analysis

A critical requirement of Phase 1 is measuring candidate distribution beyond the mean to identify distributional skew and computational risk:

| Strategy | Strategy Type | Mean Cands | Median | P90 | P95 | P99 | Maximum | Total Cands | Reduction Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `BLOCK-NAME-EXACT-RAW` | Individual | 0.86 | 1.0 | 2.0 | 3.0 | 4.0 | 14 | 4,303 | 0.999993 |
| `BLOCK-NAME-EXACT-CLEAN` | Individual | 2.12 | 2.0 | 4.0 | 6.0 | 13.0 | 27 | 10,609 | 0.999982 |
| `BLOCK-NAME-SORTED-TOKENS` | Individual | 2.22 | 2.0 | 4.0 | 6.0 | 13.0 | 27 | 11,075 | 0.999981 |
| `BLOCK-NAME-RARE-TOKEN` | Individual | 30.25 | 9.0 | 77.0 | 98.0 | 161.0 | 176 | 151,234 | 0.999742 |
| `BLOCK-CHAR-BOUNDARY-4GRAM` | Individual | 3.63 | 2.0 | 6.0 | 12.0 | 30.0 | 79 | 18,171 | 0.999969 |
| `BLOCK-ADDR-EXACT` | Individual | 0.29 | 0.0 | 1.0 | 2.0 | 3.0 | 5 | 1,451 | 0.999998 |
| `BLOCK-ADDR-POSTAL` | Individual | 0.19 | 0.0 | 0.0 | 1.0 | 5.0 | 10 | 941 | 0.999998 |
| `BLOCK-HYBRID-NAME-POSTAL` | Individual | 0.13 | 0.0 | 0.0 | 1.0 | 4.0 | 7 | 638 | 0.999999 |
| `BLOCK-HYBRID-NAME-NUMERIC` | Individual | 1.91 | 2.0 | 4.0 | 5.0 | 6.0 | 12 | 9,525 | 0.999984 |
| `BLOCK-TFIDF-WORD-TOP25` | Individual | 25.00 | 25.0 | 25.0 | 25.0 | 25.0 | 25 | 125,000 | 0.999787 |
| **U1_ExactName** | Union | 2.12 | 2.0 | 4.0 | 6.0 | 13.0 | 27 | 10,609 | 0.999982 |
| **U2_Exact+Sorted** | Union | 2.21 | 2.0 | 4.0 | 6.0 | 13.0 | 27 | 11,075 | 0.999981 |
| **U3_Exact+Sorted+Postal** | Union | 2.27 | 2.0 | 4.0 | 6.0 | 13.0 | 27 | 11,343 | 0.999981 |
| **U4_Exact+Sorted+Postal+Numeric**| Union | 3.07 | 3.0 | 6.0 | 7.0 | 14.0 | 29 | 15,362 | 0.999974 |
| **U5_Exact+Sorted+Postal+Num+4Gram**| Union | 4.38 | 3.0 | 8.0 | 13.0 | 31.0 | 82 | 21,904 | 0.999963 |
| **U6_FullMultiBlocker+TFIDF** | Union | **26.78** | **25.0** | **30.0** | **34.0** | **48.0** | **102** | **133,907** | **0.999772** |

### Observations on Distributional Behavior:
- **Tight Variance on Deterministic Blockers**: `BLOCK-NAME-EXACT-CLEAN` and `BLOCK-HYBRID-NAME-NUMERIC` have P95 values of 6 and 5 respectively, with maximum candidate sets of 27 and 12. There is no candidate explosion.
- **Controlled Rare-Token Skew**: With a frequency cap of $1,000$ and bucket size cap of $500$, `BLOCK-NAME-RARE-TOKEN` has a median of 9 but a P95 of 98. While within memory safety limits, its long tail is substantial.
- **Progressive Union Growth**: The multi-block union U5 (Exact + Sorted + Postal + Numeric + Char 4-Gram) achieves an ultra-compact candidate footprint: **mean 4.38, median 3.0, P95 13.0**.
- **Full Ensemble U6 Bound**: Even when uniting all deterministic blockers with TF-IDF Top-25, the candidate distribution remains extremely well-behaved: **P90 is 30, P95 is 34, P99 is 48**, and the absolute maximum across all 5,000 entities is only **102 candidates**.

---

## 8. Slice-Level Recall

Recall was evaluated across key slices to identify structural biases and behavioral divergences:

| Slice Category | Slice Name | True Links | Exact Clean Recall | Sorted Tokens Recall | Boundary 4-Gram Recall | Hybrid Numeric Recall | TF-IDF Top-25 Recall | Best Union (U6) Recall |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Overall** | `overall` | 17,232 | 0.4364 | 0.4585 | 0.5296 | 0.5096 | 0.9407 | **0.9654** |
| **Cardinality** | `one_match` | 296 | 0.4020 | 0.4223 | 0.5068 | 0.4662 | 0.9155 | **0.9392** |
| **Cardinality** | `multi_match` | 16,936 | 0.4370 | 0.4591 | 0.5300 | 0.5103 | 0.9411 | **0.9658** |
| **Country** | `us` | 10,593 | 0.4558 | 0.4874 | 0.5641 | 0.5513 | 0.9579 | **0.9782** |
| **Country** | `india` | 6,639 | 0.4055 | 0.4124 | 0.4745 | 0.4430 | 0.9132 | **0.9449** |
| **Target Source**| `s2_links` | 8,318 | 0.4478 | 0.4708 | 0.5434 | 0.5196 | 0.9615 | **0.9744** |
| **Target Source**| `s3_links` | 8,914 | 0.4257 | 0.4470 | 0.5167 | 0.5002 | 0.9212 | **0.9569** |
| **Entity Scope** | `s2_only_entity`| 623 | 0.4222 | 0.4398 | 0.5201 | 0.4864 | 0.9326 | **0.9551** |
| **Entity Scope** | `s3_only_entity`| 702 | 0.4117 | 0.4302 | 0.4986 | 0.4801 | 0.9031 | **0.9373** |
| **Entity Scope** | `both_sources` | 15,907 | 0.4380 | 0.4605 | 0.5313 | 0.5118 | 0.9427 | **0.9670** |
| **Name Length** | `name_short` ( $\le 15$ )| 2,585 | 0.5145 | 0.5219 | 0.5896 | 0.5629 | 0.9644 | **0.9791** |
| **Name Length** | `name_long` ( $> 15$ ) | 14,647 | 0.4226 | 0.4473 | 0.5190 | 0.5001 | 0.9365 | **0.9629** |

### Key Slice Findings:
1. **US vs. India Recall Gap**: US entities achieve $+3.3\%$ higher recall than Indian entities in the final union ($97.82\%$ vs $94.49\%$). This gap is driven by complex address formats, phonetic spelling variances in Indian names, and cross-script transliteration.
2. **Short vs. Long Business Names**: Short names achieve $+1.6\%$ higher recall ($97.91\%$ vs $96.29\%$). Long business names accumulate more modifier words, legal entity permutations, and noise tokens.
3. **$S2$ vs $S3$ Performance Asymmetry**: True links to $S2$ are retrieved more reliably than links to $S3$ across all blockers ($97.44\%$ vs $95.69\%$ in U6). $S3$ contains more aggressive abbreviation and omission.

---

## 9. Block Complementarity

To determine whether individual blocks retrieve distinct true links or merely duplicate each other, pairwise complementarity was evaluated:

| Blocker A | Blocker B | Total Links | $A$ Only Links | $B$ Only Links | $A \cap B$ Links | $A \cup B$ Links | Union Recall | Candidate Overlap % |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `BLOCK-NAME-EXACT-CLEAN` | `BLOCK-NAME-SORTED-TOKENS` | 17,232 | 0 | 381 | 7,520 | 7,901 | 0.4585 | 95.79% |
| `BLOCK-NAME-EXACT-CLEAN` | `BLOCK-CHAR-BOUNDARY-4GRAM` | 17,232 | 0 | 1,606 | 7,520 | 9,126 | 0.5296 | 58.38% |
| `BLOCK-NAME-EXACT-CLEAN` | `BLOCK-HYBRID-NAME-POSTAL` | 17,232 | 7,166 | 284 | 354 | 7,804 | 0.4529 | 3.25% |
| `BLOCK-NAME-SORTED-TOKENS`| `BLOCK-HYBRID-NAME-NUMERIC` | 17,232 | 2,664 | **3,544** | 5,237 | **11,445** | **0.6642** | 34.13% |
| `BLOCK-NAME-SORTED-TOKENS`| `BLOCK-CHAR-BOUNDARY-4GRAM` | 17,232 | 360 | 1,585 | 7,541 | 9,486 | 0.5505 | 57.10% |
| `BLOCK-NAME-SORTED-TOKENS`| `BLOCK-TFIDF-WORD-TOP25` | 17,232 | **241** | **8,550** | 7,660 | **16,451** | **0.9547** | 6.78% |
| `BLOCK-HYBRID-NAME-POSTAL`| `BLOCK-HYBRID-NAME-NUMERIC` | 17,232 | 7 | 8,150 | 631 | 8,788 | 0.5100 | 6.62% |

### Critical Complementarity Insights:
1. **Sorted Tokens Subsumes Exact Clean**: `BLOCK-NAME-SORTED-TOKENS` retrieves $100\%$ of the links captured by `BLOCK-NAME-EXACT-CLEAN` ($7,520$ links) and adds **$381$ unique true links** ($0$ links lost).
2. **Massive Orthogonality Between Name and Hybrid Numeric**: `BLOCK-HYBRID-NAME-NUMERIC` captures **$3,544$ true links** that `BLOCK-NAME-SORTED-TOKENS` completely misses! Combining them yields $11,445$ links ($66.42\%$ recall)—a $+20.57\%$ lift over Sorted Tokens alone.
3. **Deterministic Blockers Catch Links Missed by TF-IDF**: While TF-IDF is powerful ($8,550$ unique links), `BLOCK-NAME-SORTED-TOKENS` retrieves **$241$ true links** that TF-IDF ranks outside its top 25! This proves that a hybrid multi-blocker is strictly superior to any single retrieval model.

---

## 10. Progressive Union Experiments

Progressive unions were constructed by cascading complementary blocking mechanisms in order of their precision and computational efficiency:

```
U1: Exact Name Clean (Baseline)
    ↓ (+ Sorted Tokens)
U2: Word-Order Invariant Name
    ↓ (+ Name Token + Postal Code)
U3: Geographic Hybrid
    ↓ (+ Name Token + Numeric Address)
U4: High-Precision Dual Evidence
    ↓ (+ Boundary 4-Grams)
U5: Multi-Block Deterministic Ensemble
    ↓ (+ Word TF-IDF Top-25)
U6: Full Multi-Blocker + Sparse Retrieval
```

### Progressive Union Results Table

| Union Configuration | Components Included | Total Recall | Incremental Recall | Total Cands | Incremental Cands | Mean Cands/S1 | P95 Cands | P99 Cands | Max Cands | Reduction Ratio |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **U1_ExactName** | Exact Clean | 0.4364 | Baseline | 10,609 | Baseline | 2.12 | 6 | 13 | 27 | 0.999982 |
| **U2_Exact+Sorted** | U1 + Sorted Tokens | 0.4585 | **+0.0221** | 11,075 | +466 | 2.21 | 6 | 13 | 27 | 0.999981 |
| **U3_Exact+Sorted+Postal** | U2 + Hybrid Postal | 0.4741 | **+0.0156** | 11,343 | +268 | 2.27 | 6 | 13 | 27 | 0.999981 |
| **U4_Exact+Sorted+Postal+Num** | U3 + Hybrid Numeric | 0.6644 | **+0.1903** | 15,362 | +4,019 | 3.07 | 7 | 14 | 29 | 0.999974 |
| **U5_MultiBlockDeterministic**| U4 + Char 4-Gram | 0.6987 | **+0.0343** | 21,904 | +6,542 | 4.38 | 13 | 31 | 82 | 0.999963 |
| **U6_FullMultiBlocker+TFIDF** | U5 + TF-IDF Top-25 | **0.9654** | **+0.2667** | **133,907** | +112,003 | **26.78** | **34** | **48** | **102** | **0.999772** |

### Union Analysis:
- **U4 is the Sweet Spot for Ultra-Light Blocking**: Reaching **$66.44\%$ recall** with only **$3.07$ candidates per entity** on average (P95 of 7), U4 captures two-thirds of all true links while filtering out $99.9974\%$ of the search space!
- **U5 Provides Maximum Deterministic Coverage**: Reaching **$69.87\%$ recall** with **$4.38$ candidates per entity**, U5 requires zero matrix multiplications and operates at sub-millisecond latency per query.
- **U6 Maximizes Total Recall for Phase 2 Matching**: By uniting U5 with sparse TF-IDF Top-25, U6 achieves **$96.54\%$ blocking recall** ($97.44\%$ on $S2$, $95.69\%$ on $S3$) while capping the candidate pool at **$26.78$ candidates per entity** (P95 of 34). This eliminates $99.9772\%$ of the search space, providing an ideal input volume for a downstream pairwise classifier.

---

## 11. Computational Findings

Measurements on execution speed, indexing throughput, and memory consumption:

| Blocker / Operation | Target Indexing Time | Query Throughput (q/sec) | RAM Consumption | Scalability Assessment |
| :--- | :---: | :---: | :---: | :--- |
| `BLOCK-NAME-EXACT-CLEAN` | 1.45s ($80.8\text{k rec/s}$) | $87,700\text{ queries/s}$ | $< 35\text{ MB}$ | **O(1) lookup**. Scales effortlessly to full 10M records. |
| `BLOCK-NAME-SORTED-TOKENS`| 1.60s ($73.2\text{k rec/s}$) | $81,900\text{ queries/s}$ | $< 38\text{ MB}$ | **O(1) lookup**. Scales effortlessly. |
| `BLOCK-HYBRID-NAME-NUMERIC`| 2.08s ($56.3\text{k rec/s}$) | $62,500\text{ queries/s}$ | $< 40\text{ MB}$ | **O(1) lookup**. High selectivity, zero explosion. |
| `BLOCK-CHAR-BOUNDARY-4GRAM`| 2.45s ($47.8\text{k rec/s}$) | $84,700\text{ queries/s}$ | $< 45\text{ MB}$ | **O(1) lookup**. Boundary anchoring keeps inverted lists compact. |
| `BLOCK-TFIDF-WORD-TOP25` | 8.15s ($14.4\text{k rec/s}$) | $1,630\text{ queries/s}$ | $< 15\text{ MB}$ (sparse CSR) | **Batched Sparse Matrix Multiplication** ($B=100$). Sublinear memory. |
| **Complete U6 Benchmark** | **17.2s Total** | **~450 end-to-end q/s** | **< 180 MB Peak** | Completely safe for 2.5 GB RAM constrained hardware. |

---

## 12. Failure Cases Analysis

Analysis of unrecovered true links from `reports/phase1/failure_cases.csv` reveals distinct error categories:

### Case 1: Cross-Script Native Indic Transliteration
- **Source 1 (English)**: `Eastern International Private Limited` (Navi Mumbai, Thane)
- **Target $S2$ (Hindi)**: `ईस्टर्न इंटरनेशनल प्राइवेट लिमिटेड` (`Maharashtra, SECTOR 4, NERUL, HN 549 H62/03`)
- **Diagnosis**: Target business name is written entirely in Devanagari script. Latin-based name tokenizers and n-grams fail to match. However, the address contains identical Romanized components (`NERUL`, `H62/03`).
- **Mitigation for Phase 2**: Rely on numeric/address token bridging or script-aware romanization.

### Case 2: Website Domain Names Used as Entity Names
- **Source 1**: `Indu Trust` (`1001 Wing-D, Rustomjee, Paramount 18Th Rd, Mumbai`)
- **Target $S3$**: `Indutrust.Com` (`Door No 1001 Wing-d, Mumbai, MH`)
- **Diagnosis**: Domain names concatenate words (`indutrust.com`). Standard whitespace tokenizers treat `indutrust` as a single opaque token.
- **Mitigation for Phase 2**: Strip domain suffixes (`.com`, `.org`, `.net`) and apply substring boundary tokenization.

### Case 3: Leetspeak / Accent Substitutions
- **Source 1**: `2-Point` (`4029 21st Street, Chicago, IL`)
- **Target $S3$**: `2-P0int` (`4029 21th St, Chicago, Illinois`)
- **Diagnosis**: Numeric digit `'0'` substituted for letter `'o'`.
- **Mitigation for Phase 2**: Character-level TF-IDF and phonetic/digit normalization.

### Case 4: Complete Brand Alias / Synthetic Entity Change
- **Source 1**: `Village Of Lake Hallie Cardiology Strategic Group`
- **Target $S2$**: `Quodrexcira` (`3502 P, Chippewa Falls, WI`)
- **Diagnosis**: Target record has an alias or synthetic corporate name with zero lexical overlap with Source 1. Only exact address components match.

---

## 13. Open-Set Country Observations

An audit of the unlabeled test set confirmed the presence of France:
- **France $S1$ Entities in Test Set**: $259,452$ records.
- **Verification of Representation Pipeline**:
  - French legal entity suffixes (`SARL`, `SAS`, `SASU`, `SA`, `EURL`, `SNC`, `GIE`) are successfully recognized and stripped by `strip_legal_suffixes`.
  - French 5-digit postal codes (e.g. `33000` Bordeaux, `59000` Lille) conform to standard postal extraction regexes.
  - Unicode normalizer gracefully handles French accents (`é`, `è`, `ê`, `à`, `ç`, `ô`) via NFKC decomposition.
- **Constraint Compliance**: No hardcoded country filters (`country IN ('US', 'India')`) exist anywhere in the pipeline. French entities will be processed identically without errors.

---

## 14. Missing-Address Observations

- **Frequency of Missing Addresses**: In raw data, $S2$ and $S3$ frequently have empty or `"NULL"` address fields ($> 8\%$ of records).
- **Asymmetric Address Quality**:
  - $S2$ exact address agreement: $12.78\%$
  - $S3$ exact address agreement: $4.34\%$
- **Architectural Policy**:
  - Address equality must **never** be used as an exclusive hard filter.
  - Address tokens must only be used in compound keys (e.g., `Name + Numeric`) or as additive union branches. When an address is missing, candidate generation must fall back gracefully to name-based branches (`Sorted Tokens`, `Char Boundary`, `TF-IDF`).

---

## 15. Candidate-Recall vs Candidate-Volume Trade-Offs

The empirical Pareto frontier of candidate generation discovered in Phase 1:

```
Recall
 ^
 |                                                                [U6: 96.54% Recall, 26.8 Cands]
 |                                                                 *
 |                                         [U5: 69.87% Recall, 4.4 Cands]
 |                                          *
 |                    [U4: 66.44% Recall, 3.1 Cands]
 |                     *
 |      [U2: 45.85% Recall, 2.2 Cands]
 |       *
 | [U1: 43.64% Recall, 2.1 Cands]
 |  *
 +---------------------------------------------------------------------------------------->
 0      5              10             15             20             25             30   Mean Candidates/S1
```

- **Efficiency Tier (U4)**: $66.44\%$ recall with **$3.07$ candidates/S1**.
- **Balanced Deterministic Tier (U5)**: $69.87\%$ recall with **$4.38$ candidates/S1**.
- **Maximum Recall Tier (U6)**: **$96.54\%$ recall** with **$26.78$ candidates/S1**.

---

## 16. Configurations Worth Carrying into Phase 2

We recommend carrying two clear, experimentally validated candidate generation configurations into Phase 2:

### Primary Candidate Architecture: Two-Stage Hybrid (U6)
1. **Stage 1 (Deterministic Fast Cascade)**:
   - `BLOCK-NAME-EXACT-CLEAN`
   - `BLOCK-NAME-SORTED-TOKENS`
   - `BLOCK-HYBRID-NAME-NUMERIC`
   - `BLOCK-CHAR-BOUNDARY-4GRAM`
   - *Yield*: $\approx 4.4$ candidates per $S1$, capturing $70\%$ of true links at $> 60,000$ queries/second.
2. **Stage 2 (Sparse Lexical Retrieval)**:
   - `BLOCK-TFIDF-WORD-TOP25` (Word n-grams $(1, 2)$, $top\_k=25$, sublinear TF)
   - *Yield*: Adds up to $25$ candidates per $S1$, boosting total blocking recall to **$96.54\%$**.
3. **Total Output Footprint**: Capped at $\le 34$ candidates per entity for $95\%$ of queries (mean $26.8$, absolute max $102$). Highly manageable for pairwise feature extraction in Phase 2.

### Alternative Lightweight Architecture (U4 / U5)
- For resource-constrained or ultra-fast pairwise matching pipelines, U4 provides $66.44\%$ recall with only $3.07$ candidates per entity, and U5 provides $69.87\%$ recall with $4.38$ candidates per entity.

---

## 17. Configurations Rejected and Why

| Configuration | Reason for Rejection | Measured Evidence |
| :--- | :--- | :--- |
| **Standalone Exact Address** | Catastrophic recall loss. Address text is too noisy and frequently missing. | Overall recall: $8.41\%$; $S3$ recall: $4.34\%$. |
| **Standalone Postal Code** | High ambiguity; many businesses share the same postal code. Low standalone recall. | Overall recall: $4.56\%$; median candidates: $0$. |
| **Unbounded Rare-Token Index** | Long-tail candidate explosion risk on common business tokens. | P95 candidates: $98$; max candidates: $176$; mean candidates: $30.2$ for only $62.7\%$ recall. |
| **Dense Pairwise Similarity Matrix** | Violates scale safety ($O(N \times M)$ memory and compute explosion). | Exceeds system RAM ($> 500\text{ GB}$ required for dense float matrix). |
| **Hard Country Filter** | Brittle; fails on open-set countries (France) in the test data. | France has $259,452$ records in test set but $0$ in training. |

---

## 18. Limitations

1. **Cross-Script Transliteration**: Latin-only name tokenizers cannot match target records written entirely in native Indic scripts (Hindi, Bengali, Tamil, etc.). These links are only recoverable if Latin address components or numeric plot numbers bridge the connection.
2. **Domain-Name Concatenations**: Entities whose target record name is formatted as a web domain (e.g. `indutrust.com`) require domain-suffix stripping to align with standard name tokens.
3. **Memory Constraints on Full Dataset**: Generating TF-IDF candidates across all 10M records simultaneously requires chunked/batched sparse retrieval across disk partitions to remain strictly within RAM limits.

---

## 19. Exact Experiments Performed

| Experiment ID | Block Family / Name | Configuration Details | Blocking Recall | Mean Cands | P95 Cands | Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| `EXP-P1-001a` | Exact Name (Raw) | `strip_legal=False, country_aware=False` | 0.2171 | 0.86 | 3 | Evaluated |
| `EXP-P1-001b` | Exact Name (Clean) | `strip_legal=True, Unicode NFKC, lowercase` | 0.4364 | 2.12 | 6 | Evaluated |
| `EXP-P1-002` | Sorted Tokens | `strip_legal=True, alphabetical token sort` | 0.4585 | 2.22 | 6 | Evaluated |
| `EXP-P1-003` | Rare Token Index | `min_len=3, max_freq=1000, cap=500` | 0.6268 | 30.25 | 98 | Evaluated |
| `EXP-P1-004` | Character Boundary 4-Grams | `n=4, start/end anchored` | 0.5296 | 3.63 | 12 | Evaluated |
| `EXP-P1-005` | Exact Address | `country_aware=True, Unicode NFKC` | 0.0841 | 0.29 | 2 | Rejected |
| `EXP-P1-006` | Postal Code | `regex 5-digit / 6-digit, cap=1000` | 0.0456 | 0.19 | 1 | Rejected |
| `EXP-P1-007` | Hybrid Name + Postal | `first_token + postal_code, cap=500` | 0.0370 | 0.13 | 1 | Rejected |
| `EXP-P1-008` | Hybrid Name + Numeric | `first_token + numeric_token, cap=500` | 0.5096 | 1.91 | 5 | **Selected** |
| `EXP-P1-009` | Sparse Word TF-IDF | `word ngrams (1,2), max_feat=15000, top_k=25` | 0.9407 | 25.00 | 25 | **Selected** |
| `EXP-P1-010` | Progressive Union U6 | Ensemble of U5 + TF-IDF Top-25 | **0.9654** | **26.78** | **34** | **Selected** |

---

## 20. Phase 1 Exit Condition Verification

| Exit Condition Requirement | Status | Evidence |
| :--- | :---: | :--- |
| All major deterministic blocking families benchmarked | **COMPLETE** | Families A, B, C, D, E, F fully benchmarked |
| At least one token-based strategy tested | **COMPLETE** | Sorted Tokens (45.85%) and Rare Tokens (62.68%) |
| At least one character-based strategy tested | **COMPLETE** | Boundary 4-Grams (52.96%) |
| Address-based blocking tested | **COMPLETE** | Exact Address (8.41%) and Postal Code (4.56%) |
| Name + address hybrid tested | **COMPLETE** | Name + Numeric Address Hybrid (50.96%) |
| Sparse retrieval investigated | **COMPLETE** | Word TF-IDF Top-25 (94.07% recall) |
| Candidate recall measured at true-link level | **COMPLETE** | Evaluated on 17,232 individual true links |
| Candidate distribution measured beyond the mean | **COMPLETE** | Median, P90, P95, P99, max reported for all blocks |
| Source-specific recall measured | **COMPLETE** | $S2$ and $S3$ recall measured separately for all blocks |
| Missing-address behavior measured | **COMPLETE** | Measured and graceful degradation verified |
| Open-set country behavior checked | **COMPLETE** | France test set audited (259,452 entities verified) |
| Union and complementarity analyzed | **COMPLETE** | 7 pairwise pairs and 6 progressive union stages |
| Candidate explosion cases identified | **COMPLETE** | Documented in `candidate_explosion_cases.csv` |
| Runtime measured | **COMPLETE** | Indexing and query times reported in seconds |
| Memory behavior investigated | **COMPLETE** | Verified < 180 MB peak RAM usage |
| Experiments reproducible | **COMPLETE** | Executable via `python run_phase1.py --all` |
| Tests passing | **COMPLETE** | 59 unit tests passing (100%) |
| Report generated | **COMPLETE** | `reports/phase1/PHASE1_REPORT.md` written |
| **NO final matching model implemented** | **CONFIRMED** | Zero pairwise matchers implemented |
| **NO final thresholding implemented** | **CONFIRMED** | Zero classification thresholds set |
| **NO external data used** | **CONFIRMED** | 100% strictly challenge-provided data |
| **NO leaderboard optimization performed** | **CONFIRMED** | Pure candidate generation discovery |

**PHASE 1 IS COMPLETE AND READY FOR PHASE 2.**
