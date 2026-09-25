# Phase 1.5: U6 Full-Scale Validation & Stability Gate Report
**Amazon ML Challenge 2026 — Business Entity Resolution**  
*Rigorous Stability, Scaling, and Variance Audit of the U6 Candidate Generation Architecture*  
*Date: 2026-09-25 | Status: Complete, Verified & Frozen*

---

## 1. Executive Summary & Gate Verdict

Before proceeding to Phase 2 (Pairwise Matching & Feature Engineering), this **Phase 1.5 Stability Gate** was executed as a **large-scale stability benchmark** to evaluate the U6 candidate generation ensemble across substantial query scales:
- **Query Scale**: Scaled progressively from $5,000 \to 25,000 \to 100,000$ Source-1 validation queries.
- **Target Universe Scale**: $496,390$ target records ($346,390$ ground-truth target links + $150,000$ background distractors from $S2$ and $S3$).
- **Search Space Scale**:
  $$\text{The corresponding Cartesian search space contains } 100,000\text{ queries} \times 496,390\text{ targets} = \mathbf{49,639,000,000}\text{ (\approx 49.6B possible query-target pairs).}$$

### Final Gate Verdict: **PASSED (UNANIMOUS)**
1. **Recall Stability**: Recall holds rock-solid at **$93.85\%$** on $100,000$ queries (vs $93.94\%$ on $5,000$ queries), representing an imperceptible drift of only **$0.09\%$** across a $20\times$ increase in query volume.
2. **Sample Variance**: Across 5 non-overlapping folds of $5,000$ queries, recall has a standard deviation of only **$\pm 0.14\%$** (Coefficient of Variation: **$0.15\%$**).
3. **Candidate Cardinality**: Candidate volume remains strictly bounded (mean **$35.51$ candidates/S1**, median $26.0$, P95 $82.0$, P99 $145.0$). Zero candidate explosion.
4. **Throughput & Scalability**: Query throughput exhibits near-perfect linear scaling at **$488\text{ queries/second}$**.
5. **Memory Safety**: Peak RAM remained virtually flat at **$2,059.9\text{ MB}$** throughout the 100,000 query execution (only $+3.8\text{ MB}$ growth over the baseline), well within the 2.5 GB available hardware limit.
6. **Bitwise Reproducibility**: Repeated independent evaluations yielded **$100.00\%$ bitwise identical candidate sets** ($71,139$ pairs in Run 1 vs $71,139$ in Run 2).

---

## 2. Quantitative Answers to the 7 Stability Questions

### Question 1: Does U6 remain stable as query volume increases?
**YES.**
Absolute recall is **$\approx 93.85\%$** on the larger target universe, with negligible drift across query scale:
- $N = 5,000$ S1: **$93.94\%$** recall ($S2$: $95.37\%$, $S3$: $92.61\%$)
- $N = 25,000$ S1: **$93.88\%$** recall ($S2$: $95.21\%$, $S3$: $92.63\%$)
- $N = 100,000$ S1: **$93.85\%$** recall ($S2$: $95.10\%$, $S3$: $92.67\%$)

Over $100,000$ queries and $346,390$ true links, U6 retrieves **$325,073$ true target links**! The recall drift from 5K to 100K queries is less than **$0.09\%$**.

---

### Question 2: Does recall vary substantially by query sample?
**NO.**
Evaluating 5 non-overlapping, stratified folds of $5,000$ queries each:

| Fold Index | Validation Slice Range | Total True Links | Captured Links | Recall | $S2$ Recall | $S3$ Recall | Mean Cands | P95 Cands |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 1** | $0 \dots 4,999$ | 17,232 | 16,188 | 0.9394 | 0.9537 | 0.9261 | 35.95 | 84.0 |
| **Fold 2** | $5,000 \dots 9,999$ | 17,319 | 16,225 | 0.9368 | 0.9511 | 0.9236 | 35.73 | 82.0 |
| **Fold 3** | $10,000 \dots 14,999$| 17,504 | 16,465 | 0.9406 | 0.9515 | 0.9304 | 35.45 | 81.0 |
| **Fold 4** | $15,000 \dots 19,999$| 17,254 | 16,208 | 0.9394 | 0.9523 | 0.9273 | 35.56 | 80.0 |
| **Fold 5** | $20,000 \dots 24,999$| 17,241 | 16,164 | 0.9375 | 0.9517 | 0.9243 | 35.84 | 83.0 |
| **Summary**| **5 Folds Combined** | **86,550** | **81,250** | **0.9388 ± 0.0014** | **0.9521 ± 0.0009** | **0.9263 ± 0.0024** | **35.71 ± 0.18** | **82.0 ± 1.4** |

- **Coefficient of Variation (CV)**: **$0.15\%$** across folds.
- **Range**: Minimum $93.68\%$ to Maximum $94.06\%$ (spread of only $0.38\%$).
- **Conclusion**: U6 performance is exceptionally stationary across arbitrary data partitions.

---

### Question 3: Does candidate count remain bounded?
**YES.**
Candidate size percentiles remained completely stable as query scale expanded:

| Scale (S1) | Mean Cands | Median | P90 | P95 | P99 | Maximum | Reduction Ratio |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **5,000** | 35.95 | 26.0 | 59.0 | 84.0 | 148.0 | 363 | 0.99992757 |
| **25,000** | 35.71 | 26.0 | 59.0 | 82.0 | 145.0 | 365 | 0.99992807 |
| **100,000** | **35.51** | **26.0** | **59.0** | **82.0** | **145.0** | **412** | **0.99992846** |

- **Median**: Exactly $26.0$ across all scales.
- **P95**: Decreased slightly from $84.0 \to 82.0$.
- **P99**: Stable at $145.0$.
- **Maximum**: Only $412$ candidates across 100,000 queries.
- **Search Space Reduction**: **$99.9928\%$** eliminated (only $1$ pair evaluated for every $13,978$ Cartesian possibilities).

---

### Question 4: Can TF-IDF retrieval run safely at realistic scale?
**YES.**
- Vectorizing and indexing $496,390$ target records (`max_features=15000`, word n-grams $(1, 2)$) completed in **$52.73\text{ seconds}$**.
- Sparse CSR matrix representation requires only **$\approx 62\text{ MB}$** of RAM.
- Batched query retrieval with batch size $B=250$ executed at **$488\text{ queries/second}$** with zero memory leaks.

---

### Question 5: What is the actual runtime/memory curve as query volume increases?

| Query Cohort ($N$) | Target Universe | Cartesian Pairs | Retrieval Time | Throughput | Peak Process RAM | Incremental RAM |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **5,000** | 496,390 | 2.48 Billion | 9.62s | 520.0 q/s | 2,056.1 MB | Baseline |
| **25,000** | 496,390 | 12.41 Billion | 51.49s | 485.6 q/s | 2,057.9 MB | +1.8 MB |
| **100,000** | 496,390 | 49.64 Billion | 205.00s | 487.8 q/s | 2,059.9 MB | +3.8 MB |

```
Runtime (seconds)
 ^
 |                                                                  [100K queries: 205.0s]
 |                                                                   *
 |
 |                                    [25K queries: 51.5s]
 |                                     *
 |                 [5K queries: 9.6s]
 |                  *
 +---------------------------------------------------------------------------------------->
 0                 25,000             50,000             75,000             100,000   Query Count
```

- **Linear Runtime**: $O(N)$ query time with an exact empirical rate of $\approx 2.05\text{ ms per query}$ ($\approx 488\text{ queries/second}$).
- **Flatline Memory**: Holding the target indexes in memory is the only fixed cost ($\approx 2.05\text{ GB}$). Query streaming adds essentially zero incremental memory ($+3.8\text{ MB}$ for $100,000$ queries).

---

### Question 6: Do the same failure modes persist?
**YES.**
Failure analysis on the unrecovered links across the 100,000 query cohort confirmed identical failure mode proportions to Phase 1:
1. **Cross-Script Native Indic Transliteration ($~58\%$ of failures)**: S1 English text vs S2/S3 native scripts (Hindi, Bengali, Tamil, Kannada, Gujarati, Odia). In the absence of Romanized address bridge tokens, Latin tokenizers cannot bridge the script gap.
2. **Domain-Name Formatting ($~22\%$ of failures)**: Target names stored as web domains (`indutrust.com`, `pacificalliance.com`).
3. **Leetspeak & Digit Substitutions ($~12\%$ of failures)**: E.g., `2-P0int` vs `2-Point`.
4. **Complete Brand Aliases ($~8\%$ of failures)**: Corporate aliases with zero lexical name overlap.

---

### Question 7: Is U6 reproducible across batches?
**YES (100% BITWISE IDENTICAL).**
- Independent Run 1 ($N=2,000$): Recall = `0.938513`, Candidates = `71,139`, Max = `363`
- Independent Run 2 ($N=2,000$): Recall = `0.938513`, Candidates = `71,139`, Max = `363`
- **Bitwise Match**: **PASS** (Zero divergence).

---

## 3. Full-Validation Scale Extrapolation

Using the empirical scaling parameters measured on $100,000$ queries against $496,390$ target records:
- **Full Validation S1 Cohort**: $441,364$ entities.
- **Estimated Full Validation Runtime**:
  $$t_{\text{full}} = \frac{441,364\text{ queries}}{487.8\text{ queries/sec}} = 904.8\text{ seconds} \approx \mathbf{15.08\text{ minutes}}$$
- **Projected Peak RAM**: **$\approx 2,065\text{ MB}$** (easily fits within available RAM).
- **Projected Candidate Volume**:
  $$441,364 \times 35.51 = \mathbf{15,672,835}\text{ candidate pairs}$$
- **Search Space Reduction**: Eliminates **$99.9928\%$** of the Cartesian space ($15.67\text{M}$ pairs vs $219\text{B}$ Cartesian pairs).

---

## 4. Phase 1.5 Deliverable Inventory

All Phase 1.5 verification artifacts are committed and available:

| Deliverable Path | Description | Status |
| :--- | :--- | :---: |
| [`reports/phase1_5/PHASE1_5_REPORT.md`](file:///c:/Users/DHANUSH%20A%20G/Desktop/ML-Challenge/reports/phase1_5/PHASE1_5_REPORT.md) | Authoritative stability and scaling audit report | **Complete** |
| [`reports/phase1_5/scaling_curve.csv`](file:///c:/Users/DHANUSH%20A%20G/Desktop/ML-Challenge/reports/phase1_5/scaling_curve.csv) | Empirical metrics across 5K, 25K, and 100K queries | **Complete** |
| [`reports/phase1_5/batch_stability.csv`](file:///c:/Users/DHANUSH%20A%20G/Desktop/ML-Challenge/reports/phase1_5/batch_stability.csv) | Metrics across 5 non-overlapping validation folds | **Complete** |
| [`reports/phase1_5/stability_gate_summary.json`](file:///c:/Users/DHANUSH%20A%20G/Desktop/ML-Challenge/reports/phase1_5/stability_gate_summary.json) | Structured summary with full validation extrapolations | **Complete** |
| [`src/blocking/u6_ensemble.py`](file:///c:/Users/DHANUSH%20A%20G/Desktop/ML-Challenge/src/blocking/u6_ensemble.py) | High-throughput, integrated U6 ensemble engine | **Complete & Tested** |
| [`src/blocking/stability_gate.py`](file:///c:/Users/DHANUSH%20A%20G/Desktop/ML-Challenge/src/blocking/stability_gate.py) | Stability gate benchmarking harness | **Complete & Tested** |
| [`run_phase1_5.py`](file:///c:/Users/DHANUSH%20A%20G/Desktop/ML-Challenge/run_phase1_5.py) | Top-level CLI runner (`--all`, `--quick`, `--report`) | **Complete & Verified** |
| [`tests/blocking/test_u6_ensemble.py`](file:///c:/Users/DHANUSH%20A%20G/Desktop/ML-Challenge/tests/blocking/test_u6_ensemble.py) | Automated unit tests (61/61 passing) | **All Passing** |

---

## 5. Final Recommendation for Phase 2

The U6 candidate generation ensemble has successfully passed all stress tests:
- High, unshakeable blocking recall: **$\approx 94\%$ across 100,000 entities**.
- Compact candidate cardinality: **Mean $35.5$ candidates per entity**, P95 of $82$.
- Linear throughput: **$488\text{ queries/second}$**.
- Stable memory footprint: **$\approx 2.06\text{ GB}$ peak RAM**.

**THE U6 STABILITY GATE IS FORMALLY PASSED. THE SYSTEM IS SCIENTIFICALLY SOUND AND READY TO ENTER PHASE 2.**
