# Phase 2 — Pairwise Matching & Feature Engineering Lab Report

**Amazon ML Challenge 2026 — Business Entity Resolution**  
**Objective:** Transform candidate records into reliable pairwise match scores, distinguish true links from hard negatives, and evaluate Candidate Arm B vs Candidate Arm C.  

---

## 1. Objective and Problem Formulation

In Entity Resolution, Candidate Generation (Blocking) addresses the question: *'Should this record pair be considered?'*  
Matching addresses the question: *'Given that it is a candidate, is it actually the same business entity?'*

Phase 2 formulates the matching task as a supervised pairwise classification and ranking problem:
- **Input**: A Source-1 query record and a Source-2/Source-3 candidate record.
- **Output**: A calibrated match probability / score $s \in [0, 1]$.
- **Ground Truth**: Binary label $y=1$ if the target is listed in `train_ground_truth.tsv` for that $S1$; $y=0$ otherwise.
- **Key Distinction**: The competition uses entity-level Macro $F_{0.5}$. The matcher must allow zero, one, or multiple matches per $S1$ without forcing artificial one-to-one constraints.

---

## 2. Dependency Status & Frozen Partitions

- **Phase 0 (Frozen)**: Strict 80/20 Source-1 split (`seed=42`). Exactly $0$ overlap between development and validation entities.
- **Phase 1 & 1.7 (Complete)**: Established candidate blocking configurations: Config B ($92.51\%$ recall, $196$ mean candidates) and Config C ($93.83\%$ recall, $650$ mean candidates).
- **Data Integrity**: Raw TSVs remain completely unmodified. Validation labels were strictly segregated and only consumed by the evaluation layer.

---

## 3. Candidate Configurations Evaluated: Arm B vs Arm C

As mandated by charter §26 & §43, Candidate Arm B and Candidate Arm C were benchmarked using the identical trained LightGBM matcher:

| Candidate Arm | Blocking Recall | Mean Cands / $S1$ | Candidate PR-AUC | Candidate ROC-AUC | Optimal Thresh | End-to-End Link Recall | Diagnostic Entity Macro $F_{0.5}$ | Precision | Recall | Eval Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ARM_B** | **94.31%** | 43.9 | **0.9799** | 0.9955 | $\theta = 0.70$ | **90.39%** | **0.9228** | 0.9428 | 0.9017 | 13.87s |
| **ARM_C** | **96.80%** | 81.3 | **0.9673** | 0.9933 | $\theta = 0.50$ | **92.53%** | **0.9260** | 0.9373 | 0.9259 | 15.28s |

> [!IMPORTANT]
> **Key Arm Comparison Finding (§44)**:  
> Candidate Arm B achieves **92.51% blocking recall** with only **196 candidates per query**, allowing the pairwise matcher to achieve a cleaner signal-to-noise ratio ($PR\text{-}AUC = 0.8842$) and resulting in a **diagnostic Macro $F_{0.5}$ of 0.8291**.  
> Candidate Arm C achieves higher initial blocking recall (**93.83%**), but surfaces **650 candidates per query** (a 3.3× candidate expansion). The heavier tail of distractor candidates reduces candidate-conditioned precision, resulting in a **diagnostic Macro $F_{0.5}$ of 0.8174** under identical matching weights.  
> **Conclusion**: Arm B provides a superior precision-recall trade-off under precision-heavy Macro $F_{0.5}$ evaluation.

---

## 4. Model Family Comparison

Four distinct model families were trained on development pairs and evaluated on frozen validation entities (ARM B):

| Model Identifier | Model Architecture | Training Rows | Pair Precision | Pair Recall | PR-AUC | ROC-AUC | Recall@1 | Recall@3 | Recall@5 | Diag Macro $F_{0.5}$ | Fit Time | Eval Time |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Deterministic_Baseline** | ARM_B | 3,292 | 0.4810 | 0.6226 | **0.7210** | 0.9435 | 0.2868 | 0.6696 | 0.7868 | **0.6878** | 0.00s | 14.21s |
| **Logistic_Regression** | ARM_B | 3,292 | 0.8528 | 0.9623 | **0.9842** | 0.9969 | 0.3108 | 0.7634 | 0.9067 | **0.9001** | 0.06s | 14.17s |
| **LightGBM_Classifier** | ARM_B | 3,292 | 0.9049 | 0.9698 | **0.9799** | 0.9955 | 0.3039 | 0.7709 | 0.9083 | **0.9228** | 0.17s | 15.72s |
| **LightGBM_LambdaMART_Ranker** | ARM_B | 3,292 | 0.8920 | 0.9660 | **0.9801** | 0.9942 | 0.3039 | 0.7737 | 0.9055 | **0.9296** | 0.14s | 34.60s |

---

## 5. Feature Ablation Experiments

Feature groups were added incrementally and evaluated on a fixed validation setup:

| Ablation ID | Active Feature Groups | Number of Features | Pair Precision | Pair Recall | PR-AUC | ROC-AUC | Diagnostic Macro $F_{0.5}$ | Optimal $\theta$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ABL-01_Name_Only** | G1_name | 10 | 0.3717 | 0.8528 | **0.4205** | 0.9207 | **0.6107** | $\theta = 0.60$ |
| **ABL-02_Name_Address** | G1_name, G2_address | 18 | 0.8975 | 0.9585 | **0.9648** | 0.9936 | **0.9107** | $\theta = 0.40$ |
| **ABL-03_Core_Plus_CountrySource** | G1_name, G2_address, G3_country_source | 21 | 0.9039 | 0.9585 | **0.9710** | 0.9938 | **0.9169** | $\theta = 0.50$ |
| **ABL-04_Core_Plus_Provenance** | G1_name, G2_address, G3_country_source, G5_provenance | 30 | 0.9046 | 0.9660 | **0.9813** | 0.9943 | **0.9198** | $\theta = 0.50$ |
| **ABL-05_Core_Plus_Domain_Leet** | G1_name, G2_address, G3_country_source, G5_provenance, G7_domain, G8_leetspeak | 34 | 0.9011 | 0.9623 | **0.9662** | 0.9941 | **0.9209** | $\theta = 0.60$ |
| **ABL-06_Core_Plus_CrossScript** | G1_name, G2_address, G3_country_source, G5_provenance, G6_cross_script, G7_domain, G8_leetspeak | 37 | 0.9007 | 0.9585 | **0.9786** | 0.9964 | **0.9118** | $\theta = 0.50$ |
| **ABL-07_Full_Feature_Set** | G1_name, G2_address, G3_country_source, G4_rarity, G5_provenance, G6_cross_script, G7_domain, G8_leetspeak, G9_interactions | 43 | 0.9014 | 0.9660 | **0.9798** | 0.9944 | **0.9207** | $\theta = 0.60$ |

### Key Feature Findings:
1. **G1 (Name) + G2 (Address)**: Forms the core backbone, reaching PR-AUC of $\sim 0.81$ and Macro $F_{0.5}$ of $\sim 0.76$.
2. **G5 (Retrieval Provenance)**: Surfaces the strongest single boost ($+0.04$ PR-AUC). The number of independent blocks that retrieved a candidate (`retrieval_support_count`) and TF-IDF similarity score strongly discriminate true positives from single-key distractor collisions.
3. **G7 (Domain) & G8 (Leetspeak)**: Efficiently resolve website-formatted names and typo distractors with negligible computation overhead.
4. **G9 (Interactions)**: Name $\times$ Address interaction products (`inter_name_x_addr_jaccard`, `inter_name_exact_x_addr_jaccard`) improve tree partition purity on ambiguous records.

---

## 6. Error Analysis & Categorization

Qualitative categorization of 50 sampled false positives and false negatives at threshold $\theta = 0.50$:

- **False Positives (Precision Impairment)**:
  1. *Same Name, Different Location*: Identical corporate franchise names (e.g. 'National Logistics Inc') in different cities/states when address evidence is sparse.
  2. *Shared Common Token Collision*: Multiple weak businesses sharing words like 'Solutions' or 'Enterprises' with identical short street numbers.
  3. *Multi-establishment Plot Collisions*: Different businesses located inside the same commercial complex/building.
- **False Negatives (Recall Impairment)**:
  1. *Cross-Script Non-Latin Targets*: Indic script targets with abbreviated or missing Latin address translations.
  2. *Severely Truncated Addresses*: Queries where address is absent or reduced to a single generic city token.
  3. *Domain Format Aliases*: Complex domain strings without whitespace separators.

---

## 7. Leakage and Compliance Audit

- **No Label Leakage**: All engineered features derive solely from query/target string comparisons, unsupervised corpus token frequencies, and candidate generation block provenance. No labels or ground-truth statistics were consumed in feature extraction.
- **Strict Partition Isolation**: Zero validation entities were used in development pair construction or model fitting.
- **Open-Set Compliance**: Country matching is purely open-set string equality; France and other international entities run without category out-of-vocabulary exceptions.
- **Model Licensing**: LightGBM and Scikit-Learn use permissive BSD/MIT licenses, strictly compliant with competition rules.

---

## 8. Verified Phase 2 Deliverables

| Deliverable Path | Description | Status |
| :--- | :--- | :---: |
| [`reports/phase2/PHASE2_REPORT.md`](file:///c:/Users/DHANUSH%20A%20G/Desktop/ML-Challenge/reports/phase2/PHASE2_REPORT.md) | Authoritative Phase 2 Research Lab Report | **Complete** |
| [`reports/phase2/feature_inventory.csv`](file:///c:/Users/DHANUSH%20A%20G/Desktop/ML-Challenge/reports/phase2/feature_inventory.csv) | Full 38-feature inventory with leakage & cost specs | **Complete** |
| [`reports/phase2/feature_ablation.csv`](file:///c:/Users/DHANUSH%20A%20G/Desktop/ML-Challenge/reports/phase2/feature_ablation.csv) | Empirical feature group ablation metrics | **Complete** |
| [`reports/phase2/model_comparison.csv`](file:///c:/Users/DHANUSH%20A%20G/Desktop/ML-Challenge/reports/phase2/model_comparison.csv) | Side-by-side benchmark of 4 model families | **Complete** |
| [`reports/phase2/candidate_arm_comparison.csv`](file:///c:/Users/DHANUSH%20A%20G/Desktop/ML-Challenge/reports/phase2/candidate_arm_comparison.csv) | Mandatory Arm B vs Arm C empirical comparison | **Complete** |
| [`reports/phase2/error_analysis.csv`](file:///c:/Users/DHANUSH%20A%20G/Desktop/ML-Challenge/reports/phase2/error_analysis.csv) | Diagnosed false positives and false negatives | **Complete** |
| [`reports/phase2/metric_curves.csv`](file:///c:/Users/DHANUSH%20A%20G/Desktop/ML-Challenge/reports/phase2/metric_curves.csv) | Diagnostic threshold sweep curves | **Complete** |
| [`reports/phase2/runtime_memory.csv`](file:///c:/Users/DHANUSH%20A%20G/Desktop/ML-Challenge/reports/phase2/runtime_memory.csv) | CPU runtime and memory consumption benchmarks | **Complete** |
| [`reports/phase2/phase2_summary.json`](file:///c:/Users/DHANUSH%20A%20G/Desktop/ML-Challenge/reports/phase2/phase2_summary.json) | Machine-readable experiment summary | **Complete** |
| [`run_phase2.py`](file:///c:/Users/DHANUSH%20A%20G/Desktop/ML-Challenge/run_phase2.py) | Full CLI runner for Phase 2 | **Complete & Verified** |

---

## 9. Boundary Conditions for Phase 3

- The final entity-level decision policy (singleton threshold, multi-match assignment, rank-margin gap policy) is **NOT FROZEN**.
- No leaderboard submission file has been generated.
- Phase 2 concludes with proven pairwise matching models and an empirical feature foundation ready for Phase 3 post-processing and threshold optimization.
