# Frozen Policy Specification — Phase 3

**Amazon ML Challenge 2026 — Business Entity Resolution**  
**Status:** Frozen and Ready for Phase 4 Execution  

```
                          END-TO-END PREDICTION PIPELINE

                   ┌─────────────────────────────────────────┐
                   │        Source-1 Query Entity (r_S1)     │
                   └────────────────────┬────────────────────┘
                                        │
                                        ▼
                   ┌─────────────────────────────────────────┐
                   │   Pass 1: Candidate Arm B Retrieval     │
                   │   (Domain + FreqAware + TF-IDF k=40)    │
                   └────────────────────┬────────────────────┘
                                        │
                                        ▼
                   ┌─────────────────────────────────────────┐
                   │ 43-Feature Extraction & LightGBM Scoring│
                   └────────────────────┬────────────────────┘
                                        │
                                        ▼
                   ┌─────────────────────────────────────────┐
                   │       Check Rescue Condition:           │
                   │       top_score < 0.75 or margin < 0.06 │
                   └────────────┬───────────────────┬────────┘
                                │ Yes               │ No
                                ▼                   │
                   ┌─────────────────────────┐      │
                   │ Pass 2: Arm C Recovery  │      │
                   │ + Merge Candidate Pool  │      │
                   └────────────┬────────────┘      │
                                │                   │
                                └─────────┬─────────┘
                                          │
                                          ▼
                   ┌─────────────────────────────────────────┐
                   │   Relative Margin Set-Selection Policy  │
                   │                                         │
                   │ 1. If top_score < 0.75:                 │
                   │      Output [] (Singleton Protection)   │
                   │                                         │
                   │ 2. Accept top candidate c_1             │
                   │                                         │
                   │ 3. For c_i (i >= 2):                    │
                   │      Accept if score_i >= 0.80 AND      │
                   │      (top_score - score_i) <= 0.08      │
                   │      up to max_k = 3                    │
                   └────────────────────┬────────────────────┘
                                        │
                                        ▼
                   ┌─────────────────────────────────────────┐
                   │ Final Predicted Match Set for S1 Entity │
                   └─────────────────────────────────────────┘
```

### Invariant Principles
1. **Rule 5 Invariance**: Only target IDs prefixed with `S2-` or `S3-` are emitted. S1 IDs are never emitted.
2. **Deduplication**: Predictions are sorted by score descending, deduplicated by max score, with lexicographical tie-breaking.
3. **Zero-Match Precision**: True singletons (~5.8% of the entity population) are shielded from distractor false merges, preventing catastrophic macro precision penalties under $F_{0.5}$.
4. **Multi-Match Recall**: Legitimate multi-branch business entities with tightly clustered match scores ($\le 0.08$ gap) are recovered up to $K=3$.