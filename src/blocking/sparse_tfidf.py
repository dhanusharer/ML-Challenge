"""Block Family F: Sparse TF-IDF Top-K Retrieval Blocker.

Uses sparse character n-gram or word TF-IDF vectorization and sparse matrix dot-products
to retrieve top-k candidate target records for each query record.

NOTE: Retrieval similarity score is strictly a candidate generation mechanism,
NOT a final entity resolution match decision.
"""

from typing import Any, Dict, List, Optional, Sequence, Set
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import time

from src.blocking.base import BaseBlocker
from src.representations.name import standard_clean, strip_legal_suffixes


class SparseTfidfBlocker(BaseBlocker):
    """Memory-safe sparse TF-IDF top-k candidate generator."""

    def __init__(
        self,
        name: str = "BLOCK-TFIDF-WORD-TOPK",
        analyzer: str = "word",
        ngram_range: tuple = (1, 2),
        top_k: int = 25,
        min_df: int = 3,
        max_df: float = 0.20,
        max_features: int = 20000,
    ):
        super().__init__(name=name, config={
            "analyzer": analyzer,
            "ngram_range": ngram_range,
            "top_k": top_k,
            "min_df": min_df,
            "max_df": max_df,
            "max_features": max_features,
        })
        self.top_k = top_k
        self.vectorizer = TfidfVectorizer(
            analyzer=analyzer,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            max_features=max_features,
            sublinear_tf=True,
            dtype=np.float32,
        )
        self.target_matrix: Optional[csr_matrix] = None
        self.target_ids: List[str] = []

    def _prepare_text(self, record: Dict[str, str]) -> str:
        name = strip_legal_suffixes(record.get("business_name", ""))
        addr = standard_clean(record.get("business_address", ""))
        return f"{name} {addr}".strip()

    def fit(self, targets: Sequence[Dict[str, str]]) -> "SparseTfidfBlocker":
        t0 = time.time()
        self.target_ids = [rec["entity_id"] for rec in targets]
        texts = [self._prepare_text(rec) for rec in targets]
        self.target_matrix = self.vectorizer.fit_transform(texts)
        self.indexing_time_sec = round(time.time() - t0, 3)
        return self

    def retrieve(self, queries: Sequence[Dict[str, str]]) -> Dict[str, Set[str]]:
        t0 = time.time()
        query_texts = [self._prepare_text(q) for q in queries]
        query_matrix = self.vectorizer.transform(query_texts)

        candidates: Dict[str, Set[str]] = {}
        batch_size = 100

        # Query in smaller chunks to keep intermediate sparse products bounded
        for start_idx in range(0, len(queries), batch_size):
            end_idx = min(start_idx + batch_size, len(queries))
            sub_q = query_matrix[start_idx:end_idx]

            # Sparse dot product: (batch_size, num_targets)
            sim_batch = sub_q.dot(self.target_matrix.T)

            for i in range(end_idx - start_idx):
                s1_id = queries[start_idx + i]["entity_id"]
                row = sim_batch.getrow(i)
                if row.nnz == 0:
                    candidates[s1_id] = set()
                    continue

                col_indices = row.indices
                data_scores = row.data

                # Top-k selection
                if len(data_scores) > self.top_k:
                    top_idx = np.argpartition(data_scores, -self.top_k)[-self.top_k:]
                    selected_cols = col_indices[top_idx]
                else:
                    selected_cols = col_indices

                candidates[s1_id] = {self.target_ids[c] for c in selected_cols}

        self.query_time_sec = round(time.time() - t0, 3)
        return candidates

