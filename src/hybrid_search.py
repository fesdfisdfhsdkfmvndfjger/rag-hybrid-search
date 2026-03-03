from __future__ import annotations
import numpy as np
from typing import List, Dict, Tuple
from rank_bm25 import BM25Okapi


class HybridSearch:
    """
    Combines BM25 keyword search with FAISS vector search via
    score-based fusion (weighted average of normalized scores).
    """

    def __init__(self, chunks: List[Dict], vector_store):
        self.chunks = chunks
        self.vector_store = vector_store
        texts = [c["text"] for c in chunks]
        tokenized = [t.lower().split() for t in texts]
        self.bm25 = BM25Okapi(tokenized)

    def bm25_search(self, query: str, top_k: int) -> Tuple[List[int], List[float]]:
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        ranked = np.argsort(scores)[::-1][:top_k].tolist()
        top_scores = scores[ranked].tolist()

        # Min-max normalize to 0-1
        if top_scores:
            top_arr = np.array(top_scores)
            min_s, max_s = top_arr.min(), top_arr.max()
            if max_s > min_s:
                norm_scores = ((top_arr - min_s) / (max_s - min_s)).tolist()
            else:
                norm_scores = [0.0] * len(top_scores)
        else:
            norm_scores = []

        return ranked, norm_scores

    def vector_search(
        self, query_embedding: np.ndarray, top_k: int
    ) -> Tuple[List[int], List[float]]:
        indices, scores = self.vector_store.search(query_embedding, top_k)
        return indices.tolist(), scores.tolist()

    def hybrid_search(
        self,
        query: str,
        query_embedding: np.ndarray,
        top_k: int,
        bm25_weight: float = 0.3,
        vector_weight: float = 0.7,
    ) -> Tuple[List[int], List[float]]:
        """Weighted fusion of normalized BM25 + vector scores."""
        bm25_indices, bm25_scores = self.bm25_search(query, top_k * 2)
        vec_indices, vec_scores = self.vector_search(query_embedding, top_k * 2)

        
        bm25_dict = dict(zip(bm25_indices, bm25_scores))
        vec_dict = dict(zip(vec_indices, vec_scores))

       
        all_indices = set(bm25_indices) | set(vec_indices)

        
        hybrid_scores: Dict[int, float] = {}
        for idx in all_indices:
            hybrid_scores[idx] = (
                bm25_weight * bm25_dict.get(idx, 0.0)
                + vector_weight * vec_dict.get(idx, 0.0)
            )

        ranked = sorted(hybrid_scores, key=hybrid_scores.get, reverse=True)[:top_k]
        scores = [hybrid_scores[i] for i in ranked]
        return ranked, scores