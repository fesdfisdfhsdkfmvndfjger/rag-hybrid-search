from __future__ import annotations
import numpy as np
from functools import lru_cache
from sentence_transformers import SentenceTransformer, CrossEncoder

_embedding_model: SentenceTransformer | None = None
_rerank_model: CrossEncoder | None = None

def _best_device() -> str:
    try:
        import torch
        if torch.cuda.is_available(): return "cuda"
        if torch.backends.mps.is_available(): return "mps"
    except ImportError:
        pass
    return "cpu"

def _get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device=_best_device())
    return _embedding_model

def _get_rerank_model() -> CrossEncoder:
    global _rerank_model
    if _rerank_model is None:
        _rerank_model = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L-2-v2", device=_best_device())
    return _rerank_model

def embed_chunks(chunks: list[str], batch_size: int = 64) -> np.ndarray:
    model = _get_embedding_model()
    return model.encode(
        chunks, batch_size=batch_size, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=True,
    ).astype(np.float32)

def embed_query(query: str) -> np.ndarray:
    return _embed_query_cached(query)

@lru_cache(maxsize=512)
def _embed_query_cached(query: str) -> np.ndarray:
    model = _get_embedding_model()
    return model.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
    ).astype(np.float32)

def rerank_chunks(query: str, chunks: list[str], top_n: int) -> list[int]:
    if not chunks:
        return []
    model = _get_rerank_model()
    scores = model.predict([[query, chunk] for chunk in chunks], show_progress_bar=False)
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return ranked[:top_n]

def rerank_scores(query: str, chunks: list[str]) -> list[float]:
    """Returns raw rerank scores for all chunks (for heatmap / transparency)."""
    if not chunks:
        return []
    model = _get_rerank_model()
    scores = model.predict([[query, chunk] for chunk in chunks], show_progress_bar=False)
    return scores.tolist()