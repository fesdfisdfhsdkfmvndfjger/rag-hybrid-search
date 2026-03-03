from __future__ import annotations
import numpy as np
import faiss
from pathlib import Path

class VectorStore:
    def __init__(self, embeddings: np.ndarray):
        embeddings = np.ascontiguousarray(self._f32(embeddings))
        self.dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(embeddings)

    def save(self, path: Path) -> None:
        faiss.write_index(self.index, str(path))

    @classmethod
    def load(cls, path: Path) -> "VectorStore":
        inst = cls.__new__(cls)
        inst.index = faiss.read_index(str(path))
        inst.dim = inst.index.d
        return inst

    def search(self, query_embedding: np.ndarray, top_k: int):
        q = np.ascontiguousarray(self._f32(query_embedding))
        if q.ndim == 1: 
            q = q.reshape(1, -1)
            
        top_k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(q, top_k)
        return indices[0], scores[0]

    @staticmethod
    def _f32(a: np.ndarray) -> np.ndarray:
        return a.astype(np.float32) if a.dtype != np.float32 else a