from __future__ import annotations
import os
import pickle
import tempfile
from pathlib import Path
import numpy as np

def save_chunks(chunks: list, path: Path) -> None:
    _atomic_write(path, pickle.dumps(chunks, protocol=pickle.HIGHEST_PROTOCOL))

def load_chunks(path: Path) -> list:
    with open(path, "rb") as f:
        return pickle.load(f)

def save_embeddings(embeddings: np.ndarray, path: Path) -> None:
    tmp = path.with_suffix(".tmp.npy")
    np.save(tmp, embeddings)
    tmp.replace(path)

def load_embeddings(path: Path) -> np.ndarray:
    return np.load(path)

def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp_")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except Exception:
        try: os.unlink(tmp)
        except OSError: pass
        raise