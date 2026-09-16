from functools import lru_cache
import threading
from typing import Iterable, List, Sequence
import numpy as np
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
EMBEDDING_DIM = 384
_MODEL_LOCK = threading.Lock()

@lru_cache(maxsize=1)
def get_model():
    return SentenceTransformer(EMBEDDING_MODEL)

def _dimension(model):
    method = getattr(model, "get_embedding_dimension", None)
    if callable(method):
        return int(method())
    legacy = getattr(model, "get_sentence_embedding_dimension", None)
    if callable(legacy):
        return int(legacy())
    raise RuntimeError("Cannot determine embedding dimension.")

def _encode(texts: Sequence[str]) -> List[List[float]]:
    clean=[str(x).strip() for x in texts if str(x).strip()]
    if not clean: return []
    with _MODEL_LOCK:
        model=get_model()
        vectors=model.encode(clean, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False, batch_size=32)
        dim=_dimension(model)
    arr=np.asarray(vectors,dtype=np.float32)
    if arr.ndim==1: arr=arr.reshape(1,-1)
    if arr.shape[1] != EMBEDDING_DIM:
        raise RuntimeError(f"Embedding dimension mismatch: got {arr.shape[1]}, expected {EMBEDDING_DIM} for {EMBEDDING_MODEL}; model reports {dim}.")
    return arr.tolist()

def get_query_embedding(text: str) -> List[float]:
    text=str(text).strip()
    if not text: raise ValueError("Query text is empty.")
    return _encode([f"query: {text}"])[0]

def get_document_embeddings(texts: Iterable[str]) -> List[List[float]]:
    texts=[str(x).strip() for x in texts if str(x).strip()]
    return _encode([f"passage: {x}" for x in texts])
