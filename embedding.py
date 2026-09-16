"""
Single embedding implementation used by BOTH ingestion and query.

Important:
- Documents: passage:<text>
- Queries:   query:<text>
- Model: intfloat/multilingual-e5-base
- Dimension: 768

The exact same model and prefixes must be used on both sides of vector search.
"""

from functools import lru_cache
import threading
from typing import Iterable, List, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
EMBEDDING_DIM = 768

_MODEL_LOCK = threading.Lock()


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL)


def _encode(texts: Sequence[str]) -> List[List[float]]:
    clean = [str(x).strip() for x in texts]
    if not clean:
        return []

    with _MODEL_LOCK:
        vectors = get_model().encode(
            clean,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=32,
        )

    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    if arr.shape[1] != EMBEDDING_DIM:
        raise RuntimeError(
            f"Embedding dimension mismatch: got {arr.shape[1]}, "
            f"expected {EMBEDDING_DIM} for {EMBEDDING_MODEL}"
        )

    return arr.tolist()


def get_query_embedding(text: str) -> List[float]:
    text = str(text).strip()
    if not text:
        raise ValueError("Query text is empty.")
    return _encode([f"query: {text}"])[0]


def get_document_embeddings(texts: Iterable[str]) -> List[List[float]]:
    texts = [str(x).strip() for x in texts if str(x).strip()]
    if not texts:
        return []
    return _encode([f"passage: {text}" for text in texts])
