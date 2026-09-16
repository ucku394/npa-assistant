"""
Embedding engine for the Belarus occupational-safety Telegram bot.

IMPORTANT:
- Model: intfloat/multilingual-e5-base
- Dimension: 768
- Documents use: passage: <text>
- Queries use:   query: <text>

The same model and prefixes MUST be used for both ingestion and search.
"""

import logging
import os
import threading
from functools import lru_cache
from typing import Iterable, List, Sequence

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


logger = logging.getLogger(__name__)


# ============================================================
# CONFIG
# ============================================================

EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
EMBEDDING_DIM = 768

# Railway / small CPU container:
# keep the number of native threads low to reduce RAM usage.
CPU_THREADS = int(os.getenv("EMBEDDING_CPU_THREADS", "1"))

# Limit BLAS/OpenMP thread creation.
os.environ.setdefault("OMP_NUM_THREADS", str(CPU_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(CPU_THREADS))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(CPU_THREADS))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    torch.set_num_threads(CPU_THREADS)
except Exception:
    pass

try:
    torch.set_num_interop_threads(1)
except Exception:
    pass


_MODEL_LOCK = threading.Lock()


# ============================================================
# MODEL
# ============================================================

@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """
    Load the E5 model exactly once per process.

    The cache is important:
    the model must NOT be loaded for every Telegram message.
    """

    logger.info(
        "EMBEDDING | loading model: %s | cpu_threads=%s",
        EMBEDDING_MODEL,
        CPU_THREADS,
    )

    model = SentenceTransformer(
        EMBEDDING_MODEL,
        device="cpu",
    )

    model.eval()

    dimension = model.get_sentence_embedding_dimension()

    logger.info(
        "EMBEDDING | model loaded | dimension=%s | device=cpu",
        dimension,
    )

    if dimension != EMBEDDING_DIM:
        raise RuntimeError(
            f"Embedding dimension mismatch: "
            f"got {dimension}, expected {EMBEDDING_DIM}"
        )

    return model


# ============================================================
# WARMUP
# ============================================================

def warmup_model() -> None:
    """
    Load and test the model during application startup.

    This makes model initialization explicit instead of hiding it
    inside the first user request.
    """

    logger.info("EMBEDDING | startup warmup started")

    model = get_model()

    test_text = "query: проверка загрузки модели"

    with _MODEL_LOCK:
        with torch.inference_mode():
            vector = model.encode(
                test_text,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=1,
            )

    arr = np.asarray(vector, dtype=np.float32).reshape(-1)

    if arr.shape[0] != EMBEDDING_DIM:
        raise RuntimeError(
            f"Warmup embedding dimension mismatch: "
            f"got {arr.shape[0]}, expected {EMBEDDING_DIM}"
        )

    logger.info(
        "EMBEDDING | startup warmup completed | dimension=%s",
        arr.shape[0],
    )


# ============================================================
# INTERNAL ENCODE
# ============================================================

def _encode(texts: Sequence[str]) -> List[List[float]]:
    clean = [str(x).strip() for x in texts if str(x).strip()]

    if not clean:
        return []

    logger.info(
        "EMBEDDING | encoding %s text(s)",
        len(clean),
    )

    model = get_model()

    with _MODEL_LOCK:
        with torch.inference_mode():
            vectors = model.encode(
                clean,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=1,
            )

    arr = np.asarray(vectors, dtype=np.float32)

    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    if arr.shape[1] != EMBEDDING_DIM:
        raise RuntimeError(
            f"Embedding dimension mismatch: "
            f"got {arr.shape[1]}, expected {EMBEDDING_DIM}"
        )

    logger.info(
        "EMBEDDING | encoding completed | shape=%s",
        tuple(arr.shape),
    )

    return arr.tolist()


# ============================================================
# QUERY
# ============================================================

def get_query_embedding(text: str) -> List[float]:
    """
    Create embedding for a user search query.

    E5 requires the query prefix:
        query:
    """

    text = str(text).strip()

    if not text:
        raise ValueError("Query text is empty.")

    logger.info(
        "EMBEDDING | query started | chars=%s",
        len(text),
    )

    result = _encode(
        [f"query: {text}"]
    )[0]

    logger.info(
        "EMBEDDING | query completed | dimension=%s",
        len(result),
    )

    return result


# ============================================================
# DOCUMENTS
# ============================================================

def get_document_embeddings(
    texts: Iterable[str],
) -> List[List[float]]:
    """
    Create embeddings for document chunks.

    E5 requires:
        passage:
    """

    texts = [
        str(x).strip()
        for x in texts
        if str(x).strip()
    ]

    if not texts:
        return []

    return _encode(
        [f"passage: {text}" for text in texts]
    )
