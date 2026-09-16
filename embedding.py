"""
Local embedding implementation for BOTH ingestion and query.

Model:
    intfloat/multilingual-e5-small

Dimension:
    384

Important:
    Documents -> passage:<text>
    Queries   -> query:<text>

The SAME model and prefixes must be used for
both indexing and searching.
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

EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
EMBEDDING_DIM = 384

CPU_THREADS = int(
    os.getenv("EMBEDDING_CPU_THREADS", "1")
)

BATCH_SIZE = int(
    os.getenv("EMBEDDING_BATCH_SIZE", "8")
)


# ============================================================
# CPU OPTIMIZATION
# ============================================================

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
    logger.info(
        "EMBEDDING | loading model: %s | "
        "dimension=%s | cpu_threads=%s | batch_size=%s",
        EMBEDDING_MODEL,
        EMBEDDING_DIM,
        CPU_THREADS,
        BATCH_SIZE,
    )

    model = SentenceTransformer(
        EMBEDDING_MODEL,
        device="cpu",
    )

    model.eval()

    actual_dimension = (
        model.get_sentence_embedding_dimension()
    )

    if actual_dimension != EMBEDDING_DIM:
        raise RuntimeError(
            "Embedding dimension mismatch: "
            f"model returned {actual_dimension}, "
            f"expected {EMBEDDING_DIM}"
        )

    logger.info(
        "EMBEDDING | model loaded successfully | "
        "dimension=%s",
        actual_dimension,
    )

    return model


# ============================================================
# ENCODE
# ============================================================

def _encode(
    texts: Sequence[str],
) -> List[List[float]]:

    clean = [
        str(x).strip()
        for x in texts
        if str(x).strip()
    ]

    if not clean:
        return []

    with _MODEL_LOCK:

        model = get_model()

        with torch.inference_mode():

            vectors = model.encode(
                clean,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=BATCH_SIZE,
            )

    arr = np.asarray(
        vectors,
        dtype=np.float32,
    )

    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    if arr.shape[1] != EMBEDDING_DIM:
        raise RuntimeError(
            "Embedding dimension mismatch: "
            f"got {arr.shape[1]}, "
            f"expected {EMBEDDING_DIM}"
        )

    return arr.tolist()


# ============================================================
# QUERY EMBEDDING
# ============================================================

def get_query_embedding(
    text: str,
) -> List[float]:

    text = str(text).strip()

    if not text:
        raise ValueError(
            "Query text is empty."
        )

    return _encode(
        [f"query: {text}"]
    )[0]


# ============================================================
# DOCUMENT EMBEDDINGS
# ============================================================

def get_document_embeddings(
    texts: Iterable[str],
) -> List[List[float]]:

    texts = [
        str(x).strip()
        for x in texts
        if str(x).strip()
    ]

    if not texts:
        return []

    return _encode(
        [
            f"passage: {text}"
            for text in texts
        ]
    )


# ============================================================
# STARTUP WARMUP
# ============================================================

def warmup_model() -> None:
    """
    Explicitly loads the model during application startup.

    This prevents the first user request from triggering
    a large model initialization.
    """

    logger.info(
        "EMBEDDING | startup warmup started"
    )

    get_model()

    logger.info(
        "EMBEDDING | startup warmup completed"
    )
