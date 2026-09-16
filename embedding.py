from functools import lru_cache
import logging
import os
import threading
from typing import Iterable, List, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer


logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
EMBEDDING_DIM = 384

_MODEL_LOCK = threading.Lock()


@lru_cache(maxsize=1)
def get_model():
    pid = os.getpid()

    logger.info(
        "EMBEDDING | loading model=%s | pid=%s",
        EMBEDDING_MODEL,
        pid,
    )

    try:
        model = SentenceTransformer(
            EMBEDDING_MODEL,
            device="cpu",
        )

        dimension = _dimension(model)

        logger.info(
            "EMBEDDING | model loaded successfully | "
            "model=%s | dimension=%s | pid=%s",
            EMBEDDING_MODEL,
            dimension,
            pid,
        )

        if dimension != EMBEDDING_DIM:
            raise RuntimeError(
                f"Embedding dimension mismatch: "
                f"model reports {dimension}, expected {EMBEDDING_DIM}"
            )

        return model

    except Exception:
        logger.exception(
            "EMBEDDING | model loading FAILED | model=%s | pid=%s",
            EMBEDDING_MODEL,
            pid,
        )
        raise


def _dimension(model):
    method = getattr(model, "get_embedding_dimension", None)

    if callable(method):
        return int(method())

    legacy = getattr(
        model,
        "get_sentence_embedding_dimension",
        None,
    )

    if callable(legacy):
        return int(legacy())

    raise RuntimeError(
        "Cannot determine embedding dimension."
    )


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

    logger.info(
        "EMBEDDING | encode started | count=%s | pid=%s",
        len(clean),
        os.getpid(),
    )

    with _MODEL_LOCK:

        model = get_model()

        vectors = model.encode(
            clean,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=8,
        )

        dim = _dimension(model)

    arr = np.asarray(
        vectors,
        dtype=np.float32,
    )

    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    if arr.shape[1] != EMBEDDING_DIM:
        raise RuntimeError(
            f"Embedding dimension mismatch: "
            f"got {arr.shape[1]}, "
            f"expected {EMBEDDING_DIM} "
            f"for {EMBEDDING_MODEL}; "
            f"model reports {dim}."
        )

    logger.info(
        "EMBEDDING | encode completed | "
        "count=%s | dimension=%s | pid=%s",
        len(clean),
        arr.shape[1],
        os.getpid(),
    )

    return arr.tolist()


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


def get_document_embeddings(
    texts: Iterable[str],
) -> List[List[float]]:

    texts = [
        str(x).strip()
        for x in texts
        if str(x).strip()
    ]

    return _encode(
        [f"passage: {x}" for x in texts]
    )