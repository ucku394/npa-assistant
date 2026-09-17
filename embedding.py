import os

# ============================================================
# ENVIRONMENT OPTIMIZATION
# IMPORTANT:
# These variables must be set BEFORE importing torch.
# ============================================================

os.environ.setdefault(
    "TOKENIZERS_PARALLELISM",
    "false",
)

os.environ.setdefault(
    "OMP_NUM_THREADS",
    "1",
)

os.environ.setdefault(
    "MKL_NUM_THREADS",
    "1",
)

os.environ.setdefault(
    "OPENBLAS_NUM_THREADS",
    "1",
)


import logging
import threading
from functools import lru_cache
from typing import Iterable, List, Sequence

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


# ============================================================
# LOGGING
# ============================================================

logger = logging.getLogger(__name__)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

EMBEDDING_MODEL = "intfloat/multilingual-e5-small"

EMBEDDING_DIM = 384

DEVICE = "cpu"

# 1 GB RAM server:
# keep this deliberately small.
BATCH_SIZE = 1


# ============================================================
# MODEL LOCK
# ============================================================

_MODEL_LOCK = threading.Lock()


# ============================================================
# PYTORCH THREAD LIMIT
# ============================================================

try:

    torch.set_num_threads(1)

    torch.set_num_interop_threads(1)

    logger.info(
        "EMBEDDING | torch threads configured | "
        "num_threads=%s | interop_threads=%s",
        torch.get_num_threads(),
        torch.get_num_interop_threads(),
    )

except Exception as exc:

    logger.warning(
        "EMBEDDING | unable to configure torch threads | "
        "error=%s",
        exc,
    )


# ============================================================
# MEMORY MONITOR
# ============================================================

def _memory_info() -> str:
    """
    Returns current process RSS memory.

    Works inside Linux containers without psutil.
    """

    try:

        with open(
            "/proc/self/status",
            "r",
            encoding="utf-8",
        ) as file:

            for line in file:

                if line.startswith("VmRSS:"):

                    return line.strip()

    except Exception:

        pass

    return "VmRSS=unknown"

logger.info(
    "EMBEDDING | memory after module imports: %s",
    _memory_info(),
)
# ============================================================
# MODEL DIMENSION
# ============================================================

def _dimension(model) -> int:

    method = getattr(
        model,
        "get_embedding_dimension",
        None,
    )

    if callable(method):

        return int(
            method()
        )

    legacy = getattr(
        model,
        "get_sentence_embedding_dimension",
        None,
    )

    if callable(legacy):

        return int(
            legacy()
        )

    raise RuntimeError(
        "Cannot determine embedding dimension."
    )


# ============================================================
# LOAD MODEL
# ============================================================

@lru_cache(maxsize=1)
def get_model():

    pid = os.getpid()

    logger.info(
        "============================================================"
    )

    logger.info(
        "EMBEDDING | loading model"
    )

    logger.info(
        "EMBEDDING | model=%s",
        EMBEDDING_MODEL,
    )

    logger.info(
        "EMBEDDING | device=%s",
        DEVICE,
    )

    logger.info(
        "EMBEDDING | pid=%s",
        pid,
    )

    logger.info(
        "EMBEDDING | memory before loading: %s",
        _memory_info(),
    )

    try:

        model = SentenceTransformer(
            EMBEDDING_MODEL,
            device=DEVICE,
        )

        dimension = _dimension(
            model
        )

        logger.info(
            "EMBEDDING | model loaded successfully"
        )

        logger.info(
            "EMBEDDING | dimension=%s",
            dimension,
        )

        logger.info(
            "EMBEDDING | memory after loading: %s",
            _memory_info(),
        )

        if dimension != EMBEDDING_DIM:

            raise RuntimeError(
                "Embedding dimension mismatch: "
                f"model reports {dimension}, "
                f"expected {EMBEDDING_DIM}"
            )

        logger.info(
            "EMBEDDING | model validation OK"
        )

        logger.info(
            "============================================================"
        )

        return model

    except Exception:

        logger.exception(
            "EMBEDDING | MODEL LOADING FAILED"
        )

        logger.error(
            "EMBEDDING | memory at failure: %s",
            _memory_info(),
        )

        raise


# ============================================================
# ENCODE
# ============================================================

def _encode(
    texts: Sequence[str],
) -> List[List[float]]:

    clean = [
        str(text).strip()
        for text in texts
        if str(text).strip()
    ]

    if not clean:

        return []

    logger.info(
        "EMBEDDING | encode started | "
        "count=%s | memory=%s",
        len(clean),
        _memory_info(),
    )

    with _MODEL_LOCK:

        model = get_model()

        try:

            vectors = model.encode(
                clean,

                normalize_embeddings=True,

                convert_to_numpy=True,

                show_progress_bar=False,

                batch_size=BATCH_SIZE,
            )

        except Exception:

            logger.exception(
                "EMBEDDING | encode FAILED | "
                "memory=%s",
                _memory_info(),
            )

            raise

    arr = np.asarray(
        vectors,
        dtype=np.float32,
    )

    if arr.ndim == 1:

        arr = arr.reshape(
            1,
            -1,
        )

    if arr.shape[1] != EMBEDDING_DIM:

        raise RuntimeError(
            "Embedding dimension mismatch: "
            f"got {arr.shape[1]}, "
            f"expected {EMBEDDING_DIM} "
            f"for {EMBEDDING_MODEL}."
        )

    logger.info(
        "EMBEDDING | encode completed | "
        "count=%s | dimension=%s | memory=%s",
        len(clean),
        arr.shape[1],
        _memory_info(),
    )

    return arr.tolist()


# ============================================================
# QUERY EMBEDDING
# ============================================================

def get_query_embedding(
    text: str,
) -> List[float]:

    text = str(
        text
    ).strip()

    if not text:

        raise ValueError(
            "Query text is empty."
        )

    return _encode(
        [
            f"query: {text}"
        ]
    )[0]


# ============================================================
# DOCUMENT EMBEDDINGS
# ============================================================

def get_document_embeddings(
    texts: Iterable[str],
) -> List[List[float]]:

    texts = [
        str(text).strip()
        for text in texts
        if str(text).strip()
    ]

    if not texts:

        return []

    return _encode(
        [
            f"passage: {text}"
            for text in texts
        ]
        def warmup_model():
    """
    Предварительная загрузка embedding-модели в память.
    Не изменяет основную логику кодирования.
    """
    logger.info("EMBEDDING | warmup started")
    model = get_model()
    logger.info(
        "EMBEDDING | warmup completed | model=%s | dimension=%s | device=%s",
        EMBEDDING_MODEL,
        EMBEDDING_DIM,
        DEVICE,
    )
    return model
    )
