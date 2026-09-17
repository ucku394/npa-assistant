import os

# ============================================================
# ENVIRONMENT OPTIMIZATION
# IMPORTANT:
# These variables must be set BEFORE importing torch.
# ============================================================

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

# Reduce unnecessary Hugging Face background activity
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


import gc
import logging
import threading
from functools import lru_cache
from typing import Iterable, List, Sequence

import numpy as np
import torch

logger = logging.getLogger(__name__)


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


def _log_memory(label: str) -> None:
    logger.info(
        "MEMORY | %s | %s",
        label,
        _memory_info(),
    )


# ============================================================
# TORCH
# ============================================================

_log_memory("before torch configuration")

try:

    torch.set_num_threads(1)

    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

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

_log_memory("after torch import")


# ============================================================
# SENTENCE TRANSFORMERS
# ============================================================

from sentence_transformers import SentenceTransformer

_log_memory("after sentence_transformers import")


# ============================================================
# MODEL CONFIGURATION
# ============================================================

EMBEDDING_MODEL = "intfloat/multilingual-e5-small"

# multilingual-e5-small = 384 dimensions
EMBEDDING_DIM = 384

DEVICE = "cpu"

# One query at a time.
BATCH_SIZE = 1


# ============================================================
# MODEL LOCK
# ============================================================

_MODEL_LOCK = threading.Lock()


# ============================================================
# MODULE STATUS
# ============================================================

logger.info(
    "============================================================"
)

logger.info(
    "EMBEDDING | module initialized"
)

logger.info(
    "EMBEDDING | model=%s",
    EMBEDDING_MODEL,
)

logger.info(
    "EMBEDDING | dimension=%s",
    EMBEDDING_DIM,
)

logger.info(
    "EMBEDDING | device=%s",
    DEVICE,
)

logger.info(
    "EMBEDDING | batch_size=%s",
    BATCH_SIZE,
)

_log_memory(
    "after embedding module initialization"
)

logger.info(
    "EMBEDDING | lazy loading ENABLED"
)

logger.info(
    "EMBEDDING | model will NOT be loaded during import"
)

logger.info(
    "============================================================"
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
    """
    Lazy-load the embedding model.

    The model is loaded only on the first embedding request.

    After successful loading the same model instance is reused.
    """

    pid = os.getpid()

    logger.info(
        "============================================================"
    )

    logger.info(
        "EMBEDDING | lazy loading model"
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

    _log_memory(
        "before model loading"
    )

    try:

        logger.info(
            "EMBEDDING | creating SentenceTransformer..."
        )

        model = SentenceTransformer(
            EMBEDDING_MODEL,
            device=DEVICE,
        )

        _log_memory(
            "after SentenceTransformer creation"
        )

        # ----------------------------------------------------
        # DIMENSION CHECK
        # ----------------------------------------------------

        dimension = _dimension(
            model
        )

        logger.info(
            "EMBEDDING | detected dimension=%s",
            dimension,
        )

        if dimension != EMBEDDING_DIM:

            raise RuntimeError(
                "Embedding dimension mismatch: "
                f"model reports {dimension}, "
                f"expected {EMBEDDING_DIM}"
            )

        # ----------------------------------------------------
        # EVAL MODE
        # ----------------------------------------------------

        try:

            model.eval()

        except Exception as exc:

            logger.warning(
                "EMBEDDING | model.eval() failed | "
                "error=%s",
                exc,
            )

        _log_memory(
            "model ready"
        )

        logger.info(
            "EMBEDDING | model loaded successfully"
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

        _log_memory(
            "memory at model loading failure"
        )

        gc.collect()

        raise


# ============================================================
# ENCODE
# ============================================================

def _encode(
    texts: Sequence[str],
) -> List[List[float]]:
    """
    Encode texts using multilingual-e5-small.
    """

    clean = [
        str(text).strip()
        for text in texts
        if str(text).strip()
    ]

    if not clean:
        return []

    logger.info(
        "============================================================"
    )

    logger.info(
        "EMBEDDING | encode started | "
        "count=%s | memory=%s",
        len(clean),
        _memory_info(),
    )

    # --------------------------------------------------------
    # MODEL ACCESS
    # --------------------------------------------------------

    with _MODEL_LOCK:

        model = get_model()

        _log_memory(
            "before encode"
        )

        try:

            # ------------------------------------------------
            # INFERENCE MODE
            # ------------------------------------------------

            with torch.inference_mode():

                vectors = model.encode(
                    clean,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    batch_size=BATCH_SIZE,
                    convert_to_tensor=False,
                )

        except Exception:

            logger.exception(
                "EMBEDDING | encode FAILED | "
                "memory=%s",
                _memory_info(),
            )

            raise

        finally:

            gc.collect()

        _log_memory(
            "after encode"
        )

    # ========================================================
    # NUMPY
    # ========================================================

    arr = np.asarray(
        vectors,
        dtype=np.float32,
    )

    if arr.ndim == 1:

        arr = arr.reshape(
            1,
            -1,
        )

    # ========================================================
    # DIMENSION VALIDATION
    # ========================================================

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

    logger.info(
        "============================================================"
    )

    return arr.tolist()


# ============================================================
# QUERY EMBEDDING
# ============================================================

def get_query_embedding(
    text: str,
) -> List[float]:
    """
    Create embedding for a user query.

    E5 format:

        query: <text>
    """

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
    """
    Create embeddings for document chunks.

    E5 format:

        passage: <text>
    """

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
    )


# ============================================================
# OPTIONAL WARMUP
# ============================================================

def warmup_model():
    """
    Explicitly load the model.

    Do NOT call this during normal Railway startup.
    """

    logger.info(
        "EMBEDDING | explicit warmup requested"
    )

    _log_memory(
        "before warmup"
    )

    model = get_model()

    _log_memory(
        "after warmup"
    )

    return model


# ============================================================
# MODEL STATUS
# ============================================================

def is_model_loaded() -> bool:
    """
    Returns True if the model is already cached.
    """

    try:

        return (
            get_model.cache_info().currsize > 0
        )

    except Exception:

        return False


# ============================================================
# CLEAR MODEL CACHE
# ============================================================

def clear_model_cache() -> None:
    """
    Explicitly unload the cached model.
    """

    logger.info(
        "EMBEDDING | clearing model cache"
    )

    try:

        get_model.cache_clear()

    except Exception as exc:

        logger.warning(
            "EMBEDDING | cache clear failed | "
            "error=%s",
            exc,
        )

    gc.collect()

    _log_memory(
        "after model cache clear"
    )
