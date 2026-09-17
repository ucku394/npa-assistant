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

# Prevent Hugging Face from spawning unnecessary parallel work
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


import gc
import logging
import os as _os
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

# multilingual-e5-small = 384 dimensions
EMBEDDING_DIM = 384

DEVICE = "cpu"

# Keep batches deliberately small for Railway RAM limits.
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

    # This can only be changed before parallel work starts.
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
        "EMBEDDING | unable to configure torch threads | error=%s",
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


def _log_memory(label: str) -> None:
    """
    Log current process memory with a readable label.
    """

    logger.info(
        "MEMORY | %s | %s",
        label,
        _memory_info(),
    )


# ============================================================
# STARTUP MEMORY
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

_log_memory(
    "after module imports"
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

    IMPORTANT:
    The model is NOT loaded when this module is imported.

    It is loaded only when get_model() is actually called.
    After the first successful load it remains cached.
    """

    pid = _os.getpid()

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

        # ----------------------------------------------------
        # DIMENSION VALIDATION
        # ----------------------------------------------------

        if dimension != EMBEDDING_DIM:

            raise RuntimeError(
                "Embedding dimension mismatch: "
                f"model reports {dimension}, "
                f"expected {EMBEDDING_DIM}"
            )

        # ----------------------------------------------------
        # CPU / EVAL MODE
        # ----------------------------------------------------

        try:
            model.eval()
        except Exception as exc:
            logger.warning(
                "EMBEDDING | model.eval() failed | error=%s",
                exc,
            )

        logger.info(
            "EMBEDDING | model validation OK"
        )

        _log_memory(
            "model ready"
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

        # Try to release temporary objects.
        try:
            gc.collect()
        except Exception:
            pass

        raise


# ============================================================
# ENCODE
# ============================================================

def _encode(
    texts: Sequence[str],
) -> List[List[float]]:
    """
    Encode texts using multilingual-e5-small.

    The model is loaded lazily on first call.
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
            # IMPORTANT:
            # inference_mode reduces unnecessary autograd
            # overhead and memory usage.
            # ------------------------------------------------

            with torch.inference_mode():

                vectors = model.encode(
                    clean,

                    normalize_embeddings=True,

                    convert_to_numpy=True,

                    show_progress_bar=False,

                    batch_size=BATCH_SIZE,

                    # Do not create a tensor graph.
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

            # Release temporary Python objects when possible.
            gc.collect()

        _log_memory(
            "after encode"
        )

    # ========================================================
    # NUMPY CONVERSION
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

    E5 models expect:
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

    E5 models expect:
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
    Explicitly load the embedding model.

    IMPORTANT:
    This function must NOT be called during normal Railway
    startup if you want lazy loading.

    It is kept for compatibility and manual diagnostics.
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

    logger.info(
        "EMBEDDING | warmup completed | "
        "model=%s | dimension=%s | device=%s",
        EMBEDDING_MODEL,
        EMBEDDING_DIM,
        DEVICE,
    )

    return model


# ============================================================
# MODEL STATUS
# ============================================================

def is_model_loaded() -> bool:
    """
    Returns True if the cached model already exists.
    """

    try:

        return get_model.cache_info().currsize > 0

    except Exception:

        return False


def clear_model_cache() -> None:
    """
    Explicitly unload the embedding model.

    Useful for diagnostics or memory recovery.
    """

    logger.info(
        "EMBEDDING | clearing model cache"
    )

    try:

        get_model.cache_clear()

    except Exception as exc:

        logger.warning(
            "EMBEDDING | cache clear failed | error=%s",
            exc,
        )

    gc.collect()

    _log_memory(
        "after model cache clear"
    )
