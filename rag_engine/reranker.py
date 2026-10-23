"""Cross-Encoder Reranker with ONNX Runtime acceleration for Belarusian OHS legal assistant.

This module provides two-stage retrieval refinement:
1. First stage (bi-encoder E5 + PGroonga full-text) retrieves 25-30 candidates.
2. Second stage (Cross-Encoder) evaluates (query, document) pairs with deep cross-attention,
   dramatically improving precision for complex regulatory syntax.
"""

import logging
import os
import threading
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default model supporting Russian and multilingual legal texts
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-base"

_RERANKER_LOCK = threading.Lock()
_RERANKER_INSTANCE = None
_RERANKER_INITIALIZED = False


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _sigmoid(x: float) -> float:
    import math
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def get_reranker_model(
    model_name: Optional[str] = None,
    prefer_onnx: bool = True,
):
    """Lazy-load the CrossEncoder model with ONNX or PyTorch backend."""
    global _RERANKER_INSTANCE, _RERANKER_INITIALIZED

    if _RERANKER_INITIALIZED:
        return _RERANKER_INSTANCE

    with _RERANKER_LOCK:
        if _RERANKER_INITIALIZED:
            return _RERANKER_INSTANCE

        model_to_load = model_name or os.getenv("RERANKER_MODEL", DEFAULT_RERANKER_MODEL)
        use_onnx = prefer_onnx and os.getenv("USE_ONNX", "true").lower() in ("true", "1", "yes")

        try:
            from sentence_transformers import CrossEncoder

            if use_onnx:
                try:
                    logger.info("RERANKER | attempting to load CrossEncoder with ONNX backend: %s", model_to_load)
                    model = CrossEncoder(model_to_load, device="cpu", backend="onnx", max_length=512)
                    logger.info("RERANKER | successfully loaded with ONNX runtime")
                    _RERANKER_INSTANCE = model
                    _RERANKER_INITIALIZED = True
                    return _RERANKER_INSTANCE
                except Exception as onnx_exc:
                    logger.warning("RERANKER | ONNX backend failed (%s); falling back to PyTorch", onnx_exc)

            logger.info("RERANKER | loading CrossEncoder with PyTorch backend: %s", model_to_load)
            model = CrossEncoder(model_to_load, device="cpu", backend="torch", max_length=512)
            _RERANKER_INSTANCE = model
            _RERANKER_INITIALIZED = True
            logger.info("RERANKER | successfully loaded with PyTorch")
            return _RERANKER_INSTANCE

        except Exception as exc:
            logger.warning("RERANKER | CrossEncoder unavailable (%s); retrieval will proceed without reranking", exc)
            _RERANKER_INSTANCE = None
            _RERANKER_INITIALIZED = True
            return None


def format_chunk_for_rerank(chunk: Dict[str, Any]) -> str:
    """Format chunk into structured legal context for CrossEncoder."""
    doc = str(chunk.get("doc_name") or chunk.get("document") or "").strip()
    point = str(chunk.get("point_num") or chunk.get("point") or "").strip().rstrip(".")
    content = str(chunk.get("content") or chunk.get("text") or "").strip()

    header = []
    if doc:
        header.append(doc)
    if point:
        header.append(f"п. {point}")

    prefix = f"[{' | '.join(header)}]\n" if header else ""
    return f"{prefix}{content}"


def rerank_chunks(
    query: str,
    chunks: List[Dict[str, Any]],
    top_k: Optional[int] = None,
    weight: float = 0.5,
) -> List[Dict[str, Any]]:
    """Rerank candidate chunks using CrossEncoder scores.

    If reranker is disabled via USE_RERANKER=false or unavailable,
    returns original chunks preserving combined score order.
    """
    if not chunks:
        return []

    use_reranker_env = os.getenv("USE_RERANKER", "true").lower() in ("true", "1", "yes")
    if not use_reranker_env:
        return chunks[:top_k] if top_k else chunks

    model = get_reranker_model()
    if model is None:
        return chunks[:top_k] if top_k else chunks

    try:
        pairs = [[query, format_chunk_for_rerank(chunk)] for chunk in chunks]
        raw_scores = model.predict(pairs)

        for i, chunk in enumerate(chunks):
            raw_score = _safe_float(raw_scores[i] if hasattr(raw_scores, "__len__") else raw_scores)
            norm_score = _sigmoid(raw_score)
            chunk["_rerank_raw_score"] = raw_score
            chunk["_rerank_score"] = norm_score

            base_score = _safe_float(chunk.get("_combined_score", 0.0))
            # Blended score: 50% base retrieval score + 50% fine-grained cross-encoder score
            chunk["_combined_score"] = round(base_score * (1.0 - weight) + norm_score * weight, 4)

        ranked = sorted(
            chunks,
            key=lambda c: _safe_float(c.get("_combined_score", 0.0)),
            reverse=True,
        )
        return ranked[:top_k] if top_k else ranked

    except Exception as exc:
        logger.warning("RERANKER | error during reranking: %s. Using base scores.", exc)
        return chunks[:top_k] if top_k else chunks
