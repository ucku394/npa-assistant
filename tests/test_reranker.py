"""Unit tests for the Cross-Encoder Reranker."""
import math
from unittest.mock import MagicMock, patch

from rag_engine.reranker import (
    _safe_float,
    _sigmoid,
    format_chunk_for_rerank,
    rerank_chunks,
)


def test_safe_float():
    assert _safe_float("1.25") == 1.25
    assert _safe_float(None) == 0.0
    assert _safe_float("invalid", 42.0) == 42.0


def test_sigmoid():
    assert math.isclose(_sigmoid(0.0), 0.5)
    assert _sigmoid(100.0) > 0.99
    assert _sigmoid(-100.0) < 0.01


def test_format_chunk_for_rerank():
    chunk = {
        "doc_name": "Правила по охране труда № 11",
        "point_num": "120.",
        "content": "В страховочной системе усилие не более 6 кН.",
    }
    formatted = format_chunk_for_rerank(chunk)
    assert "[Правила по охране труда № 11 | п. 120]" in formatted
    assert "В страховочной системе усилие не более 6 кН." in formatted


def test_rerank_empty_chunks():
    assert rerank_chunks("test query", []) == []


def test_rerank_disabled_via_env(monkeypatch):
    monkeypatch.setenv("USE_RERANKER", "false")
    chunks = [
        {"doc_name": "A", "content": "Text A", "_combined_score": 0.8},
        {"doc_name": "B", "content": "Text B", "_combined_score": 0.6},
    ]
    result = rerank_chunks("query", chunks)
    assert result == chunks
    assert "_rerank_score" not in result[0]


def test_rerank_with_model_mock(monkeypatch):
    monkeypatch.setenv("USE_RERANKER", "true")
    mock_model = MagicMock()
    # Mock returns raw cross-encoder logits: chunk B is much more relevant than chunk A
    mock_model.predict.return_value = [-2.0, 3.0]

    with patch("rag_engine.reranker.get_reranker_model", return_value=mock_model):
        chunks = [
            {"doc_name": "A", "content": "Irrelevant text", "_combined_score": 0.9},
            {"doc_name": "B", "content": "Direct regulatory match", "_combined_score": 0.5},
        ]
        result = rerank_chunks("query", chunks)

        assert len(result) == 2
        # Chunk B got high rerank logit (3.0 -> sigmoid ~0.95), so it should rank first
        assert result[0]["doc_name"] == "B"
        assert result[1]["doc_name"] == "A"
        assert "_rerank_score" in result[0]
        assert result[0]["_rerank_score"] > result[1]["_rerank_score"]
