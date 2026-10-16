"""Tests for the local query embedding pipeline."""
from embedding import EMBEDDING_DIM, get_query_embedding


def test_get_query_embedding_dimension():
    text = "Какие требования предъявляются к работам на высоте?"
    vector = get_query_embedding(text)

    assert isinstance(vector, list)
    assert len(vector) == EMBEDDING_DIM
    assert all(isinstance(x, float) for x in vector[:5])
