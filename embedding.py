import logging
from functools import lru_cache

from sentence_transformers import SentenceTransformer


logger = logging.getLogger(__name__)

# Мультиязычная модель.
# Размер вектора = 768.
MODEL_NAME = "intfloat/multilingual-e5-base"

# Размерность должна соответствовать:
# Supabase: embedding vector(768)
EMBEDDING_DIMENSION = 768


@lru_cache(maxsize=1)
def get_model():
    """
    Загружает embedding-модель один раз
    и повторно использует её в процессе работы бота.
    """

    logger.info("Загрузка embedding-модели: %s", MODEL_NAME)

    model = SentenceTransformer(MODEL_NAME)

    logger.info(
        "Embedding-модель загружена. Размерность: %s",
        model.get_sentence_embedding_dimension()
    )

    return model


def get_query_embedding(text: str) -> list[float]:
    """
    Создает embedding пользовательского запроса.

    Для E5-моделей поисковый запрос должен начинаться
    с префикса 'query:'.
    """

    if not text or not text.strip():
        raise ValueError("Нельзя создать embedding для пустого текста.")

    model = get_model()

    prepared_text = f"query: {text.strip()}"

    vector = model.encode(
        prepared_text,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    result = vector.tolist()

    if len(result) != EMBEDDING_DIMENSION:
        raise ValueError(
            f"Неверная размерность embedding: {len(result)}. "
            f"Ожидалось: {EMBEDDING_DIMENSION}"
        )

    return result


def get_document_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Создает embeddings для массива документов/чанков.

    Для E5-модели используется префикс 'passage:'.
    """

    if not texts:
        return []

    model = get_model()

    prepared_texts = [
        f"passage: {text.strip()}"
        for text in texts
        if text and text.strip()
    ]

    vectors = model.encode(
        prepared_texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    result = [vector.tolist() for vector in vectors]

    for vector in result:
        if len(vector) != EMBEDDING_DIMENSION:
            raise ValueError(
                f"Неверная размерность embedding: {len(vector)}. "
                f"Ожидалось: {EMBEDDING_DIMENSION}"
            )

    return result
