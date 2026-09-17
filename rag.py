import asyncio
import logging
import os
import re
from typing import Any, Optional

from supabase import Client

from config import (
    SUPABASE_MATCH_THRESHOLD,
    SUPABASE_MATCH_COUNT,
)

from embedding import get_query_embedding

logger = logging.getLogger(__name__)


# =========================================================
# НАСТРОЙКИ
# =========================================================

# Сколько документов передавать AI для анализа.
#
# ВАЖНО:
# Это количество найденных кандидатов, а не количество
# источников, которые обязательно должны попасть в ответ.
RAG_FINAL_COUNT = int(
    os.getenv("RAG_FINAL_COUNT", "5")
)


# =========================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =========================================================

def _safe_float(
    value: Any,
) -> Optional[float]:
    """
    Безопасно преобразует similarity/score в float.
    """

    try:

        if value is None or value == "":
            return None

        return float(value)

    except (
        TypeError,
        ValueError,
    ):

        return None


def _semantic_score(
    chunk: dict[str, Any],
) -> float:
    """
    Получает semantic similarity
    из результата Supabase.
    """

    value = _safe_float(
        chunk.get(
            "similarity",
            chunk.get("score"),
        )
    )

    return (
        value
        if value is not None
        else 0.0
    )


def _normalize_identifier(
    value: Any,
) -> str:
    """
    Нормализует значение для формирования SOURCE_ID.
    """

    value = str(
        value or ""
    ).strip()

    if not value:
        return "UNKNOWN"

    # Оставляем буквы, цифры и _
    value = re.sub(
        r"[^A-Za-zА-Яа-яЁё0-9]+",
        "_",
        value,
    )

    value = re.sub(
        r"_+",
        "_",
        value,
    )

    return value.strip("_")


def _get_document_name(
    chunk: dict[str, Any],
) -> str:
    """
    Возвращает название НПА.
    """

    return str(
        chunk.get("doc_name")
        or "НПА"
    ).strip()


def _get_point_number(
    chunk: dict[str, Any],
) -> str:
    """
    Возвращает номер пункта/статьи/раздела.
    """

    value = (
        chunk.get("point_num")
        or chunk.get("article")
        or chunk.get("section")
        or ""
    )

    return str(
        value
    ).strip()


def _build_source_id(
    chunk: dict[str, Any],
    index: int,
) -> str:
    """
    Формирует стабильный идентификатор источника.

    Пример:

    NPA_175_P51

    или:

    NPA_ЗООТ_P123
    """

    # Если SOURCE_ID уже существует в БД —
    # используем его.
    existing_id = (
        chunk.get("source_id")
        or chunk.get("source")
        or chunk.get("chunk_id")
    )

    if existing_id:

        return _normalize_identifier(
            existing_id
        )

    doc_name = _get_document_name(
        chunk
    )

    point_num = _get_point_number(
        chunk
    )

    doc_clean = _normalize_identifier(
        doc_name
    )

    point_clean = _normalize_identifier(
        point_num
    )

    if point_clean:

        return (
            f"NPA_{doc_clean}_P{point_clean}"
        )

    return (
        f"NPA_{doc_clean}_CHUNK{index}"
    )


# =========================================================
# ФОРМАТИРОВАНИЕ ФРАГМЕНТА
# =========================================================

def _format_chunk(
    chunk: dict[str, Any],
    index: int,
) -> str:
    """
    Приводит один найденный фрагмент НПА
    к юридически однозначному формату.
    """

    doc_name = _get_document_name(
        chunk
    )

    point_num = _get_point_number(
        chunk
    )

    content = str(
        chunk.get("content") or ""
    ).strip()

    source_id = _build_source_id(
        chunk,
        index,
    )

    # Сохраняем SOURCE_ID внутри chunk,
    # чтобы его можно было использовать дальше.
    chunk["_source_id"] = source_id

    result = (
        f"SOURCE_ID: {source_id}\n"
        f"Документ: {doc_name}\n"
    )

    if point_num:

        result += (
            f"Пункт/статья: "
            f"{point_num}\n"
        )

    result += (
        f"Текст НПА:\n"
        f"{content}"
    )

    return result


# =========================================================
# SUPABASE SEARCH
# =========================================================

def _search_chunks(
    supabase: Client,
    query_vector: list[float],
) -> list[dict[str, Any]]:
    """
    Выполняет векторный поиск в Supabase.
    """

    response = supabase.rpc(
        "match_npa_chunks",
        {
            "query_embedding": query_vector,
            "match_threshold": (
                SUPABASE_MATCH_THRESHOLD
            ),
            "match_count": (
                SUPABASE_MATCH_COUNT
            ),
        },
    ).execute()

    return response.data or []


# =========================================================
# СОРТИРОВКА
# =========================================================

def _sort_by_semantic_similarity(
    chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Сортирует найденные фрагменты
    по semantic similarity.
    """

    result = []

    for index, chunk in enumerate(
        chunks
    ):

        item = dict(chunk)

        item["_semantic_score"] = (
            _semantic_score(chunk)
        )

        item["_original_index"] = (
            index
        )

        result.append(item)

    result.sort(
        key=lambda item: (
            item["_semantic_score"],
            -item["_original_index"],
        ),
        reverse=True,
    )

    return result


# =========================================================
# ФОРМИРОВАНИЕ КОНТЕКСТА
# =========================================================

def _build_retrieved_text(
    chunks: list[dict[str, Any]],
) -> str:
    """
    Формирует юридический контекст
    для передачи AI.

    ВАЖНО:

    Найденный источник ≠ источник,
    который обязательно должен быть
    указан в финальном ответе.
    """

    blocks = []

    for index, chunk in enumerate(
        chunks,
        start=1,
    ):

        chunk_text = _format_chunk(
            chunk,
            index,
        )

        semantic = chunk.get(
            "_semantic_score"
        )

        source_id = chunk.get(
            "_source_id"
        )

        metadata = []

        if source_id:

            metadata.append(
                f"SOURCE_ID: {source_id}"
            )

        if semantic is not None:

            metadata.append(
                "Semantic similarity: "
                f"{semantic:.4f}"
            )

        metadata_text = ""

        if metadata:

            metadata_text = (
                "\n"
                + "\n".join(
                    metadata
                )
            )

        blocks.append(
            f"===== "
            f"RAG SOURCE {index} "
            f"=====\n"
            f"{chunk_text}"
            f"{metadata_text}"
        )

    return "\n\n".join(
        blocks
    )


# =========================================================
# ОСНОВНОЙ RAG
# =========================================================

async def retrieve_context(
    user_query: str,
    supabase: Client,
) -> dict[str, Any]:
    """
    Основная функция RAG.

    Алгоритм:

    1. Получаем embedding вопроса.
    2. Ищем похожие фрагменты НПА.
    3. Сортируем по similarity.
    4. Берём TOP-N.
    5. Присваиваем каждому SOURCE_ID.
    6. Формируем юридически структурированный
       контекст для AI.

    Gemini здесь НЕ используется.
    """

    user_query = (
        user_query or ""
    ).strip()

    if not user_query:

        return {
            "chunks": [],
            "retrieved_text": "",
            "found": False,
            "candidate_count": 0,
            "final_count": 0,
            "source_references": [],
        }

    logger.info(
        "RAG | query=%s",
        user_query,
    )

    # =====================================================
    # ШАГ 1. EMBEDDING
    # =====================================================

    query_vector = await asyncio.to_thread(
        get_query_embedding,
        user_query,
    )

    logger.info(
        "RAG | embedding dimension=%s",
        len(query_vector),
    )

    if len(query_vector) != 384:

        raise RuntimeError(
            "Unexpected embedding dimension: "
            f"{len(query_vector)}; "
            "expected 384."
        )

    # =====================================================
    # ШАГ 2. ПОИСК
    # =====================================================

    candidate_chunks = (
        await asyncio.to_thread(
            _search_chunks,
            supabase,
            query_vector,
        )
    )

    logger.info(
        "RAG | candidates=%s",
        len(candidate_chunks),
    )

    if not candidate_chunks:

        logger.info(
            "RAG | no relevant chunks found"
        )

        return {
            "chunks": [],
            "retrieved_text": "",
            "found": False,
            "candidate_count": 0,
            "final_count": 0,
            "source_references": [],
        }

    # =====================================================
    # ШАГ 3. СОРТИРОВКА
    # =====================================================

    ranked_chunks = (
        _sort_by_semantic_similarity(
            candidate_chunks
        )
    )

    # =====================================================
    # ШАГ 4. TOP-N
    # =====================================================

    final_chunks = ranked_chunks[
        :RAG_FINAL_COUNT
    ]

    logger.info(
        "RAG | final chunks=%s",
        len(final_chunks),
    )

    # =====================================================
    # ШАГ 5. SOURCE ID
    # =====================================================

    for index, chunk in enumerate(
        final_chunks,
        start=1,
    ):

        source_id = _build_source_id(
            chunk,
            index,
        )

        chunk["_source_id"] = (
            source_id
        )

    # =====================================================
    # ЛОГИРОВАНИЕ
    # =====================================================

    for index, chunk in enumerate(
        final_chunks,
        start=1,
    ):

        doc_name = _get_document_name(
            chunk
        )

        point_num = _get_point_number(
            chunk
        )

        similarity = chunk.get(
            "_semantic_score",
            0.0,
        )

        source_id = chunk.get(
            "_source_id",
            "UNKNOWN",
        )

        logger.info(
            "RAG | TOP %s | "
            "source_id=%s | "
            "similarity=%.4f | "
            "document=%s | "
            "point=%s",
            index,
            source_id,
            similarity,
            doc_name,
            point_num,
        )

    # =====================================================
    # ШАГ 6. КОНТЕКСТ
    # =====================================================

    retrieved_text = (
        _build_retrieved_text(
            final_chunks
        )
    )

    # =====================================================
    # ШАГ 7. ССЫЛКИ НА ВСЕ НАЙДЕННЫЕ НОРМЫ
    #
    # ВАЖНО:
    # Это именно RAG candidates.
    # Они НЕ означают, что AI обязан
    # сослаться на все эти нормы.
    # =====================================================

    source_references = (
        get_source_references(
            final_chunks
        )
    )

    return {
        "chunks": final_chunks,
        "retrieved_text": retrieved_text,
        "found": bool(final_chunks),
        "candidate_count": len(
            candidate_chunks
        ),
        "final_count": len(
            final_chunks
        ),
        "source_references": (
            source_references
        ),
    }


# =========================================================
# ИСТОЧНИКИ
# =========================================================

def get_source_references(
    chunks: list[dict[str, Any]],
) -> list[str]:
    """
    Возвращает уникальные ссылки
    на найденные фрагменты.

    Формат:

    Документ — пункт/статья
    """

    result = []

    seen = set()

    for chunk in chunks:

        doc_name = _get_document_name(
            chunk
        )

        point_num = _get_point_number(
            chunk
        )

        key = (
            doc_name,
            point_num,
        )

        if key in seen:

            continue

        seen.add(key)

        if point_num:

            result.append(
                f"{doc_name} — "
                f"{point_num}"
            )

        else:

            result.append(
                doc_name
            )

    return result


def get_source_names(
    chunks: list[dict[str, Any]],
) -> list[str]:
    """
    Возвращает уникальные названия НПА.
    """

    result = []

    seen = set()

    for chunk in chunks:

        name = _get_document_name(
            chunk
        )

        if (
            name
            and name not in seen
        ):

            seen.add(name)

            result.append(
                name
            )

    return result


def get_source_ids(
    chunks: list[dict[str, Any]],
) -> list[str]:
    """
    Возвращает SOURCE_ID найденных
    фрагментов.

    Например:

    [
        "NPA_175_P51",
        "NPA_175_P53",
        "NPA_175_P47"
    ]
    """

    result = []

    seen = set()

    for index, chunk in enumerate(
        chunks,
        start=1,
    ):

        source_id = (
            chunk.get("_source_id")
            or _build_source_id(
                chunk,
                index,
            )
        )

        if source_id in seen:

            continue

        seen.add(
            source_id
        )

        result.append(
            source_id
        )

    return result
