"""
NPA Loader
=========

Полная загрузка НПА в Supabase Vector Store.

Embedding model:
    intfloat/multilingual-e5-small

Embedding dimension:
    384

ВАЖНО:
    Для документов используется:
        passage:<text>

    Для пользовательских запросов в bot.py используется:
        query:<text>

Оба направления должны использовать одну и ту же модель.

Первое полное переиндексирование:
    REINDEX_ALL=true

После успешной загрузки:
    REINDEX_ALL=false

Никакого Gemini Embedding API здесь НЕ используется.
"""

import hashlib
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import docx
from dotenv import load_dotenv
from supabase import Client, create_client

from embedding import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    get_document_embeddings,
    warmup_model,
)


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# PATHS / ENV
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
ENV_PATH = SCRIPT_DIR / ".env"

load_dotenv(dotenv_path=ENV_PATH)


SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_SERVICE_ROLE_KEY = os.getenv(
    "SUPABASE_SERVICE_ROLE_KEY",
    "",
).strip()


# ============================================================
# CONFIG
# ============================================================

# Первый запуск после миграции на e5-small:
#
#     REINDEX_ALL=true
#
# После успешного полного запуска:
#
#     REINDEX_ALL=false
#
REINDEX_ALL = (
    os.getenv("REINDEX_ALL", "false")
    .strip()
    .lower()
    in {"1", "true", "yes", "y", "on"}
)


# Размер пачки embeddings.
#
# Для e5-small и компьютеров с ограниченной RAM
# 8 является безопасным значением.
EMBEDDING_BATCH_SIZE = int(
    os.getenv("EMBEDDING_BATCH_SIZE", "8")
)


# Размер пачки при вставке в Supabase.
UPLOAD_BATCH_SIZE = int(
    os.getenv("UPLOAD_BATCH_SIZE", "8")
)


# Количество повторных попыток Supabase при временной ошибке.
SUPABASE_RETRIES = int(
    os.getenv("SUPABASE_RETRIES", "3")
)


# Пауза между повторными попытками.
SUPABASE_RETRY_DELAY = float(
    os.getenv("SUPABASE_RETRY_DELAY", "2")
)


# ============================================================
# VALIDATION
# ============================================================

def validate_environment() -> None:
    """
    Проверка обязательных переменных окружения.
    """

    missing = []

    if not SUPABASE_URL:
        missing.append("SUPABASE_URL")

    if not SUPABASE_SERVICE_ROLE_KEY:
        missing.append("SUPABASE_SERVICE_ROLE_KEY")

    if missing:
        raise RuntimeError(
            "Не найдены обязательные переменные окружения: "
            + ", ".join(missing)
            + f"\nПроверь файл: {ENV_PATH}"
        )

    if EMBEDDING_DIM != 384:
        raise RuntimeError(
            "Неверная размерность embedding: "
            f"{EMBEDDING_DIM}. Ожидается 384."
        )

    if EMBEDDING_MODEL != "intfloat/multilingual-e5-small":
        raise RuntimeError(
            "Используется неправильная embedding model: "
            f"{EMBEDDING_MODEL}. "
            "Ожидается intfloat/multilingual-e5-small."
        )

    if EMBEDDING_BATCH_SIZE < 1:
        raise RuntimeError(
            "EMBEDDING_BATCH_SIZE должен быть >= 1"
        )

    if UPLOAD_BATCH_SIZE < 1:
        raise RuntimeError(
            "UPLOAD_BATCH_SIZE должен быть >= 1"
        )


# ============================================================
# SUPABASE
# ============================================================

supabase: Client | None = None


def get_supabase() -> Client:
    """
    Создаёт Supabase client один раз.
    """

    global supabase

    if supabase is None:
        supabase = create_client(
            SUPABASE_URL,
            SUPABASE_SERVICE_ROLE_KEY,
        )

    return supabase


# ============================================================
# DOCUMENT NAME MAP
# ============================================================

"""
Ключ:
    имя файла без расширения.

Значение:
    каноническое имя документа, которое хранится в Supabase.

ВАЖНО:
    lookup выполняется после .strip(),
    поэтому случайный пробел перед .docx больше
    не создаёт ошибку.
"""


DOC_NAME_MAP: Dict[str, str] = {

    # --------------------------------------------------------
    # 1
    # --------------------------------------------------------

    "Закон об охране труда от 23 июня 2008 г. № 356-З":
        "Закон об охране труда от 23 июня 2008 г. № 356-З",

    # --------------------------------------------------------
    # 2
    # --------------------------------------------------------

    "Кодекс Республики Беларусь об административных правонарушениях от 6 января 2021 г. № 91-З":
        "Кодекс Республики Беларусь об административных правонарушениях от 6 января 2021 г. № 91-З",

    # --------------------------------------------------------
    # 3
    # --------------------------------------------------------

    "О бесплатном обеспечении работников молоком или равноценными пищевыми продуктами при работе с вредными веществами 27 февраля 2002 г. № 260":
        "О бесплатном обеспечении работников молоком или равноценными пищевыми продуктами при работе с вредными веществами 27 февраля 2002 г. № 260",

    # --------------------------------------------------------
    # 4
    # --------------------------------------------------------

    "О документах, необходимых для расследования и учета несчастных случаев на производстве и профессиональных заболеваний Минтруда и Соцзащиты от 4 октября 2024 г. № 81 144":
        "О документах, необходимых для расследования и учета несчастных случаев на производстве и профессиональных заболеваний Минтруда и Соцзащиты от 4 октября 2024 г. № 81 144",

    # --------------------------------------------------------
    # 5
    # --------------------------------------------------------

    "О контроле состояния водителей от 9 июля 2013 г. № 25.28":
        "О контроле состояния водителей от 9 июля 2013 г. № 25.28",

    # --------------------------------------------------------
    # 6
    # --------------------------------------------------------

    "О мерах по укреплению общественной безопасности и дисциплины Директива от 11 марта 2004 г. № 1":
        "О мерах по укреплению общественной безопасности и дисциплины Директива от 11 марта 2004 г. № 1",

    # --------------------------------------------------------
    # 7
    # --------------------------------------------------------

    "О пожарной безопасности Закон РБ от 15 июня 1993 г. № 2403-XII":
        "О пожарной безопасности Закон РБ от 15 июня 1993 г. № 2403-XII",

    # --------------------------------------------------------
    # 8
    # --------------------------------------------------------

    "О порядке обучения, стажировки, инструктажа и проверки знаний работающих по вопросам охраны труда № 175 от 28 ноября 2008 г":
        "О порядке обучения, стажировки, инструктажа и проверки знаний работающих по вопросам охраны труда № 175 от 28 ноября 2008 г.",

    # --------------------------------------------------------
    # 9
    # --------------------------------------------------------

    "О порядке проведения предрейсовых и иных медицинских обследований водителей механических транспортных средств (за исключением колесных тракторов) от 3 декабря 2002 г. № 84":
        "О порядке проведения предрейсовых и иных медицинских обследований водителей механических транспортных средств (за исключением колесных тракторов) от 3 декабря 2002 г. № 84",

    # --------------------------------------------------------
    # 10
    # --------------------------------------------------------

    "О порядке разработки и принятия локальных правовых актов по охране труда от 28 ноября 2008 г. № 176":
        "О порядке разработки и принятия локальных правовых актов по охране труда от 28 ноября 2008 г. № 176",

    # --------------------------------------------------------
    # 11
    # --------------------------------------------------------

    "О порядке расследования и учета несчастных случаев МЧС от 6 января 2023 г. № 6":
        "О порядке расследования и учета несчастных случаев МЧС от 6 января 2023 г. № 6",

    # --------------------------------------------------------
    # 12
    # --------------------------------------------------------

    "О проведении обязательных и внеочередных медицинских осмотров работающих от 29 июля 2019 г. № 74":
        "О проведении обязательных и внеочередных медицинских осмотров работающих от 29 июля 2019 г. № 74",

    # --------------------------------------------------------
    # 13
    # --------------------------------------------------------

    "О расследовании и учете несчастных случаев на производстве и профессиональных заболеваний от 15 января 2004 г. № 30":
        "О расследовании и учете несчастных случаев на производстве и профессиональных заболеваний от 15 января 2004 г. № 30",

    # --------------------------------------------------------
    # 14
    # --------------------------------------------------------

    "Об обеспечении пожарной безопасности постановление МЧС 21 декабря 2021 г. № 82":
        "Об обеспечении пожарной безопасности постановление МЧС 21 декабря 2021 г. № 82",

    # --------------------------------------------------------
    # 15
    # --------------------------------------------------------

    "Об утверждении Инструкции о порядке осуществления контроля за соблюдением работниками требований по охране труда от 15 мая 2020 г. № 51":
        "Об утверждении Инструкции о порядке осуществления контроля за соблюдением работниками требований по охране труда от 15 мая 2020 г. № 51",

    # --------------------------------------------------------
    # 16
    # --------------------------------------------------------

    "Об утверждении Межотраслевых правил по охране труда при проведении погрузочно-разгрузочных работ от 26 января 2018 г. № 12":
        "Об утверждении Межотраслевых правил по охране труда при проведении погрузочно-разгрузочных работ от 26 января 2018 г. № 12",

    # --------------------------------------------------------
    # 17
    # --------------------------------------------------------

    "Об утверждении Межотраслевых правил по охране труда при эксплуатации напольного безрельсового транспорта и грузовых тележек от 30 декабря 2003 г. № 165":
        "Об утверждении Межотраслевых правил по охране труда при эксплуатации напольного безрельсового транспорта и грузовых тележек от 30 декабря 2003 г. № 165",

    # --------------------------------------------------------
    # 18
    # --------------------------------------------------------

    "Об утверждении Правил по охране труда при выполнении работ на высоте от 6 февраля 2025 г. № 11":
        "Об утверждении Правил по охране труда при выполнении работ на высоте от 6 февраля 2025 г. № 11",

    # --------------------------------------------------------
    # 19
    # --------------------------------------------------------

    "Об утверждении Правил по охране труда при выполнении строительных работ от 31 мая 2019 г. № 24 33":
        "Об утверждении Правил по охране труда при выполнении строительных работ от 31 мая 2019 г. № 24 33",

    # --------------------------------------------------------
    # 20
    # --------------------------------------------------------

    "Об утверждении Правил по охране труда при производстве пищевой продукции от 31 декабря 2024 г. № 122":
        "Об утверждении Правил по охране труда при производстве пищевой продукции от 31 декабря 2024 г. № 122",

    # --------------------------------------------------------
    # 21
    # --------------------------------------------------------

    "Об утверждении Правил по охране труда при эксплуатации автомобильного и городского электрического транспорта от 6 декабря 2022 г. № 78 104":
        "Об утверждении Правил по охране труда при эксплуатации автомобильного и городского электрического транспорта от 6 декабря 2022 г. № 78 104",

    # --------------------------------------------------------
    # 22
    # --------------------------------------------------------

    "Об утверждении специфических санитарно-эпидемиологических требований от 24 января 2020 г. № 42":
        "Об утверждении специфических санитарно-эпидемиологических требований от 24 января 2020 г. № 42",

    # --------------------------------------------------------
    # 23
    # --------------------------------------------------------

    "Об утверждении специфических санитарно-эпидемиологических требований Постановление от 1 февраля 2020 г. № 66":
        "Об утверждении специфических санитарно-эпидемиологических требований Постановление от 1 февраля 2020 г. № 66",

    # --------------------------------------------------------
    # 24
    # --------------------------------------------------------

    "Об утверждении специфических требований по обеспечению пожарной безопасности взрывопожароопасных и пожароопасных производств 20 ноября 2019 г. № 779":
        "Об утверждении специфических требований по обеспечению пожарной безопасности взрывопожароопасных и пожароопасных производств 20 ноября 2019 г. № 779",

    # --------------------------------------------------------
    # 25
    # --------------------------------------------------------

    "Правила по обеспечению СИЗ №209":
        "Правила по обеспечению СИЗ №209",

    # --------------------------------------------------------
    # 26
    # --------------------------------------------------------

    "Правила по охране труда № 53":
        "Правила по охране труда № 53",

    # --------------------------------------------------------
    # 27
    # --------------------------------------------------------

    "Трудовой кодекс Республики Беларусь 2026":
        "Трудовой кодекс Республики Беларусь 2026",
}


# ============================================================
# HELPERS
# ============================================================

def sha256_text(text: str) -> str:
    """
    SHA-256 хэш текста чанка.
    """

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def normalize_filename_stem(file_path: Path) -> str:
    """
    Нормализация имени файла.

    Особенно важно для файла:

        ... № 11 .docx

    где перед .docx находится лишний пробел.

    Path.stem даст:
        ... № 11

    или может сохранить конечный пробел в зависимости
    от имени файла/ОС.

    Поэтому используем strip().
    """

    return file_path.stem.strip()


def resolve_doc_name(file_path: Path) -> str:
    """
    Находит каноническое название документа.
    """

    file_stem = normalize_filename_stem(file_path)

    if file_stem in DOC_NAME_MAP:
        return DOC_NAME_MAP[file_stem].strip()

    # Дополнительная защита:
    # сравнение нормализованных ключей.
    normalized_stem = re.sub(
        r"\s+",
        " ",
        file_stem,
    ).strip()

    for key, value in DOC_NAME_MAP.items():

        normalized_key = re.sub(
            r"\s+",
            " ",
            key.strip(),
        ).strip()

        if normalized_key == normalized_stem:
            return value.strip()

    raise ValueError(
        f"Для файла '{file_path.name}' нет записи в DOC_NAME_MAP.\n"
        f"Нормализованное имя: '{file_stem}'"
    )


# ============================================================
# FILE READING
# ============================================================

def read_docx(file_path: Path) -> str:
    """
    Чтение текста из DOCX.

    Берём только непустые параграфы.
    """

    document = docx.Document(file_path)

    paragraphs: List[str] = []

    for paragraph in document.paragraphs:

        text = paragraph.text.strip()

        if text:
            paragraphs.append(text)

    return "\n".join(paragraphs)


def read_txt(file_path: Path) -> str:
    """
    Чтение TXT UTF-8.
    """

    with open(
        file_path,
        "r",
        encoding="utf-8",
    ) as file:

        return file.read()


def read_source(file_path: Path) -> str:
    """
    Универсальное чтение исходного документа.
    """

    extension = file_path.suffix.lower()

    if extension == ".docx":
        return read_docx(file_path)

    if extension == ".txt":
        return read_txt(file_path)

    raise ValueError(
        f"Неподдерживаемое расширение: "
        f"{file_path.suffix}"
    )


# ============================================================
# TEXT CHUNKING
# ============================================================

def split_text_into_chunks(
    text: str,
    doc_name: str,
) -> List[dict]:
    """
    Разбиение НПА на чанки.

    Основные границы:

        Статья 1
        Статья 2

        Пункт 1
        Пункт 2

        1.
        2.
        3.

    Каждый найденный новый пункт начинает новый чанк.

    Если документ не содержит явных номеров,
    весь текст останется одним чанком.
    """

    paragraphs = text.splitlines()

    chunks: List[dict] = []

    current_chunk: List[str] = []

    current_point = "1"

    for paragraph in paragraphs:

        p = paragraph.strip()

        if not p:
            continue

        match = re.match(
            r"^(Статья\s+\d+(?:\.\d+)*"
            r"|Пункт\s+\d+(?:\.\d+)*"
            r"|\d+(?:\.\d+)*\.)",
            p,
            re.IGNORECASE,
        )

        if match and current_chunk:

            content = "\n".join(
                current_chunk
            ).strip()

            if content:

                chunks.append(
                    {
                        "doc_name": doc_name,
                        "point_num": current_point,
                        "content": content,
                    }
                )

            current_chunk = []

            current_point = match.group(
                0
            ).strip()

        elif match:

            current_point = match.group(
                0
            ).strip()

        current_chunk.append(p)

    # Последний чанк.
    if current_chunk:

        content = "\n".join(
            current_chunk
        ).strip()

        if content:

            chunks.append(
                {
                    "doc_name": doc_name,
                    "point_num": current_point,
                    "content": content,
                }
            )

    return chunks


# ============================================================
# SUPABASE EXISTING HASHES
# ============================================================

def get_existing_content_hashes(
    doc_name: str,
) -> set:
    """
    Получает хэши уже существующих чанков.

    Сравнение идёт по content, а не по point_num.
    """

    client = get_supabase()

    response = (
        client
        .table("npa_chunks")
        .select("content")
        .eq("doc_name", doc_name)
        .execute()
    )

    rows = response.data or []

    hashes = set()

    for row in rows:

        content = row.get("content")

        if not content:
            continue

        hashes.add(
            sha256_text(content)
        )

    return hashes


# ============================================================
# DELETE OLD DOCUMENT
# ============================================================

def delete_document_vectors(
    doc_name: str,
) -> None:
    """
    Удаляет все старые чанки документа.

    Используется только при:

        REINDEX_ALL=true

    Это необходимо для перехода со старых
    768-мерных Gemini vectors на 384-мерные e5-small.
    """

    client = get_supabase()

    logger.info(
        "REINDEX | удаление старых чанков: %s",
        doc_name,
    )

    (
        client
        .table("npa_chunks")
        .delete()
        .eq("doc_name", doc_name)
        .execute()
    )

    logger.info(
        "REINDEX | старые чанки удалены: %s",
        doc_name,
    )


# ============================================================
# SUPABASE INSERT
# ============================================================

def insert_rows(
    rows: List[dict],
) -> None:
    """
    Вставляет строки в Supabase небольшими пачками.

    Размер:
        UPLOAD_BATCH_SIZE
    """

    if not rows:
        return

    client = get_supabase()

    total = len(rows)

    for start in range(
        0,
        total,
        UPLOAD_BATCH_SIZE,
    ):

        batch = rows[
            start:start + UPLOAD_BATCH_SIZE
        ]

        last_number = min(
            start + len(batch),
            total,
        )

        for attempt in range(
            1,
            SUPABASE_RETRIES + 1,
        ):

            try:

                (
                    client
                    .table("npa_chunks")
                    .insert(batch)
                    .execute()
                )

                logger.info(
                    "SUPABASE | загружено %d/%d",
                    last_number,
                    total,
                )

                break

            except Exception as error:

                if (
                    attempt
                    >= SUPABASE_RETRIES
                ):
                    raise

                logger.warning(
                    "SUPABASE | ошибка вставки "
                    "(попытка %d/%d): %s",
                    attempt,
                    SUPABASE_RETRIES,
                    error,
                )

                time.sleep(
                    SUPABASE_RETRY_DELAY
                )


# ============================================================
# EMBEDDING VALIDATION
# ============================================================

def validate_embeddings(
    embeddings: List[List[float]],
    expected_count: int,
    file_name: str,
) -> None:
    """
    Проверяет:

    1. количество embeddings;
    2. размерность каждого vector.
    """

    if len(embeddings) != expected_count:

        raise RuntimeError(
            f"Embedding count mismatch "
            f"для '{file_name}': "
            f"получено {len(embeddings)}, "
            f"ожидалось {expected_count}"
        )

    for index, vector in enumerate(
        embeddings,
        start=1,
    ):

        if len(vector) != EMBEDDING_DIM:

            raise RuntimeError(
                f"Неверная размерность embedding "
                f"в '{file_name}', "
                f"vector #{index}: "
                f"{len(vector)} != {EMBEDDING_DIM}"
            )


# ============================================================
# PROCESS FILE
# ============================================================

def process_file(
    file_path: Path,
) -> Tuple[int, int]:
    """
    Обрабатывает один документ.

    Returns:
        uploaded, skipped
    """

    logger.info(
        "=" * 60
    )

    logger.info(
        "DOCUMENT | %s",
        resolve_doc_name(file_path),
    )

    logger.info(
        "FILE | %s",
        file_path.name,
    )

    doc_name = resolve_doc_name(
        file_path
    )

    # --------------------------------------------------------
    # REINDEX
    # --------------------------------------------------------

    if REINDEX_ALL:

        delete_document_vectors(
            doc_name
        )

    # --------------------------------------------------------
    # READ
    # --------------------------------------------------------

    raw_text = read_source(
        file_path
    )

    if not raw_text.strip():

        logger.warning(
            "Файл пустой: %s",
            file_path.name,
        )

        return 0, 0

    logger.info(
        "SOURCE | символов: %d",
        len(raw_text),
    )

    # --------------------------------------------------------
    # CHUNKS
    # --------------------------------------------------------

    chunks = split_text_into_chunks(
        raw_text,
        doc_name,
    )

    total_chunks = len(chunks)

    logger.info(
        "CHUNKS | документ разбит на %d чанков",
        total_chunks,
    )

    if not chunks:

        logger.warning(
            "Не удалось создать чанки: %s",
            file_path.name,
        )

        return 0, 0

    # --------------------------------------------------------
    # EXISTING
    # --------------------------------------------------------

    if REINDEX_ALL:

        existing_hashes = set()

        logger.info(
            "REINDEX | существующие чанки "
            "не проверяются"
        )

    else:

        existing_hashes = (
            get_existing_content_hashes(
                doc_name
            )
        )

        logger.info(
            "SUPABASE | найдено существующих "
            "уникальных чанков: %d",
            len(existing_hashes),
        )

    # --------------------------------------------------------
    # FILTER
    # --------------------------------------------------------

    pending_chunks: List[dict] = []

    skipped = 0

    for chunk in chunks:

        content = chunk["content"]

        content_hash = sha256_text(
            content
        )

        if content_hash in existing_hashes:

            skipped += 1

            continue

        chunk["_hash"] = content_hash

        pending_chunks.append(
            chunk
        )

    logger.info(
        "CHUNKS | новых: %d | пропущено: %d",
        len(pending_chunks),
        skipped,
    )

    if not pending_chunks:

        logger.info(
            "DOCUMENT | все чанки уже загружены: %s",
            doc_name,
        )

        return 0, skipped

    # --------------------------------------------------------
    # EMBEDDINGS + INSERT
    # --------------------------------------------------------

    uploaded = 0

    total_pending = len(
        pending_chunks
    )

    for start in range(
        0,
        total_pending,
        EMBEDDING_BATCH_SIZE,
    ):

        batch = pending_chunks[
            start:start + EMBEDDING_BATCH_SIZE
        ]

        batch_start_number = start + 1

        batch_end_number = min(
            start + len(batch),
            total_pending,
        )

        logger.info(
            "EMBEDDING | [%d-%d/%d] "
            "векторизация...",
            batch_start_number,
            batch_end_number,
            total_pending,
        )

        texts = [
            item["content"]
            for item in batch
        ]

        # ----------------------------------------------------
        # LOCAL E5-SMALL
        # ----------------------------------------------------

        embeddings = (
            get_document_embeddings(
                texts
            )
        )

        # ----------------------------------------------------
        # VALIDATE
        # ----------------------------------------------------

        validate_embeddings(
            embeddings,
            len(batch),
            file_path.name,
        )

        # ----------------------------------------------------
        # CREATE SUPABASE ROWS
        # ----------------------------------------------------

        rows: List[dict] = []

        for chunk, embedding in zip(
            batch,
            embeddings,
        ):

            rows.append(
                {
                    "doc_name": chunk[
                        "doc_name"
                    ],

                    "doc_type": "НПА",

                    "point_num": chunk[
                        "point_num"
                    ],

                    "content": chunk[
                        "content"
                    ],

                    "embedding": embedding,
                }
            )

        # ----------------------------------------------------
        # INSERT
        # ----------------------------------------------------

        insert_rows(rows)

        uploaded += len(rows)

        logger.info(
            "PROGRESS | %s | "
            "uploaded=%d/%d",
            doc_name,
            uploaded,
            total_pending,
        )

    # --------------------------------------------------------
    # DONE
    # --------------------------------------------------------

    logger.info(
        "DOCUMENT DONE | %s | "
        "uploaded=%d | skipped=%d",
        doc_name,
        uploaded,
        skipped,
    )

    return uploaded, skipped


# ============================================================
# FIND SOURCE FILES
# ============================================================

def find_source_files() -> List[Path]:
    """
    Находит все DOCX/TXT в папке скрипта.
    """

    supported_extensions = {
        ".docx",
        ".txt",
    }

    files: List[Path] = []

    for path in SCRIPT_DIR.iterdir():

        if not path.is_file():
            continue

        if path.name.startswith("~$"):
            continue

        if path.suffix.lower() not in supported_extensions:
            continue

        files.append(path)

    return sorted(
        files,
        key=lambda item: item.name.lower(),
    )


# ============================================================
# CHECK FILE MAP BEFORE START
# ============================================================

def validate_source_files(
    files: List[Path],
) -> None:
    """
    Проверяет, что каждому файлу соответствует
    запись в DOC_NAME_MAP.

    Это выполняется ДО начала загрузки,
    чтобы не получить ситуацию, когда половина
    документов уже загружена, а затем обнаруживается
    ошибка имени.
    """

    errors: List[str] = []

    for file_path in files:

        try:

            resolve_doc_name(
                file_path
            )

        except Exception as error:

            errors.append(
                f"{file_path.name}: {error}"
            )

    if errors:

        message = (
            "\n"
            + "=" * 70
            + "\n"
            + "ОШИБКИ СОПОСТАВЛЕНИЯ ФАЙЛОВ\n"
            + "=" * 70
            + "\n"
        )

        message += "\n".join(
            errors
        )

        message += (
            "\n"
            + "=" * 70
        )

        raise RuntimeError(
            message
        )


# ============================================================
# STARTUP MODEL TEST
# ============================================================

def validate_embedding_model() -> None:
    """
    Загружает локальную модель и проверяет,
    что она действительно выдаёт 384 измерения.

    Никаких запросов к Gemini здесь нет.
    """

    logger.info(
        "=" * 60
    )

    logger.info(
        "EMBEDDING MODEL | %s",
        EMBEDDING_MODEL,
    )

    logger.info(
        "EMBEDDING DIMENSION | %d",
        EMBEDDING_DIM,
    )

    logger.info(
        "EMBEDDING BATCH SIZE | %d",
        EMBEDDING_BATCH_SIZE,
    )

    logger.info(
        "UPLOAD BATCH SIZE | %d",
        UPLOAD_BATCH_SIZE,
    )

    logger.info(
        "REINDEX_ALL | %s",
        REINDEX_ALL,
    )

    logger.info(
        "Загрузка локальной embedding-модели..."
    )

    warmup_model()

    test_text = (
        "Охрана труда и безопасность "
        "работников на производстве."
    )

    logger.info(
        "Проверка embedding на тестовом тексте..."
    )

    vectors = get_document_embeddings(
        [test_text]
    )

    if len(vectors) != 1:

        raise RuntimeError(
            "Тест embedding вернул "
            f"{len(vectors)} vectors вместо 1."
        )

    vector = vectors[0]

    if len(vector) != EMBEDDING_DIM:

        raise RuntimeError(
            "Тест embedding вернул "
            f"{len(vector)} измерений, "
            f"ожидалось {EMBEDDING_DIM}."
        )

    logger.info(
        "EMBEDDING TEST | OK | dimension=%d",
        len(vector),
    )

    logger.info(
        "=" * 60
    )


# ============================================================
# MAIN
# ============================================================

def main() -> int:
    """
    Главная функция.
    """

    start_time = time.time()

    logger.info(
        "=" * 60
    )

    logger.info(
        "NPA LOADER START"
    )

    logger.info(
        "SCRIPT DIR | %s",
        SCRIPT_DIR,
    )

    logger.info(
        "EMBEDDING MODEL | %s",
        EMBEDDING_MODEL,
    )

    logger.info(
        "EMBEDDING DIM | %d",
        EMBEDDING_DIM,
    )

    logger.info(
        "REINDEX_ALL | %s",
        REINDEX_ALL,
    )

    # --------------------------------------------------------
    # CONFIG
    # --------------------------------------------------------

    try:

        validate_environment()

    except Exception as error:

        logger.error(
            "Ошибка конфигурации: %s",
            error,
        )

        return 1

    # --------------------------------------------------------
    # FILES
    # --------------------------------------------------------

    files = find_source_files()

    if not files:

        logger.error(
            "В папке %s не найдено "
            "ни одного DOCX/TXT файла.",
            SCRIPT_DIR,
        )

        return 1

    logger.info(
        "Найдено документов для обработки: %d",
        len(files),
    )

    for file_path in files:

        logger.info(
            " - %s",
            file_path.name,
        )

    # --------------------------------------------------------
    # MAP VALIDATION
    # --------------------------------------------------------

    try:

        validate_source_files(
            files
        )

    except Exception as error:

        logger.error(
            "%s",
            error,
        )

        return 1

    logger.info(
        "DOC_NAME_MAP | все файлы успешно сопоставлены."
    )

    # --------------------------------------------------------
    # EMBEDDING MODEL
    # --------------------------------------------------------

    try:

        validate_embedding_model()

    except Exception as err
