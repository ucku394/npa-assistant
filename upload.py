"""
NPA DOCUMENT UPLOADER
=====================

Назначение:
    Чтение НПА из DOCX/TXT,
    разбиение на чанки,
    генерация embeddings через локальный
    intfloat/multilingual-e5-small,
    загрузка в Supabase.

ВАЖНО:
    Этот файл НЕ использует Gemini для embeddings.

Embedding model:
    intfloat/multilingual-e5-small

Embedding dimension:
    384

Documents:
    passage:<text>

Queries:
    query:<text>

Один и тот же embedding.py используется
и для загрузки документов, и для поиска.
"""

import hashlib
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Set

import docx
from dotenv import load_dotenv
from supabase import Client, create_client

from embedding import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    get_document_embeddings,
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
# PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

ENV_PATH = SCRIPT_DIR / ".env"


# ============================================================
# ENV
# ============================================================

load_dotenv(dotenv_path=ENV_PATH)


SUPABASE_URL = os.getenv("SUPABASE_URL")

SUPABASE_SERVICE_ROLE_KEY = os.getenv(
    "SUPABASE_SERVICE_ROLE_KEY"
)


# ------------------------------------------------------------
# REINDEX_ALL
#
# true:
#     удалить старые чанки каждого документа
#     и создать embeddings заново.
#
# false:
#     существующие чанки пропускаются.
# ------------------------------------------------------------

REINDEX_ALL = (
    os.getenv(
        "REINDEX_ALL",
        "false",
    )
    .strip()
    .lower()
    in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }
)


# ------------------------------------------------------------
# Размер пачки embeddings.
#
# Для Railway / 1 GB RAM оставляем небольшим.
# ------------------------------------------------------------

UPLOAD_BATCH_SIZE = int(
    os.getenv(
        "UPLOAD_BATCH_SIZE",
        "8",
    )
)


# ============================================================
# VALIDATION
# ============================================================

if not SUPABASE_URL:
    raise ValueError(
        f"SUPABASE_URL не найден в {ENV_PATH}"
    )


if not SUPABASE_SERVICE_ROLE_KEY:
    raise ValueError(
        "SUPABASE_SERVICE_ROLE_KEY "
        f"не найден в {ENV_PATH}"
    )


if UPLOAD_BATCH_SIZE < 1:
    raise ValueError(
        "UPLOAD_BATCH_SIZE должен быть >= 1"
    )


# ============================================================
# SUPABASE
# ============================================================

supabase: Client = create_client(
    SUPABASE_URL,
    SUPABASE_SERVICE_ROLE_KEY,
)


# ============================================================
# DOCUMENT NAMES
# ============================================================

DOC_NAME_MAP: Dict[str, str] = {

    "Закон об охране труда от 23 июня 2008 г. № 356-З":
        "Закон об охране труда от 23 июня 2008 г. № 356-З",

    "О порядке обучения, стажировки, инструктажа и проверки знаний "
    "работающих по вопросам охраны труда № 175 от 28 ноября 2008 г":
        "О порядке обучения, стажировки, инструктажа и проверки знаний "
        "работающих по вопросам охраны труда № 175 от 28 ноября 2008 г.",

    "Об утверждении Правил по охране труда при выполнении работ на высоте "
    "от 6 февраля 2025 г. № 11":
        "Об утверждении Правил по охране труда при выполнении работ на высоте "
        "от 6 февраля 2025 г. № 11",

    "Правила по обеспечению СИЗ №209":
        "Правила по обеспечению СИЗ №209",

    "Правила по охране труда № 53":
        "Правила по охране труда № 53",

    "Трудовой кодекс Республики Беларусь 2026":
        "Трудовой кодекс Республики Беларусь 2026",

    "О расследовании и учете несчастных случаев на производстве "
    "и профессиональных заболеваний от 15 января 2004 г. № 30":
        "О расследовании и учете несчастных случаев на производстве "
        "и профессиональных заболеваний от 15 января 2004 г. № 30",

    "О порядке разработки и принятия локальных правовых актов "
    "по охране труда от 28 ноября 2008 г. № 176":
        "О порядке разработки и принятия локальных правовых актов "
        "по охране труда от 28 ноября 2008 г. № 176",

    "О документах, необходимых для расследования и учета "
    "несчастных случаев на производстве и профессиональных заболеваний "
    "Минтруда и Соцзащиты от 4 октября 2024 г. № 81 144":
        "О документах, необходимых для расследования и учета "
        "несчастных случаев на производстве и профессиональных заболеваний "
        "Минтруда и Соцзащиты от 4 октября 2024 г. № 81 144",

    "Об утверждении Правил по охране труда при выполнении "
    "строительных работ от 31 мая 2019 г. № 24 33":
        "Об утверждении Правил по охране труда при выполнении "
        "строительных работ от 31 мая 2019 г. № 24 33",

    "О мерах по укреплению общественной безопасности и дисциплины "
    "Директива от 11 марта 2004 г. № 1":
        "О мерах по укреплению общественной безопасности и дисциплины "
        "Директива от 11 марта 2004 г. № 1",

    "Об обеспечении пожарной безопасности "
    "постановление МЧС 21 декабря 2021 г. № 82":
        "Об обеспечении пожарной безопасности "
        "постановление МЧС 21 декабря 2021 г. № 82",

    "Об утверждении Правил по охране труда при производстве "
    "пищевой продукции от 31 декабря 2024 г. № 122":
        "Об утверждении Правил по охране труда при производстве "
        "пищевой продукции от 31 декабря 2024 г. № 122",

    "Об утверждении специфических санитарно-эпидемиологических "
    "требований Постановление от 1 февраля 2020 г. № 66":
        "Об утверждении специфических санитарно-эпидемиологических "
        "требований Постановление от 1 февраля 2020 г. № 66",

    "Об утверждении специфических санитарно-эпидемиологических "
    "требований от 24 января 2020 г. № 42":
        "Об утверждении специфических санитарно-эпидемиологических "
        "требований от 24 января 2020 г. № 42",

    "О бесплатном обеспечении работников молоком или равноценными "
    "пищевыми продуктами при работе с вредными веществами "
    "27 февраля 2002 г. № 260":
        "О бесплатном обеспечении работников молоком или равноценными "
        "пищевыми продуктами при работе с вредными веществами "
        "27 февраля 2002 г. № 260",

    "О контроле состояния водителей от 9 июля 2013 г. № 25.28":
        "О контроле состояния водителей от 9 июля 2013 г. № 25.28",

    "О пожарной безопасности Закон РБ от 15 июня 1993 г. № 2403-XII":
        "О пожарной безопасности Закон РБ от 15 июня 1993 г. № 2403-XII",

    "О порядке проведения предрейсовых и иных медицинских "
    "обследований водителей механических транспортных средств "
    "(за исключением колесных тракторов) от 3 декабря 2002 г. № 84":
        "О порядке проведения предрейсовых и иных медицинских "
        "обследований водителей механических транспортных средств "
        "(за исключением колесных тракторов) от 3 декабря 2002 г. № 84",

    "О порядке расследования и учета несчастных случаев "
    "МЧС от 6 января 2023 г. № 6":
        "О порядке расследования и учета несчастных случаев "
        "МЧС от 6 января 2023 г. № 6",

    "О проведении обязательных и внеочередных медицинских "
    "осмотров работающих от 29 июля 2019 г. № 74":
        "О проведении обязательных и внеочередных медицинских "
        "осмотров работающих от 29 июля 2019 г. № 74",

    "Об утверждении Инструкции о порядке осуществления контроля "
    "за соблюдением работниками требований по охране труда "
    "от 15 мая 2020 г. № 51":
        "Об утверждении Инструкции о порядке осуществления контроля "
        "за соблюдением работниками требований по охране труда "
        "от 15 мая 2020 г. № 51",

    "Об утверждении Межотраслевых правил по охране труда "
    "при проведении погрузочно-разгрузочных работ "
    "от 26 января 2018 г. № 12":
        "Об утверждении Межотраслевых правил по охране труда "
        "при проведении погрузочно-разгрузочных работ "
        "от 26 января 2018 г. № 12",

    "Об утверждении Межотраслевых правил по охране труда "
    "при эксплуатации напольного безрельсового транспорта "
    "и грузовых тележек от 30 декабря 2003 г. № 165":
        "Об утверждении Межотраслевых правил по охране труда "
        "при эксплуатации напольного безрельсового транспорта "
        "и грузовых тележек от 30 декабря 2003 г. № 165",

    "Об утверждении Правил по охране труда при эксплуатации "
    "автомобильного и городского электрического транспорта "
    "от 6 декабря 2022 г. № 78 104":
        "Об утверждении Правил по охране труда при эксплуатации "
        "автомобильного и городского электрического транспорта "
        "от 6 декабря 2022 г. № 78 104",

    "Об утверждении специфических требований по обеспечению "
    "пожарной безопасности взрывопожароопасных и пожароопасных "
    "производств 20 ноября 2019 г. № 779":
        "Об утверждении специфических требований по обеспечению "
        "пожарной безопасности взрывопожароопасных и пожароопасных "
        "производств 20 ноября 2019 г. № 779",

    "Кодекс Республики Беларусь об административных правонарушениях "
    "от 6 января 2021 г. № 91-З":
        "Кодекс Республики Беларусь об административных правонарушениях "
        "от 6 января 2021 г. № 91-З",
}


# ============================================================
# FILE READING
# ============================================================

def read_docx(file_path: Path) -> str:
    """
    Чтение текста из DOCX.
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
    Чтение текста из TXT.
    """

    with open(
        file_path,
        "r",
        encoding="utf-8",
    ) as file:

        return file.read()


# ============================================================
# CHUNKING
# ============================================================

def split_text_into_chunks(
    text: str,
    doc_name: str,
) -> List[dict]:
    """
    Разбиение НПА на чанки.

    Новый чанк начинается при обнаружении:
        Статья N
        Пункт N
        N.
    """

    paragraphs = text.split("\n")

    chunks: List[dict] = []

    current_chunk: List[str] = []

    current_point = "1"

    for paragraph in paragraphs:

        p = paragraph.strip()

        if not p:
            continue

        match = re.match(
            r"^(Статья\s+\d+|Пункт\s+\d+|\d+\.)",
            p,
            re.IGNORECASE,
        )

        if match and current_chunk:

            chunks.append(
                {
                    "doc_name": doc_name,
                    "point_num": current_point,
                    "content": "\n".join(
                        current_chunk
                    ),
                }
            )

            current_chunk = []

            current_point = (
                match.group(0).strip()
            )

        current_chunk.append(p)

    if current_chunk:

        chunks.append(
            {
                "doc_name": doc_name,
                "point_num": current_point,
                "content": "\n".join(
                    current_chunk
                ),
            }
        )

    return chunks


# ============================================================
# HASH
# ============================================================

def content_hash(content: str) -> str:
    """
    SHA256 хэш текста чанка.
    """

    return hashlib.sha256(
        content.encode("utf-8")
    ).hexdigest()


# ============================================================
# EXISTING CHUNKS
# ============================================================

def get_existing_content_hashes(
    doc_name: str,
) -> Set[str]:
    """
    Получает хэши уже существующих чанков
    конкретного документа.
    """

    response = (
        supabase
        .table("npa_chunks")
        .select("content")
        .eq("doc_name", doc_name)
        .execute()
    )

    rows = response.data or []

    return {
        content_hash(
            row["content"]
        )
        for row in rows
        if row.get("content")
    }


# ============================================================
# DELETE DOCUMENT
# ============================================================

def delete_document(
    doc_name: str,
) -> int:
    """
    Удаляет все старые чанки документа.

    Используется только при:
        REINDEX_ALL=true
    """

    response = (
        supabase
        .table("npa_chunks")
        .delete()
        .eq("doc_name", doc_name)
        .execute()
    )

    deleted = len(
        response.data or []
    )

    logger.info(
        "REINDEX | deleted=%s | %s",
        deleted,
        doc_name,
    )

    return deleted


# ============================================================
# EMBEDDINGS
# ============================================================

def generate_embeddings(
    texts: List[str],
) -> List[List[float]]:
    """
    Генерирует embeddings через локальный
    multilingual-e5-small.

    ВАЖНО:
        embedding.py уже добавляет:
            passage:
    """

    if not texts:
        return []

    embeddings = get_document_embeddings(
        texts
    )

    if len(embeddings) != len(texts):

        raise RuntimeError(
            "Количество embeddings не совпадает "
            "с количеством текстов: "
            f"{len(embeddings)} != {len(texts)}"
        )

    for index, embedding in enumerate(
        embeddings
    ):

        if len(embedding) != EMBEDDING_DIM:

            raise RuntimeError(
                "Неверная размерность embedding "
                f"для чанка {index}: "
                f"получено {len(embedding)}, "
                f"ожидалось {EMBEDDING_DIM}"
            )

    return embeddings


# ============================================================
# INSERT BATCH
# ============================================================

def insert_batch(
    chunks: List[dict],
    embeddings: List[List[float]],
) -> int:
    """
    Загружает пачку чанков в Supabase.
    """

    if len(chunks) != len(embeddings):

        raise RuntimeError(
            "Количество чанков и embeddings "
            "не совпадает."
        )

    records = []

    for chunk, embedding in zip(
        chunks,
        embeddings,
    ):

        records.append(
            {
                "doc_name": chunk["doc_name"],
                "doc_type": "НПА",
                "point_num": chunk["point_num"],
                "content": chunk["content"],
                "embedding": embedding,
            }
        )

    if not records:
        return 0

    response = (
        supabase
        .table("npa_chunks")
        .insert(records)
        .execute()
    )

    inserted = len(
        response.data or records
    )

    return inserted


# ============================================================
# PROCESS DOCUMENT
# ============================================================

def process_file(
    file_path: Path,
) -> tuple[int, int]:
    """
    Полный цикл обработки одного документа.

    Возвращает:
        uploaded, skipped
    """

    file_stem = file_path.stem

    # --------------------------------------------------------
    # Проверка DOC_NAME_MAP
    # --------------------------------------------------------

    if file_stem not in DOC_NAME_MAP:

        raise ValueError(
            f"Для файла '{file_path.name}' "
            "нет записи в DOC_NAME_MAP.\n"
            "Добавьте его в DOC_NAME_MAP."
        )

    doc_name = (
        DOC_NAME_MAP[file_stem]
        .strip()
    )

    logger.info(
        "============================================================"
    )

    logger.info(
        "DOCUMENT | %s",
        doc_name,
    )

    logger.info(
        "FILE | %s",
        file_path.name,
    )

    # --------------------------------------------------------
    # Чтение
    # --------------------------------------------------------

    extension = (
        file_path.suffix.lower()
    )

    if extension == ".docx":

        raw_text = read_docx(
            file_path
        )

    elif extension == ".txt":

        raw_text = read_txt(
            file_path
        )

    else:

        logger.warning(
            "Unsupported file: %s",
            file_path.name,
        )

        return 0, 0

    if not raw_text.strip():

        logger.warning(
            "Файл пуст: %s",
            file_path.name,
        )

        return 0, 0

    # --------------------------------------------------------
    # Chunking
    # --------------------------------------------------------

    chunks = split_text_into_chunks(
        raw_text,
        doc_name,
    )

    total_chunks = len(chunks)

    logger.info(
        "CHUNKS | total=%s",
        total_chunks,
    )

    if not chunks:

        return 0, 0

    # --------------------------------------------------------
    # REINDEX
    # --------------------------------------------------------

    if REINDEX_ALL:

        logger.info(
            "REINDEX_ALL=true | "
            "удаляем старые embeddings документа"
        )

        delete_document(
            doc_name
        )

        existing_hashes: Set[str] = set()

    else:

        existing_hashes = (
            get_existing_content_hashes(
                doc_name
            )
        )

        logger.info(
            "EXISTING | %s",
            len(existing_hashes),
        )

    # --------------------------------------------------------
    # Determine pending chunks
    # --------------------------------------------------------

    pending_chunks: List[dict] = []

    for chunk in chunks:

        chunk_hash = content_hash(
            chunk["content"]
        )

        if (
            not REINDEX_ALL
            and chunk_hash in existing_hashes
        ):
            continue

        pending_chunks.append(
            chunk
        )

    skipped = (
        total_chunks
        - len(pending_chunks)
    )

    logger.info(
        "PENDING | %s",
        len(pending_chunks),
    )

    logger.info(
        "SKIPPED | %s",
        skipped,
    )

    if not pending_chunks:

        logger.info(
            "Nothing new for %s; skipped=%s",
            doc_name,
            skipped,
        )

        return 0, skipped

    # --------------------------------------------------------
    # Generate embeddings in batches
    # --------------------------------------------------------

    uploaded = 0

    total_pending = len(
        pending_chunks
    )

    for start in range(
        0,
        total_pending,
        UPLOAD_BATCH_SIZE,
    ):

        end = min(
            start + UPLOAD_BATCH_SIZE,
            total_pending,
        )

        batch_chunks = (
            pending_chunks[start:end]
        )

        batch_number = (
            start // UPLOAD_BATCH_SIZE
        ) + 1

        total_batches = (
            (
                total_pending
                + UPLOAD_BATCH_SIZE
                - 1
            )
            // UPLOAD_BATCH_SIZE
        )

        logger.info(
            "BATCH | %s/%s | chunks=%s | progress=%s-%s/%s",
            batch_number,
            total_batches,
            len(batch_chunks),
            start + 1,
            end,
            total_pending,
        )

        # ----------------------------------------------------
        # Texts
        # ----------------------------------------------------

        texts = [
            chunk["content"]
            for chunk in batch_chunks
        ]

        # ----------------------------------------------------
        # Local E5 embeddings
        # ----------------------------------------------------

        embeddings = generate_embeddings(
            texts
        )

        logger.info(
            "EMBEDDING | model=%s | dimension=%s | count=%s",
            EMBEDDING_MODEL,
            EMBEDDING_DIM,
            len(embeddings),
        )

        # ----------------------------------------------------
        # Supabase
        # ----------------------------------------------------

        inserted = insert_batch(
            batch_chunks,
            embeddings,
        )

        uploaded += inserted

        logger.info(
            "SUPABASE | inserted=%s | uploaded_total=%s/%s",
            inserted,
            uploaded,
            total_pending,
        )

    # --------------------------------------------------------
    # Document finished
    # --------------------------------------------------------

    logger.info(
        "DONE | %s | uploaded=%s | skipped=%s",
        doc_name,
        uploaded,
        skipped,
    )

    return uploaded, skipped


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    logger.info(
        "============================================================"
    )

    logger.info(
        "NPA EMBEDDING UPLOADER"
    )

    logger.info(
        "============================================================"
    )

    logger.info(
        "Embedding model: %s",
        EMBEDDING_MODEL,
    )

    logger.info(
        "Embedding dimension: %s",
        EMBEDDING_DIM,
    )

    logger.info(
        "Upload batch size: %s",
        UPLOAD_BATCH_SIZE,
    )

    logger.info(
        "REINDEX_ALL: %s",
        REINDEX_ALL,
    )

    logger.info(
        "Supabase URL: %s",
        SUPABASE_URL,
    )

    logger.info(
        "============================================================"
    )

    # --------------------------------------------------------
    # Проверка embedding-модели ДО загрузки документов
    # --------------------------------------------------------

    logger.info(
        "EMBEDDING | проверка локальной модели..."
    )

    # Небольшой тестовый embedding.
    test_embeddings = get_document_embeddings(
        ["Тестовый фрагмент НПА."]
    )

    if not test_embeddings:

        raise RuntimeError(
            "Embedding model returned no vector."
        )

    if len(test_embeddings[0]) != EMBEDDING_DIM:

        raise RuntimeError(
            "Embedding test failed: "
            f"got {len(test_embeddings[0])}, "
            f"expected {EMBEDDING_DIM}"
        )

    logger.info(
        "EMBEDDING | OK | model=%s | dimension=%s",
        EMBEDDING_MODEL,
        EMBEDDING_DIM,
    )

    # --------------------------------------------------------
    # Search documents
    # --------------------------------------------------------

    supported_extensions = {
        ".docx",
        ".txt",
    }

    all_files: List[Path] = []

    for extension in supported_extensions:

        all_files.extend(
            SCRIPT_DIR.glob(
                f"*{extension}"
            )
        )

    files_to_process = sorted(
        [
            file
            for file in all_files
            if not file.name.startswith("~$")
            and file.name != "requirements.txt"
        ],
        key=lambda x: x.name.lower(),
    )

    if not files_to_process:

        logger.error(
            "В папке %s "
            "не найдено DOCX/TXT документов.",
            SCRIPT_DIR,
        )

        sys.exit(1)

    logger.info(
        "Найдено документов: %s",
        len(files_to_process),
    )

    for file_path in files_to_process:

        logger.info(
            "  - %s",
            file_path.name,
        )

    # --------------------------------------------------------
    # Totals
    # --------------------------------------------------------

    total_uploaded = 0
    total_skipped = 0
    failed_files = []

    # --------------------------------------------------------
    # Process documents
    # --------------------------------------------------------

    for file_path in files_to_process:

        try:

            uploaded, skipped = process_file(
                file_path
            )

            total_uploaded += uploaded
            total_skipped += skipped

        except Exception as error:

            logger.exception(
                "ОШИБКА при обработке %s: %s",
                file_path.name,
                error,
            )

            failed_files.append(
                file_path.name
            )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    logger.info(
        "============================================================"
    )

    logger.info(
        "FINISHED"
    )

    logger.info(
        "Embedding model: %s",
        EMBEDDING_MODEL,
    )

    logger.info(
        "Embedding dimension: %s",
        EMBEDDING_DIM,
    )

    logger.info(
        "uploaded=%s",
        total_uploaded,
    )

    logger.info(
        "skipped=%s",
        total_skipped,
    )

    logger.info(
        "failed=%s",
        len(failed_files),
    )

    if failed_files:

        logger.error(
            "Не обработаны файлы:"
        )

        for file_name in failed_files:

            logger.error(
                "  - %s",
                file_name,
            )

        logger.error(
            "Загрузка завершена с ошибками."
        )

        sys.exit(1)

    logger.info(
        "Загрузка завершена успешно."
    )

    logger.info(
        "============================================================"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
