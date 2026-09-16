```python
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from docx import Document
from dotenv import load_dotenv
from supabase import create_client

from embedding import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    get_document_embeddings,
    warmup_model,
)


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

SCRIPT_DIR = Path(__file__).resolve().parent

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

REINDEX_ALL = (
    os.getenv("REINDEX_ALL", "false").lower() == "true"
)

EMBEDDING_BATCH_SIZE = int(
    os.getenv("EMBEDDING_BATCH_SIZE", "8")
)

UPLOAD_BATCH_SIZE = int(
    os.getenv("UPLOAD_BATCH_SIZE", "50")
)

SUPABASE_RETRIES = int(
    os.getenv("SUPABASE_RETRIES", "3")
)

SUPABASE_RETRY_DELAY = float(
    os.getenv("SUPABASE_RETRY_DELAY", "2")
)


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("npa_loader")


# ============================================================
# DOCUMENT MAPPING
# ============================================================

DOC_NAME_MAP: Dict[str, str] = {

    "Закон об охране труда от 23 июня 2008 г. № 356-З":
        "Закон об охране труда от 23 июня 2008 г. № 356-З",

    "Кодекс Республики Беларусь об административных правонарушениях от 6 января 2021 г. № 91-З":
        "Кодекс Республики Беларусь об административных правонарушениях от 6 января 2021 г. № 91-З",

    "О бесплатном обеспечении работников молоком или равноценными пищевыми продуктами при работе с вредными веществами 27 февраля 2002 г. № 260":
        "О бесплатном обеспечении работников молоком или равноценными пищевыми продуктами при работе с вредными веществами 27 февраля 2002 г. № 260",

    "О документах, необходимых для расследования и учета несчастных случаев на производстве и профессиональных заболеваний Минтруда и Соцзащиты от 4 октября 2024 г. № 81 144":
        "О документах, необходимых для расследования и учета несчастных случаев на производстве и профессиональных заболеваний Минтруда и Соцзащиты от 4 октября 2024 г. № 81 144",

    "О контроле состояния водителей от 9 июля 2013 г. № 25.28":
        "О контроле состояния водителей от 9 июля 2013 г. № 25.28",

    "О мерах по укреплению общественной безопасности и дисциплины Директива от 11 марта 2004 г. № 1":
        "О мерах по укреплению общественной безопасности и дисциплины Директива от 11 марта 2004 г. № 1",

    "О пожарной безопасности Закон РБ от 15 июня 1993 г. № 2403-XII":
        "О пожарной безопасности Закон РБ от 15 июня 1993 г. № 2403-XII",

    "О порядке обучения, стажировки, инструктажа и проверки знаний работающих по вопросам охраны труда № 175 от 28 ноября 2008 г":
        "О порядке обучения, стажировки, инструктажа и проверки знаний работающих по вопросам охраны труда № 175 от 28 ноября 2008 г",

    "О порядке проведения предрейсовых и иных медицинских обследований водителей механических транспортных средств (за исключением колесных тракторов) от 3 декабря 2002 г. № 84":
        "О порядке проведения предрейсовых и иных медицинских обследований водителей механических транспортных средств (за исключением колесных тракторов) от 3 декабря 2002 г. № 84",

    "О порядке разработки и принятия локальных правовых актов по охране труда от 28 ноября 2008 г. № 176":
        "О порядке разработки и принятия локальных правовых актов по охране труда от 28 ноября 2008 г. № 176",

    "О порядке расследования и учета несчастных случаев МЧС от 6 января 2023 г. № 6":
        "О порядке расследования и учета несчастных случаев МЧС от 6 января 2023 г. № 6",

    "О проведении обязательных и внеочередных медицинских осмотров работающих от 29 июля 2019 г. № 74":
        "О проведении обязательных и внеочередных медицинских осмотров работающих от 29 июля 2019 г. № 74",

    "О расследовании и учете несчастных случаев на производстве и профессиональных заболеваний от 15 января 2004 г. № 30":
        "О расследовании и учете несчастных случаев на производстве и профессиональных заболеваний от 15 января 2004 г. № 30",

    "Об обеспечении пожарной безопасности постановление МЧС 21 декабря 2021 г. № 82":
        "Об обеспечении пожарной безопасности постановление МЧС 21 декабря 2021 г. № 82",

    "Об утверждении Инструкции о порядке осуществления контроля за соблюдением работниками требований по охране труда от 15 мая 2020 г. № 51":
        "Об утверждении Инструкции о порядке осуществления контроля за соблюдением работниками требований по охране труда от 15 мая 2020 г. № 51",

    "Об утверждении Межотраслевых правил по охране труда при проведении погрузочно-разгрузочных работ от 26 января 2018 г. № 12":
        "Об утверждении Межотраслевых правил по охране труда при проведении погрузочно-разгрузочных работ от 26 января 2018 г. № 12",

    "Об утверждении Межотраслевых правил по охране труда при эксплуатации напольного безрельсового транспорта и грузовых тележек от 30 декабря 2003 г. № 165":
        "Об утверждении Межотраслевых правил по охране труда при эксплуатации напольного безрельсового транспорта и грузовых тележек от 30 декабря 2003 г. № 165",

    "Об утверждении Правил по охране труда при выполнении работ на высоте от 6 февраля 2025 г. № 11":
        "Об утверждении Правил по охране труда при выполнении работ на высоте от 6 февраля 2025 г. № 11",

    "Об утверждении Правил по охране труда при выполнении строительных работ от 31 мая 2019 г. № 24 33":
        "Об утверждении Правил по охране труда при выполнении строительных работ от 31 мая 2019 г. № 24 33",

    "Об утверждении Правил по охране труда при производстве пищевой продукции от 31 декабря 2024 г. № 122":
        "Об утверждении Правил по охране труда при производстве пищевой продукции от 31 декабря 2024 г. № 122",

    "Об утверждении Правил по охране труда при эксплуатации автомобильного и городского электрического транспорта от 6 декабря 2022 г. № 78 104":
        "Об утверждении Правил по охране труда при эксплуатации автомобильного и городского электрического транспорта от 6 декабря 2022 г. № 78 104",

    "Об утверждении специфических санитарно-эпидемиологических требований от 24 января 2020 г. № 42":
        "Об утверждении специфических санитарно-эпидемиологических требований от 24 января 2020 г. № 42",

    "Об утверждении специфических санитарно-эпидемиологических требований Постановление от 1 февраля 2020 г. № 66":
        "Об утверждении специфических санитарно-эпидемиологических требований Постановление от 1 февраля 2020 г. № 66",

    "Об утверждении специфических требований по обеспечению пожарной безопасности взрывопожароопасных и пожароопасных производств 20 ноября 2019 г. № 779":
        "Об утверждении специфических требований по обеспечению пожарной безопасности взрывопожароопасных и пожароопасных производств 20 ноября 2019 г. № 779",

    "Правила по обеспечению СИЗ №209":
        "Правила по обеспечению СИЗ №209",

    "Правила по охране труда № 53":
        "Правила по охране труда № 53",

    "Трудовой кодекс Республики Беларусь 2026":
        "Трудовой кодекс Республики Беларусь 2026",
}


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text: str) -> str:
    """
    Нормализация пробелов для сопоставления имён файлов.
    """

    text = str(text or "")

    text = text.replace(
        "\u00a0",
        " ",
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def normalize_filename_stem(
    file_path: Path,
) -> str:
    """
    Возвращает имя файла без расширения
    и удаляет пробелы по краям.

    Например:

        'Правила № 11 .docx'

    ->

        'Правила № 11'
    """

    return file_path.stem.strip()


# ============================================================
# DOCUMENT NAME RESOLUTION
# ============================================================

def resolve_doc_name(
    file_path: Path,
) -> Optional[str]:
    """
    Сопоставляет DOCX-файл с DOC_NAME_MAP.
    """

    stem = normalize_filename_stem(
        file_path
    )

    # --------------------------------------------------------
    # Exact match
    # --------------------------------------------------------

    if stem in DOC_NAME_MAP:
        return DOC_NAME_MAP[stem]

    # --------------------------------------------------------
    # Normalized match
    # --------------------------------------------------------

    normalized_stem = normalize_text(
        stem
    )

    for key, value in DOC_NAME_MAP.items():

        if normalize_text(key) == normalized_stem:
            return value

    return None


# ============================================================
# DOCX READER
# ============================================================

def read_docx(
    file_path: Path,
) -> str:
    """
    Читает текст из DOCX.

    Обрабатываются:
    - обычные paragraphs;
    - таблицы.
    """

    logger.info(
        "READ DOCX | %s",
        file_path.name,
    )

    document = Document(
        str(file_path)
    )

    parts: List[str] = []

    # --------------------------------------------------------
    # Paragraphs
    # --------------------------------------------------------

    for paragraph in document.paragraphs:

        text = paragraph.text.strip()

        if text:
            parts.append(text)

    # --------------------------------------------------------
    # Tables
    # --------------------------------------------------------

    for table in document.tables:

        for row in table.rows:

            cells: List[str] = []

            for cell in row.cells:

                cell_parts = []

                for paragraph in cell.paragraphs:

                    paragraph_text = (
                        paragraph.text.strip()
                    )

                    if paragraph_text:
                        cell_parts.append(
                            paragraph_text
                        )

                cell_text = " ".join(
                    cell_parts
                )

                if cell_text:
                    cells.append(
                        cell_text
                    )

            if cells:

                parts.append(
                    " | ".join(cells)
                )

    text = "\n".join(parts)

    text = text.replace(
        "\r\n",
        "\n",
    )

    text = text.replace(
        "\r",
        "\n",
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


# ============================================================
# CHUNKING
# ============================================================

def split_text_into_chunks(
    text: str,
) -> List[Tuple[str, str]]:
    """
    Делит текст НПА на chunks.

    Возвращает:

        [
            (
                point_num,
                content
            ),
            ...
        ]

    point_num:
        - Статья 1
        - Пункт 2
        - 3.
        - general
    """

    text = str(
        text or ""
    ).strip()

    if not text:
        return []

    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(
            r"\n+",
            text,
        )
        if paragraph.strip()
    ]

    if not paragraphs:
        return []

    chunks: List[
        Tuple[str, str]
    ] = []

    current_point = "general"

    current_lines: List[str] = []

    marker_pattern = re.compile(
        r"^(Статья\s+\d+(?:\.\d+)*"
        r"|Пункт\s+\d+(?:\.\d+)*"
        r"|\d+(?:\.\d+)*\.)",
        re.IGNORECASE,
    )

    def flush_current() -> None:

        nonlocal current_lines

        if not current_lines:
            return

        content = "\n".join(
            current_lines
        ).strip()

        if content:

            chunks.append(
                (
                    current_point,
                    content,
                )
            )

        current_lines = []

    for paragraph in paragraphs:

        match = marker_pattern.match(
            paragraph
        )

        if match:

            flush_current()

            current_point = (
                match.group(1).strip()
            )

            current_lines = [
                paragraph
            ]

        else:

            current_lines.append(
                paragraph
            )

    flush_current()

    return chunks


# ============================================================
# SOURCE FILE DISCOVERY
# ============================================================

def find_source_files() -> List[Path]:
    """
    Находит только DOCX-документы.

    Игнорируются:

        requirements.txt
        .env
        .gitignore
        *.py
        *.json
        *.txt
        временные Word-файлы ~$*.docx
    """

    files: List[Path] = []

    for path in SCRIPT_DIR.iterdir():

        if not path.is_file():
            continue

        # Word temporary file
        if path.name.startswith("~$"):
            continue

        # ONLY DOCX
        if path.suffix.lower() != ".docx":
            continue

        files.append(path)

    return sorted(
        files,
        key=lambda item: item.name.lower(),
    )


# ============================================================
# SOURCE VALIDATION
# ============================================================

def validate_source_files(
    files: Sequence[Path],
) -> None:
    """
    Проверяет сопоставление всех найденных DOCX
    с DOC_NAME_MAP.
    """

    errors: List[str] = []

    for file_path in files:

        doc_name = resolve_doc_name(
            file_path
        )

        if not doc_name:

            errors.append(
                f"{file_path.name}: "
                f"Для файла '{file_path.name}' "
                f"нет записи в DOC_NAME_MAP.\n"
                f"Нормализованное имя: "
                f"'{normalize_filename_stem(file_path)}'"
            )

    if errors:

        logger.error("")

        logger.error(
            "ОШИБКИ СОПОСТАВЛЕНИЯ ФАЙЛОВ"
        )

        for error in errors:
            logger.error(error)

        raise RuntimeError(
            "Не все DOCX-файлы сопоставлены "
            "с DOC_NAME_MAP."
        )

    logger.info(
        "DOC_NAME_MAP | все файлы успешно сопоставлены."
    )


# ============================================================
# SUPABASE CLIENT
# ============================================================

def get_supabase_client():

    if not SUPABASE_URL:

        raise RuntimeError(
            "SUPABASE_URL не задан."
        )

    if not SUPABASE_SERVICE_ROLE_KEY:

        raise RuntimeError(
            "SUPABASE_SERVICE_ROLE_KEY не задан."
        )

    return create_client(
        SUPABASE_URL,
        SUPABASE_SERVICE_ROLE_KEY,
    )


# ============================================================
# SUPABASE RETRY
# ============================================================

def supabase_execute(
    operation,
    description: str = "Supabase operation",
):
    """
    Выполняет Supabase operation
    с повторными попытками.
    """

    last_error = None

    for attempt in range(
        1,
        SUPABASE_RETRIES + 1,
    ):

        try:

            return operation()

        except Exception as exc:

            last_error = exc

            logger.warning(
                "%s | attempt %s/%s failed: %s",
                description,
                attempt,
                SUPABASE_RETRIES,
                exc,
            )

            if (
                attempt
                < SUPABASE_RETRIES
            ):

                time.sleep(
                    SUPABASE_RETRY_DELAY
                    * attempt
                )

    raise RuntimeError(
        f"{description} failed after "
        f"{SUPABASE_RETRIES} attempts: "
        f"{last_error}"
    )


# ============================================================
# SUPABASE CONNECTION TEST
# ============================================================

def test_supabase_connection(
    supabase,
) -> None:

    def operation():

        return (
            supabase
            .table("npa_chunks")
            .select("id")
            .limit(1)
            .execute()
        )

    supabase_execute(
        operation,
        "SUPABASE connection test",
    )

    logger.info(
        "SUPABASE | соединение успешно."
    )


# ============================================================
# DELETE DOCUMENT
# ============================================================

def delete_document_vectors(
    supabase,
    doc_name: str,
) -> None:
    """
    При REINDEX_ALL=true удаляет все старые chunks
    конкретного документа.
    """

    logger.info(
        "REINDEX | удаление старых chunks: %s",
        doc_name,
    )

    def operation():

        return (
            supabase
            .table("npa_chunks")
            .delete()
            .eq(
                "doc_name",
                doc_name,
            )
            .execute()
        )

    supabase_execute(
        operation,
        f"DELETE DOCUMENT | {doc_name}",
    )


# ============================================================
# EMBEDDING VALIDATION
# ============================================================

def validate_embeddings(
    embeddings: Sequence[
        Sequence[float]
    ],
    expected_count: int,
) -> None:

    if len(embeddings) != expected_count:

        raise RuntimeError(
            "Embedding count mismatch: "
            f"expected {expected_count}, "
            f"got {len(embeddings)}"
        )

    for index, embedding in enumerate(
        embeddings
    ):

        if len(embedding) != EMBEDDING_DIM:

            raise RuntimeError(
                "Embedding dimension mismatch "
                f"at index {index}: "
                f"expected {EMBEDDING_DIM}, "
                f"got {len(embedding)}"
            )


def validate_embedding_model() -> None:
    """
    Загружает модель и проверяет размерность.
    """

    logger.info(
        "EMBEDDING MODEL | %s",
        EMBEDDING_MODEL,
    )

    logger.info(
        "EMBEDDING DIM | %s",
        EMBEDDING_DIM,
    )

    warmup_model()

    test_embeddings = (
        get_document_embeddings(
            [
                "Тестовый фрагмент "
                "нормативного правового "
                "акта по охране труда."
            ]
        )
    )

    validate_embeddings(
        test_embeddings,
        1,
    )

    logger.info(
        "EMBEDDING TEST | OK | dimension=%s",
        len(
            test_embeddings[0]
        ),
    )


# ============================================================
# SUPABASE INSERT
# ============================================================

def insert_rows(
    supabase,
    rows: List[dict],
) -> int:
    """
    Загружает chunks в Supabase батчами.
    """

    if not rows:
        return 0

    uploaded = 0

    total_batches = (
        len(rows)
        + UPLOAD_BATCH_SIZE
        - 1
    ) // UPLOAD_BATCH_SIZE

    for start in range(
        0,
        len(rows),
        UPLOAD_BATCH_SIZE,
    ):

        batch = rows[
            start:
            start + UPLOAD_BATCH_SIZE
        ]

        batch_number = (
            start
            // UPLOAD_BATCH_SIZE
        ) + 1

        logger.info(
            "SUPABASE INSERT | batch %s/%s | rows=%s",
            batch_number,
            total_batches,
            len(batch),
        )

        def operation():

            return (
                supabase
                .table("npa_chunks")
                .insert(batch)
                .execute()
            )

        supabase_execute(
            operation,
            (
                f"INSERT BATCH "
                f"{batch_number}/{total_batches}"
            ),
        )

        uploaded += len(batch)

    return uploaded


# ============================================================
# PROCESS FILE
# ============================================================

def process_file(
    supabase,
    file_path: Path,
) -> Tuple[int, int]:

    doc_name = resolve_doc_name(
        file_path
    )

    if not doc_name:

        raise RuntimeError(
            f"Не найден DOC_NAME_MAP "
            f"для '{file_path.name}'."
        )

    logger.info("")

    logger.info(
        "=" * 70
    )

    logger.info(
        "PROCESS | %s",
        file_path.name,
    )

    logger.info(
        "DOC NAME | %s",
        doc_name,
    )

    logger.info(
        "=" * 70
    )

    # --------------------------------------------------------
    # READ
    # --------------------------------------------------------

    text = read_docx(
        file_path
    )

    if not text:

        logger.warning(
            "SKIP | пустой документ: %s",
            file_path.name,
        )

        return 0, 0

    logger.info(
        "TEXT | chars=%s",
        len(text),
    )

    # --------------------------------------------------------
    # CHUNKS
    # --------------------------------------------------------

    chunks = split_text_into_chunks(
        text
    )

    if not chunks:

        logger.warning(
            "SKIP | chunks не найдены: %s",
            file_path.name,
        )

        return 0, 0

    logger.info(
        "CHUNKS | найдено: %s",
        len(chunks),
    )

    # --------------------------------------------------------
    # REINDEX
    # --------------------------------------------------------

    if REINDEX_ALL:

        delete_document_vectors(
            supabase,
            doc_name,
        )

    # --------------------------------------------------------
    # PREPARE
    # --------------------------------------------------------

    prepared_chunks = []

    for point_num, content in chunks:

        content = content.strip()

        if not content:
            continue

        prepared_chunks.append(
            {
                "doc_name": doc_name,
                "doc_type": "НПА",
                "point_num": point_num,
                "content": content,
            }
        )

    logger.info(
        "CHUNKS | to upload=%s",
        len(prepared_chunks),
    )

    if not prepared_chunks:

        return 0, 0

    # --------------------------------------------------------
    # EMBEDDINGS
    # --------------------------------------------------------

    all_embeddings: List[
        List[float]
    ] = []

    total_embedding_batches = (
        len(prepared_chunks)
        + EMBEDDING_BATCH_SIZE
        - 1
    ) // EMBEDDING_BATCH_SIZE

    for start in range(
        0,
        len(prepared_chunks),
        EMBEDDING_BATCH_SIZE,
    ):

        batch = prepared_chunks[
            start:
            start + EMBEDDING_BATCH_SIZE
        ]

        batch_number = (
            start
            // EMBEDDING_BATCH_SIZE
        ) + 1

        logger.info(
            "EMBEDDING | batch %s/%s | chunks=%s",
            batch_number,
            total_embedding_batches,
            len(batch),
        )

        batch_texts = [
            item["content"]
            for item in batch
        ]

        embeddings = (
            get_document_embeddings(
                batch_texts
            )
        )

        validate_embeddings(
            embeddings,
            len(batch),
        )

        all_embeddings.extend(
            embeddings
        )

    # --------------------------------------------------------
    # BUILD SUPABASE ROWS
    # --------------------------------------------------------

    rows = []

    for item, embedding in zip(
        prepared_chunks,
        all_embeddings,
    ):

        rows.append(
            {
                "doc_name": item["doc_name"],
                "doc_type": item["doc_type"],
                "point_num": item["point_num"],
                "content": item["content"],
                "embedding": embedding,
            }
        )

    # --------------------------------------------------------
    # INSERT
    # --------------------------------------------------------

    uploaded = insert_rows(
        supabase,
        rows,
    )

    logger.info(
        "DONE FILE | %s | uploaded=%s",
        file_path.name,
        uploaded,
    )

    return uploaded, 0


# ============================================================
# MAIN
# ============================================================

def main() -> int:

    logger.info("")

    logger.info(
        "============================================================"
    )

    logger.info(
        "NPA LOADER START"
    )

    logger.info(
        "============================================================"
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
        "EMBEDDING DIM | %s",
        EMBEDDING_DIM,
    )

    logger.info(
        "REINDEX_ALL | %s",
        REINDEX_ALL,
    )

    logger.info(
        "EMBEDDING_BATCH_SIZE | %s",
        EMBEDDING_BATCH_SIZE,
    )

    logger.info(
        "UPLOAD_BATCH_SIZE | %s",
        UPLOAD_BATCH_SIZE,
    )

    # --------------------------------------------------------
    # CONFIG VALIDATION
    # --------------------------------------------------------

    if not SUPABASE_URL:

        logger.error(
            "SUPABASE_URL не задан."
        )

        return 1

    if not SUPABASE_SERVICE_ROLE_KEY:

        logger.error(
            "SUPABASE_SERVICE_ROLE_KEY не задан."
        )

        return 1

    # --------------------------------------------------------
    # FIND DOCX
    # --------------------------------------------------------

    files = find_source_files()

    logger.info(
        "Найдено документов для обработки: %s",
        len(files),
    )

    for file_path in files:

        logger.info(
            " - %s",
            file_path.name,
        )

    if not files:

        logger.error(
            "DOCX-файлы не найдены."
        )

        return 1

    # --------------------------------------------------------
    # VALIDATE MAP
    # --------------------------------------------------------

    try:

        validate_source_files(
            files
        )

    except Exception as exc:

        logger.error(
            "VALIDATION ERROR | %s",
            exc,
        )

        return 1

    # --------------------------------------------------------
    # EMBEDDING MODEL
    # --------------------------------------------------------

    try:

        validate_embedding_model()

    except Exception as exc:

        logger.exception(
            "EMBEDDING ERROR | %s",
            exc,
        )

        return 1

    # --------------------------------------------------------
    # SUPABASE
    # --------------------------------------------------------

    try:

        supabase = get_supabase_client()

        test_supabase_connection(
            supabase
        )

    except Exception as exc:

        logger.exception(
            "SUPABASE ERROR | %s",
            exc,
        )

        return 1

    # --------------------------------------------------------
    # PROCESS
    # --------------------------------------------------------

    total_uploaded = 0

    total_skipped = 0

    failed_files = []

    start_time = time.time()

    for index, file_path in enumerate(
        files,
        start=1,
    ):

        logger.info("")

        logger.info(
            "DOCUMENT %s/%s",
            index,
            len(files),
        )

        try:

            uploaded, skipped = (
                process_file(
                    supabase,
                    file_path,
                )
            )

            total_uploaded += uploaded

            total_skipped += skipped

        except Exception as exc:

            failed_files.append(
                (
                    file_path.name,
                    str(exc),
                )
            )

            logger.exception(
                "FAILED | %s | %s",
                file_path.name,
                exc,
            )

    elapsed = (
        time.time()
        - start_time
    )

    # --------------------------------------------------------
    # FINAL REPORT
    # --------------------------------------------------------

    logger.info("")

    logger.info(
        "============================================================"
    )

    logger.info(
        "FINISHED"
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
        "Reindex all: %s",
        REINDEX_ALL,
    )

    logger.info(
        "Documents found: %s",
        len(files),
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

    logger.info(
        "elapsed=%.1f sec",
        elapsed,
    )

    if failed_files:

        logger.error("")

        logger.error(
            "FAILED DOCUMENTS:"
        )

        for filename, error in failed_files:

            logger.error(
                " - %s: %s",
                filename,
                error,
            )

        return 1

    logger.info("")

    logger.info(
        "ВСЕ ДОКУМЕНТЫ УСПЕШНО ОБРАБОТАНЫ."
    )

    return 0


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        sys.exit(
            main()
        )

    except KeyboardInterrupt:

        logger.warning(
            "Остановлено пользователем."
        )

        sys.exit(130)

    except Exception as exc:

        logger.exception(
            "FATAL ERROR | %s",
            exc,
        )

        sys.exit(1)
```
