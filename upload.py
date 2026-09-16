import hashlib
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

REINDEX_ALL = os.getenv("REINDEX_ALL", "false").lower() == "true"

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
#
# Ключ = точное имя файла БЕЗ .docx.
#
# Важно:
# - пробел перед .docx обрабатывается автоматически;
# - requirements.txt сюда не попадёт, потому что загрузчик
#   ниже работает только с .docx.
#

DOC_NAME_MAP: Dict[str, str] = {

    "Об утверждении Межотраслевых типовых инструкций по охране труда":
        "Об утверждении Межотраслевых типовых инструкций по охране труда",

    "Об утверждении Правил по охране труда":
        "Об утверждении Правил по охране труда",

    "Об утверждении Правил по охране труда при выполнении работ на высоте от 6 февраля 2025 г. № 11":
        "Об утверждении Правил по охране труда при выполнении работ на высоте от 6 февраля 2025 г. № 11",

    "Об утверждении Правил по охране труда при работе с инструментом и приспособлениями":
        "Об утверждении Правил по охране труда при работе с инструментом и приспособлениями",

    "Об утверждении Правил по охране труда при эксплуатации электроустановок":
        "Об утверждении Правил по охране труда при эксплуатации электроустановок",

    "Об утверждении Правил по охране труда при эксплуатации тепловых энергоустановок":
        "Об утверждении Правил по охране труда при эксплуатации тепловых энергоустановок",

    "Об утверждении Правил по охране труда в строительстве":
        "Об утверждении Правил по охране труда в строительстве",

    "Об утверждении Правил по охране труда при производстве строительных материалов":
        "Об утверждении Правил по охране труда при производстве строительных материалов",

    "Об утверждении Правил по охране труда при погрузочно-разгрузочных работах":
        "Об утверждении Правил по охране труда при погрузочно-разгрузочных работах",

    "Об утверждении Правил по охране труда при работе с химическими веществами":
        "Об утверждении Правил по охране труда при работе с химическими веществами",

    "Об утверждении Правил по охране труда при эксплуатации систем водоснабжения и канализации":
        "Об утверждении Правил по охране труда при эксплуатации систем водоснабжения и канализации",

    "Об утверждении Правил по охране труда при эксплуатации зданий и сооружений":
        "Об утверждении Правил по охране труда при эксплуатации зданий и сооружений",

    "Об утверждении Правил по охране труда при эксплуатации транспортных средств":
        "Об утверждении Правил по охране труда при эксплуатации транспортных средств",

    "Об утверждении Правил по охране труда при производстве работ в организациях связи":
        "Об утверждении Правил по охране труда при производстве работ в организациях связи",

    "Об утверждении Правил по охране труда при проведении сварочных и других огневых работ":
        "Об утверждении Правил по охране труда при проведении сварочных и других огневых работ",

    "Об утверждении Правил по охране труда при эксплуатации промышленного транспорта":
        "Об утверждении Правил по охране труда при эксплуатации промышленного транспорта",

    "Об утверждении Правил по охране труда при эксплуатации объектов газового хозяйства":
        "Об утверждении Правил по охране труда при эксплуатации объектов газового хозяйства",

    "Об утверждении Правил по охране труда при эксплуатации объектов нефтепродуктообеспечения":
        "Об утверждении Правил по охране труда при эксплуатации объектов нефтепродуктообеспечения",

    "Об утверждении Правил по охране труда при производстве пищевой продукции":
        "Об утверждении Правил по охране труда при производстве пищевой продукции",

    "Об утверждении Правил по охране труда при производстве продукции растениеводства":
        "Об утверждении Правил по охране труда при производстве продукции растениеводства",

    "Об утверждении Правил по охране труда при производстве продукции животноводства":
        "Об утверждении Правил по охране труда при производстве продукции животноводства",

    "Об утверждении Правил по охране труда в лесной промышленности":
        "Об утверждении Правил по охране труда в лесной промышленности",

    "Об утверждении Правил по охране труда при производстве и переработке древесины":
        "Об утверждении Правил по охране труда при производстве и переработке древесины",

    "Об утверждении Правил по охране труда при эксплуатации холодильных установок":
        "Об утверждении Правил по охране труда при эксплуатации холодильных установок",

    "Об утверждении Правил по охране труда при эксплуатации компрессорных установок":
        "Об утверждении Правил по охране труда при эксплуатации компрессорных установок",

    "Об утверждении Правил по охране труда при эксплуатации грузоподъемных машин":
        "Об утверждении Правил по охране труда при эксплуатации грузоподъемных машин",

    "Об утверждении Правил по охране труда при эксплуатации котельных":
        "Об утверждении Правил по охране труда при эксплуатации котельных",
}


# ============================================================
# HELPERS
# ============================================================

def normalize_text(text: str) -> str:
    """
    Нормализует пробелы и служит для сопоставления имён файлов.
    """
    text = str(text or "")
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_filename_stem(file_path: Path) -> str:
    """
    Возвращает имя DOCX без расширения.

    Например:

    'Документ № 11 .docx'

    превращается в:

    'Документ № 11'
    """
    return file_path.stem.strip()


def resolve_doc_name(file_path: Path) -> Optional[str]:
    """
    Сопоставляет DOCX-файл с DOC_NAME_MAP.

    Сначала используется точное совпадение.
    Если оно не найдено — сравнение после нормализации пробелов.
    """

    stem = normalize_filename_stem(file_path)

    # 1. Точное совпадение
    if stem in DOC_NAME_MAP:
        return DOC_NAME_MAP[stem]

    # 2. Нормализованное совпадение
    normalized_stem = normalize_text(stem)

    for key, value in DOC_NAME_MAP.items():
        if normalize_text(key) == normalized_stem:
            return value

    return None


def sha256_text(text: str) -> str:
    """
    SHA-256 текста.

    Используется для определения уже загруженных chunks.
    """
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


# ============================================================
# DOCX READER
# ============================================================

def read_docx(file_path: Path) -> str:
    """
    Читает обычный текст из DOCX.

    Обрабатываются:
    - paragraphs
    - tables
    """

    logger.info(
        "READ DOCX | %s",
        file_path.name,
    )

    document = Document(str(file_path))

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

            cells = []

            for cell in row.cells:

                cell_text = " ".join(
                    paragraph.text.strip()
                    for paragraph in cell.paragraphs
                    if paragraph.text.strip()
                )

                if cell_text:
                    cells.append(cell_text)

            if cells:
                parts.append(" | ".join(cells))

    text = "\n".join(parts)

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Убираем слишком много пустых строк
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
    Делит НПА на chunks.

    Возвращает:

        [
            (point_num, content),
            ...
        ]

    point_num может быть:
        Статья 1
        Пункт 2
        3.
        или 'general'
    """

    text = str(text or "").strip()

    if not text:
        return []

    paragraphs = [
        p.strip()
        for p in re.split(r"\n+", text)
        if p.strip()
    ]

    if not paragraphs:
        return []

    chunks: List[Tuple[str, str]] = []

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

        content = "\n".join(current_lines).strip()

        if content:
            chunks.append(
                (
                    current_point,
                    content,
                )
            )

        current_lines = []

    for paragraph in paragraphs:

        match = marker_pattern.match(paragraph)

        if match:

            flush_current()

            current_point = match.group(1).strip()

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
    Находит ТОЛЬКО DOCX-файлы НПА.

    ВАЖНО:

    requirements.txt
    .env
    .gitignore
    *.py
    *.json
    и любые другие файлы

    полностью игнорируются.

    Также игнорируются временные Word-файлы:

        ~$document.docx
    """

    files: List[Path] = []

    for path in SCRIPT_DIR.iterdir():

        if not path.is_file():
            continue

        # Временные Word-файлы
        if path.name.startswith("~$"):
            continue

        # Загружаем ТОЛЬКО DOCX
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
    Проверяет, что каждый найденный DOCX
    присутствует в DOC_NAME_MAP.
    """

    errors: List[str] = []

    for file_path in files:

        doc_name = resolve_doc_name(file_path)

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
# SUPABASE
# ============================================================

def get_supabase_client():
    """
    Создаёт Supabase client.
    """

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


def supabase_execute(
    operation,
    description: str = "Supabase operation",
):
    """
    Выполняет Supabase operation с retry.
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

            if attempt < SUPABASE_RETRIES:
                time.sleep(
                    SUPABASE_RETRY_DELAY
                    * attempt
                )

    raise RuntimeError(
        f"{description} failed after "
        f"{SUPABASE_RETRIES} attempts: "
        f"{last_error}"
    )


def test_supabase_connection(
    supabase,
) -> None:
    """
    Проверяет соединение с Supabase.
    """

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
# EXISTING HASHES
# ============================================================

def get_existing_content_hashes(
    supabase,
    doc_name: str,
) -> set:
    """
    Получает content_hash уже существующих chunks
    конкретного документа.

    Если content_hash отсутствует в таблице,
    используется пустое множество.
    """

    def operation():

        return (
            supabase
            .table("npa_chunks")
            .select("content_hash")
            .eq("doc_name", doc_name)
            .execute()
        )

    try:

        response = supabase_execute(
            operation,
            f"GET HASHES | {doc_name}",
        )

        rows = response.data or []

        return {
            row.get("content_hash")
            for row in rows
            if row.get("content_hash")
        }

    except Exception as exc:

        logger.warning(
            "GET HASHES | не удалось получить hashes "
            "для '%s': %s",
            doc_name,
            exc,
        )

        return set()


# ============================================================
# DELETE DOCUMENT
# ============================================================

def delete_document_vectors(
    supabase,
    doc_name: str,
) -> None:
    """
    Удаляет все chunks конкретного документа.

    Используется при:

        REINDEX_ALL=true
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
            .eq("doc_name", doc_name)
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
    embeddings: Sequence[Sequence[float]],
    expected_count: int,
) -> None:
    """
    Проверяет количество embeddings
    и размерность каждого embedding.
    """

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
    Загружает embedding model и делает тестовый embedding.
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

    test_embeddings = get_document_embeddings(
        [
            "Тестовый фрагмент нормативного "
            "правового акта по охране труда."
        ]
    )

    validate_embeddings(
        test_embeddings,
        1,
    )

    logger.info(
        "EMBEDDING TEST | OK | dimension=%s",
        len(test_embeddings[0]),
    )


# ============================================================
# INSERT
# ============================================================

def insert_rows(
    supabase,
    rows: List[dict],
) -> int:
    """
    Загружает rows в Supabase батчами.
    """

    if not rows:
        return 0

    uploaded = 0

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
            start // UPLOAD_BATCH_SIZE
        ) + 1

        total_batches = (
            len(rows)
            + UPLOAD_BATCH_SIZE
            - 1
        ) // UPLOAD_BATCH_SIZE

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
    """
    Обрабатывает один DOCX.

    Возвращает:

        uploaded_count
        skipped_count
    """

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
    # Read
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
    # Chunk
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

    existing_hashes = set()

    if REINDEX_ALL:

        delete_document_vectors(
            supabase,
            doc_name,
        )

    else:

        existing_hashes = (
            get_existing_content_hashes(
                supabase,
                doc_name,
            )
        )

        logger.info(
            "EXISTING HASHES | %s",
            len(existing_hashes),
        )

    # --------------------------------------------------------
    # Prepare chunks
    # --------------------------------------------------------

    prepared_chunks = []

    skipped = 0

    for point_num, content in chunks:

        content = content.strip()

        if not content:
            continue

        content_hash = sha256_text(
            content
        )

        if (
            not REINDEX_ALL
            and content_hash in existing_hashes
        ):

            skipped += 1

            continue

        prepared_chunks.append(
            {
                "doc_name": doc_name,
                "doc_type": "НПА",
                "point_num": point_num,
                "content": content,
                "content_hash": content_hash,
            }
        )

    logger.info(
        "CHUNKS | to upload=%s | skipped=%s",
        len(prepared_chunks),
        skipped,
    )

    if not prepared_chunks:

        return 0, skipped

    # --------------------------------------------------------
    # Embeddings
    # --------------------------------------------------------

    all_embeddings: List[
        List[float]
    ] = []

    for start in range(
        0,
        len(prepared_chunks),
        EMBEDDING_BATCH_SIZE,
    ):

        batch = prepared_chunks[
            start:
            start + EMBEDDING_BATCH_SIZE
        ]

        batch_texts = [
            item["content"]
            for item in batch
        ]

        logger.info(
            "EMBEDDING | batch %s-%s / %s",
            start + 1,
            min(
                start + len(batch),
                len(prepared_chunks),
            ),
            len(prepared_chunks),
        )

        embeddings = get_document_embeddings(
            batch_texts
        )

        validate_embeddings(
            embeddings,
            len(batch),
        )

        all_embeddings.extend(
            embeddings
        )

    # --------------------------------------------------------
    # Build Supabase rows
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
                "content_hash": item["content_hash"],
                "embedding": embedding,
            }
        )

    # --------------------------------------------------------
    # Insert
    # --------------------------------------------------------

    uploaded = insert_rows(
        supabase,
        rows,
    )

    logger.info(
        "DONE FILE | %s | uploaded=%s | skipped=%s",
        file_path.name,
        uploaded,
        skipped,
    )

    return uploaded, skipped


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
    # Validate configuration
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
    # Find files
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

    # --------------------------------------------------------
    # Validate files
    # --------------------------------------------------------

    if not files:

        logger.error(
            "DOCX-файлы не найдены."
        )

        return 1

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
    # Validate embedding model
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
    # Supabase
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
    # Process all files
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

            uploaded, skipped = process_file(
                supabase,
                file_path,
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

    elapsed = time.time() - start_time

    # --------------------------------------------------------
    # Final report
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
