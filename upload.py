import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from docx import Document
from pypdf import PdfReader
from google import genai
from google.genai import types
from supabase import create_client, Client


# ============================================================
# 1. КОНФИГУРАЦИЯ
# ============================================================

script_dir = Path(__file__).parent
env_path = script_dir / ".env"

load_dotenv(dotenv_path=env_path)

api_key = os.getenv("GEMINI_API_KEY")
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not api_key:
    raise ValueError(
        f"Ключ GEMINI_API_KEY не найден в файле {env_path}"
    )

if not supabase_url or not supabase_key:
    raise ValueError(
        "Параметры SUPABASE_URL или SUPABASE_SERVICE_ROLE_KEY "
        "не найдены в .env"
    )


# ============================================================
# 2. ИНИЦИАЛИЗАЦИЯ
# ============================================================

gemini_client = genai.Client(api_key=api_key)

supabase: Client = create_client(
    supabase_url,
    supabase_key
)


# ============================================================
# 3. НАСТРОЙКИ EMBEDDING
# ============================================================

EMBEDDING_MODEL = "gemini-embedding-001"

# Должно совпадать с vector(768) в Supabase
EMBEDDING_DIMENSIONS = 768

# Количество попыток при временной ошибке API
MAX_RETRIES = 5

# Задержка между попытками
RETRY_DELAY = 3


# ============================================================
# 4. ИЗВЛЕЧЕНИЕ ТЕКСТА
# ============================================================

def extract_text(file_path: str) -> str:
    """
    Извлечение текста из DOCX или PDF.
    """

    file_path_lower = file_path.lower()

    # ----------------------------
    # DOCX
    # ----------------------------

    if file_path_lower.endswith(".docx"):

        doc = Document(file_path)

        paragraphs = []

        for paragraph in doc.paragraphs:

            text = paragraph.text.strip()

            if text:
                paragraphs.append(text)

        return "\n".join(paragraphs)

    # ----------------------------
    # PDF
    # ----------------------------

    elif file_path_lower.endswith(".pdf"):

        reader = PdfReader(file_path)

        text_parts = []

        for page in reader.pages:

            extracted = page.extract_text()

            if extracted:
                text_parts.append(extracted)

        return "\n".join(text_parts)

    else:

        raise ValueError(
            "Поддерживаются только .docx и .pdf файлы"
        )


# ============================================================
# 5. НАРЕЗКА ДОКУМЕНТА
# ============================================================

def chunk_text(text: str, doc_name: str):
    """
    Нарезка документов по статьям и пунктам.
    """

    pattern = (
        r"(?=\n"
        r"(?:Статья|Глава|Пункт|п\.)?"
        r"\s*\d+(?:\.\d+)*\.\s+)"
    )

    raw_chunks = re.split(pattern, text)

    # Если структура документа не подошла
    if len(raw_chunks) <= 1:

        raw_chunks = text.split("\n\n")

    chunks = []

    for raw in raw_chunks:

        clean = raw.strip()

        if not clean:
            continue

        if len(clean) < 30:
            continue

        # Ищем номер статьи/пункта
        match = re.search(
            r"^(?:Статья|Глава|Пункт|п\.)?"
            r"\s*(\d+(?:\.\d+)*)",
            clean
        )

        point_num = (
            match.group(1)
            if match
            else "Общее"
        )

        content = (
            f"[{doc_name}, ст./п. {point_num}]\n"
            f"{clean}"
        )

        chunks.append({
            "point_num": point_num,
            "content": content
        })

    return chunks


# ============================================================
# 6. ПОЛУЧЕНИЕ EMBEDDING
# ============================================================

def create_embedding(text: str):

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            response = gemini_client.models.embed_content(

                model=EMBEDDING_MODEL,

                contents=text,

                config=types.EmbedContentConfig(

                    task_type="RETRIEVAL_DOCUMENT",

                    output_dimensionality=EMBEDDING_DIMENSIONS
                )
            )

            # Новый формат ответа Gemini
            embedding = response.embeddings[0].values

            # Контроль размерности
            if len(embedding) != EMBEDDING_DIMENSIONS:

                raise ValueError(
                    f"Неверная размерность embedding: "
                    f"{len(embedding)}, "
                    f"ожидалось {EMBEDDING_DIMENSIONS}"
                )

            return embedding

        except Exception as e:

            last_error = e

            print(
                f"    Ошибка embedding "
                f"(попытка {attempt}/{MAX_RETRIES}): {e}"
            )

            if attempt < MAX_RETRIES:

                time.sleep(RETRY_DELAY)

    raise RuntimeError(
        f"Не удалось получить embedding "
        f"после {MAX_RETRIES} попыток: {last_error}"
    )


# ============================================================
# 7. ЗАГРУЗКА ДОКУМЕНТА
# ============================================================

def process_and_upload(
    file_path: str,
    doc_name: str,
    doc_type: str,
    issuer: str,
    source_url: str
):

    print()
    print("=" * 70)
    print(f"Обработка файла: {file_path}")
    print("=" * 70)

    # --------------------------------------------------------
    # Извлечение текста
    # --------------------------------------------------------

    text = extract_text(file_path)

    if not text.strip():

        raise ValueError(
            "Из файла не удалось извлечь текст."
        )

    print(
        f"Извлечено символов: {len(text):,}"
    )

    # --------------------------------------------------------
    # Нарезка
    # --------------------------------------------------------

    chunks = chunk_text(
        text,
        doc_name
    )

    print(
        f"Успешно нарезано на "
        f"{len(chunks)} чанков."
    )

    print()
    print("Начинаем векторизацию...")
    print()

    # --------------------------------------------------------
    # Векторизация
    # --------------------------------------------------------

    records = []

    total = len(chunks)

    for i, chunk in enumerate(chunks, start=1):

        print(
            f"[{i}/{total}] "
            f"Векторизация "
            f"ст./п. {chunk['point_num']}..."
        )

        embedding = create_embedding(
            chunk["content"]
        )

        records.append({

            "doc_name": doc_name,

            "doc_type": doc_type,

            "issuer": issuer,

            "point_num": chunk["point_num"],

            "content": chunk["content"],

            "source_url": source_url,

            "status": "active",

            "embedding": embedding
        })

        # ----------------------------------------------------
        # Загрузка пачкой по 20
        # ----------------------------------------------------

        if len(records) >= 20:

            supabase \
                .table("npa_chunks") \
                .insert(records) \
                .execute()

            print(
                f"    ✓ Загружено в Supabase: "
                f"{i}/{total}"
            )

            records = []

    # --------------------------------------------------------
    # Остаток
    # --------------------------------------------------------

    if records:

        supabase \
            .table("npa_chunks") \
            .insert(records) \
            .execute()

        print(
            f"    ✓ Финальная пачка загружена."
        )

    print()
    print("=" * 70)
    print(
        f"УСПЕХ! Документ '{doc_name}' "
        f"векторизован и загружен в Supabase."
    )
    print("=" * 70)
    print()


# ============================================================
# 8. ЗАПУСК
# ============================================================

if __name__ == "__main__":

    file_name = "ЗООТ.docx"

    possible_paths = [

        script_dir / file_name,

        script_dir / f"{file_name}.docx"
    ]

    target_path = None

    for p in possible_paths:

        if p.exists():

            target_path = p

            break

    if not target_path:

        print()
        print(
            f"ОШИБКА: файл '{file_name}' "
            f"не найден в папке:"
        )

        print(script_dir)

        print()

    else:

        process_and_upload(

            file_path=str(target_path),

            doc_name="Закон об охране труда РБ",

            doc_type="Закон",

            issuer="Парламент РБ",

            source_url="https://pravo.by"
        )