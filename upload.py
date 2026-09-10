import hashlib
import os
import re
import time
import logging
from pathlib import Path
from dotenv import load_dotenv
import docx
from google import genai
from google.genai import types
from supabase import create_client, Client

# 1. Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 2. Загрузка переменных окружения
script_dir = Path(__file__).parent
env_path = script_dir / '.env'
load_dotenv(dotenv_path=env_path)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not GEMINI_API_KEY:
    raise ValueError(f"Ключ GEMINI_API_KEY не найден в файле {env_path}")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Проверьте наличие SUPABASE_URL и SUPABASE_SERVICE_ROLE_KEY в файле .env")

# 3. Инициализация клиентов
gemini_client = genai.Client(api_key=GEMINI_API_KEY)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Константы моделей
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIM = 768
BATCH_SIZE = 40  # Размер пачки для отправки в Supabase


class QuotaExceededError(Exception):
    """
    Дневная квота Gemini API (embed_content) исчерпана.
    Повторные попытки и дальнейшие вызовы в рамках этого запуска
    бессмысленны — квота сбрасывается раз в сутки, а не за секунды.
    """
    pass


def _is_quota_exhausted(error: Exception) -> bool:
    text = str(error)
    return "RESOURCE_EXHAUSTED" in text or "429" in text


def read_docx(file_path: Path) -> str:
    """Чтение текста из файла .docx"""
    doc = docx.Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        if para.text.strip():
            full_text.append(para.text.strip())
    return "\n".join(full_text)


def read_txt(file_path: Path) -> str:
    """Чтение текста из файла .txt"""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def split_text_into_chunks(text: str, doc_name: str):
    """
    Разбиение текста на чанки по статьям и пунктам НПА.
    """
    paragraphs = text.split("\n")
    chunks = []
    current_chunk = []
    current_point = "1"

    for p in paragraphs:
        p_str = p.strip()
        if not p_str:
            continue

        match = re.match(r'^(Статья\s+\d+|Пункт\s+\d+|\d+\.)', p_str, re.IGNORECASE)
        if match and current_chunk:
            chunks.append({
                "doc_name": doc_name,
                "point_num": current_point,
                "content": "\n".join(current_chunk)
            })
            current_chunk = []
            current_point = match.group(0).strip()

        current_chunk.append(p_str)

    if current_chunk:
        chunks.append({
            "doc_name": doc_name,
            "point_num": current_point,
            "content": "\n".join(current_chunk)
        })

    return chunks


def get_existing_content_hashes(doc_name: str) -> set:
    """
    Хэши content уже загруженных чанков этого документа в Supabase.
    Сравнение по точному тексту чанка, а не по point_num — номера
    пунктов не гарантированно уникальны в пределах документа.
    """
    response = (
        supabase
        .table("npa_chunks")
        .select("content")
        .eq("doc_name", doc_name)
        .execute()
    )
    return {
        hashlib.sha256(row["content"].encode("utf-8")).hexdigest()
        for row in response.data
    }


def generate_embedding_with_retry(text: str, retries: int = 3, delay: int = 2):
    """
    Генерация вектора (768 измерений) с обработкой ошибок 503.
    При исчерпании дневной квоты (429/RESOURCE_EXHAUSTED) сразу
    поднимает QuotaExceededError — ретраи и дальнейшие вызовы API
    в этом запуске бессмысленны.
    """
    for attempt in range(1, retries + 1):
        try:
            response = gemini_client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=text,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",
                    output_dimensionality=EMBEDDING_DIM,
                ),
            )
            return response.embeddings[0].values
        except Exception as e:
            if _is_quota_exhausted(e):
                raise QuotaExceededError(
                    "Дневная квота Gemini API (embed_content) исчерпана "
                    "(лимит бесплатного тарифа — 1000 запросов/сутки). "
                    "Загрузка остановлена, уже вставленные чанки сохранены. "
                    "Запустите скрипт повторно позже — он пропустит уже "
                    "загруженные чанки и продолжит с места остановки."
                ) from e
            if ("503" in str(e) or "UNAVAILABLE" in str(e)) and attempt < retries:
                logging.warning(f"Ошибка API (попытка {attempt}/{retries}): {e}. Повтор через {delay} сек...")
                time.sleep(delay)
            else:
                logging.error(f"Не удалось получить вектор для текста: {e}")
                return None
    return None


def process_file(file_path: Path):
    """Полный цикл векторизации и отправки документа"""
    doc_name = file_path.stem
    logging.info(f"Начало обработки документа: '{doc_name}' ({file_path.name})")

    ext = file_path.suffix.lower()
    if ext == ".docx":
        raw_text = read_docx(file_path)
    elif ext == ".txt":
        raw_text = read_txt(file_path)
    else:
        logging.warning(f"Пропуск файла с неподдерживаемым расширением: {file_path.name}")
        return

    if not raw_text.strip():
        logging.warning(f"Файл {file_path.name} пуст!")
        return

    chunks = split_text_into_chunks(raw_text, doc_name)
    total_chunks = len(chunks)
    logging.info(f"Документ '{doc_name}' успешно разбит на {total_chunks} чанков.")

    # --------------------------------------------------------
    # Пропуск уже загруженных чанков (устойчиво к повторному запуску)
    # --------------------------------------------------------
    existing_hashes = get_existing_content_hashes(doc_name)
    if existing_hashes:
        logging.info(f"В базе уже есть чанков этого документа: {len(existing_hashes)} — они будут пропущены.")

    pending_chunks = []
    for chunk in chunks:
        chunk_hash = hashlib.sha256(chunk["content"].encode("utf-8")).hexdigest()
        if chunk_hash in existing_hashes:
            continue
        chunk["_hash"] = chunk_hash
        pending_chunks.append(chunk)

    skipped = total_chunks - len(pending_chunks)
    if skipped:
        logging.info(f"Пропущено уже загруженных чанков: {skipped}")

    if not pending_chunks:
        logging.info(f"Все чанки документа '{doc_name}' уже в базе — пропускаем файл целиком.")
        return

    total_pending = len(pending_chunks)
    batch_records = []

    for idx, chunk in enumerate(pending_chunks, 1):
        print(f"[{idx}/{total_pending}] Векторизация ст./п. {chunk['point_num']}...")

        try:
            vector = generate_embedding_with_retry(chunk["content"])
        except QuotaExceededError as e:
            print()
            print("!" * 70)
            print(str(e))
            print(
                f"Остановлено на {idx - 1}/{total_pending} новых чанков документа "
                f"'{doc_name}' (уже загруженные ранее — {skipped} — не в счёт)."
            )
            print("!" * 70)

            if batch_records:
                supabase.table("npa_chunks").insert(batch_records).execute()
                print(f"  --> Промежуточная пачка ({len(batch_records)}) сохранена перед остановкой.")

            print()
            raise  # прерываем и обработку остальных файлов в main() — квота общая на все документы

        if vector:
            # ИСПРАВЛЕНИЕ: Добавлено обязательное для Supabase поле doc_type
            batch_records.append({
                "doc_name": chunk["doc_name"],
                "doc_type": "НПА",  # Удовлетворяем Not-Null constraint базы
                "point_num": chunk["point_num"],
                "content": chunk["content"],
                "embedding": vector
            })

        # Отправка пакета при достижении BATCH_SIZE
        if len(batch_records) >= BATCH_SIZE:
            supabase.table("npa_chunks").insert(batch_records).execute()
            print(f"  --> Загружена пачка из {len(batch_records)} чанков в Supabase.")
            batch_records = []

    # Финальный остаток
    if batch_records:
        supabase.table("npa_chunks").insert(batch_records).execute()
        print(f"  --> Загружена финальная пачка из {len(batch_records)} чанков.")

    print(f"\n==================================================")
    print(f"УСПЕХ! Файл '{file_path.name}' полностью отправлен в базу.")
    print(f"==================================================\n")


def main():
    """Сканирование папки npa_loader и обработка всех документов"""
    supported_extensions = [".docx", ".txt"]
    all_files = []

    for ext in supported_extensions:
        all_files.extend(list(script_dir.glob(f"*{ext}")))

    # Исключаем временные файлы Word и файлы конфигураций/зависимостей
    files_to_process = [
        f for f in all_files
        if not f.name.startswith("~$") and f.name != "requirements.txt"
    ]

    if not files_to_process:
        logging.error("В папке npa_loader не найдено подходящих документов!")
        return

    logging.info(f"Найдено документов для обработки: {len(files_to_process)}")
    for f in files_to_process:
        logging.info(f" - {f.name}")

    print("\n--- СТАРТ ПАКЕТНОЙ ЗАГРУЗКИ ---")
    for file_path in files_to_process:
        try:
            process_file(file_path)
        except QuotaExceededError:
            logging.error(
                "Квота исчерпана — обработка оставшихся файлов в этом запуске "
                "отменена (квота общая на все документы, повторные вызовы "
                "сейчас всё равно провалятся). Запустите скрипт снова позже."
            )
            break


if __name__ == "__main__":
    main()
