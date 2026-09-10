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

# Константы моделей и загрузки
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIM = 768
BATCH_SIZE = 40  # Размер пачки для паакетной отправки в Supabase


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
    Умное разбиение текста на статьи и пункты НПА (например, 'Статья 5.', 'Пункт 12.', '1. ').
    """
    paragraphs = text.split("\n")
    chunks = []
    current_chunk = []
    current_point = "1"

    for p in paragraphs:
        p_str = p.strip()
        if not p_str:
            continue
            
        # Проверка начала новой статьи или пункта
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


def generate_embedding_with_retry(text: str, retries: int = 3, delay: int = 2):
    """
    Генерация вектора (768 измерений) с обработкой ошибок 503 и повторными попытками.
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
            if ("503" in str(e) or "UNAVAILABLE" in str(e)) and attempt < retries:
                logging.warning(f"Ошибка API (попытка {attempt}/{retries}): {e}. Повтор через {delay} сек...")
                time.sleep(delay)
            else:
                logging.error(f"Не удалось получить вектор для текста: {e}")
                return None
    return None


def process_file(file_path: Path):
    """Полный цикл векторизации одного документа"""
    doc_name = file_path.stem
    logging.info(f"Начало обработки документа: '{doc_name}' ({file_path.name})")

    # Чтение текста
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

    # Нарезка на чанки
    chunks = split_text_into_chunks(raw_text, doc_name)
    total_chunks = len(chunks)
    logging.info(f"Документ '{doc_name}' успешно разбит на {total_chunks} чанков.")

    # Генерация векторов и сохранение пачками
    batch_records = []
    
    for idx, chunk in enumerate(chunks, 1):
        print(f"[{idx}/{total_chunks}] Векторизация ст./п. {chunk['point_num']}...")
        vector = generate_embedding_with_retry(chunk["content"])

        if vector:
            batch_records.append({
                "doc_name": chunk["doc_name"],
                "point_num": chunk["point_num"],
                "content": chunk["content"],
                "embedding": vector
            })

        # Отправка пакета данных при накоплении BATCH_SIZE
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
    """
    8. АВТОМАТИЧЕСКИЙ ЗАПУСК
    Сканирование пачки документов в директории без жесткой привязки к имени.
    """
    supported_extensions = [".docx", ".txt"]
    all_files = []

    for ext in supported_extensions:
        all_files.extend(list(script_dir.glob(f"*{ext}")))

    # Исключение скрытых/временных файлов Word вида ~$doc.docx
    files_to_process = [f for f in all_files if not f.name.startswith("~$")]

    if not files_to_process:
        logging.error("В папке npa_loader не найдено подходящих документов (.docx или .txt)!")
        return

    logging.info(f"Найдено документов для обработки: {len(files_to_process)}")
    for f in files_to_process:
        logging.info(f" - {f.name}")

    print("\n--- СТАРТ ПАКЕТНОЙ ЗАГРУЗКИ ---")
    for file_path in files_to_process:
        process_file(file_path)


if __name__ == "__main__":
    main()
