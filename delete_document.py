"""
Удаление всех чанков документа из npa_chunks (Supabase) по doc_name.

Нужен перед загрузкой новой редакции НПА: сначала стереть чанки
старой редакции, потом запустить upload.py с новым файлом — иначе
в базе окажутся вперемешку старый и новый текст под одним doc_name.

ИСПОЛЬЗОВАНИЕ:

1. Без аргументов — просто список всех doc_name с количеством
   чанков (чтобы скопировать точное название):
       python delete_document.py

2. Проверка перед удалением (ничего не меняет в базе, только
   показывает, сколько строк будет удалено):
       python delete_document.py "Точное название документа"

3. Реальное удаление:
       python delete_document.py "Точное название документа" --apply

Название нужно передавать ТОЧНО как в списке из режима 1 (в кавычках,
если в названии есть пробелы — а они почти всегда есть).
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from collections import Counter
from dotenv import load_dotenv
from supabase import create_client, Client

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

script_dir = Path(__file__).parent
load_dotenv(dotenv_path=script_dir / '.env')

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Проверьте наличие SUPABASE_URL и SUPABASE_SERVICE_ROLE_KEY в файле .env")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

TABLE = "npa_chunks"


def fetch_all_doc_names() -> list[str]:
    """Постранично забираем doc_name всех строк (без лимита в 1000 от Supabase)."""
    all_names = []
    page_size = 1000
    offset = 0
    while True:
        resp = (
            supabase
            .table(TABLE)
            .select("doc_name")
            .range(offset, offset + page_size - 1)
            .execute()
        )
        rows = resp.data
        if not rows:
            break
        all_names.extend(r["doc_name"] for r in rows)
        if len(rows) < page_size:
            break
        offset += page_size
    return all_names


def list_documents():
    counts = Counter(fetch_all_doc_names())
    print()
    for name, count in sorted(counts.items(), key=lambda x: x[0]):
        print(f"  [{count:>4}]  {name!r}")
    print()
    print("Скопируйте нужное название ТОЧНО как оно здесь показано (в кавычках)")
    print('и запустите: python delete_document.py "Название" --apply')


def delete_document(doc_name: str, do_apply: bool):
    # Точное совпадение (.eq), не префикс — чтобы случайно не задеть
    # соседний документ с похожим началом названия.
    check = (
        supabase
        .table(TABLE)
        .select("id", count="exact")
        .eq("doc_name", doc_name)
        .execute()
    )
    affected = check.count or 0

    if affected == 0:
        logging.error(
            f"Не найдено ни одного чанка с doc_name={doc_name!r}. "
            f"Запустите скрипт без аргументов, чтобы увидеть точные названия в базе."
        )
        sys.exit(1)

    if not do_apply:
        logging.info(
            f"Будет удалено {affected} чанков с doc_name={doc_name!r}. "
            f"Это только проверка — в базе ничего не изменено."
        )
        logging.info("Добавьте --apply в конец команды, чтобы удалить по-настоящему.")
        return

    logging.info(f"Удаляю {affected} чанков с doc_name={doc_name!r}...")
    supabase.table(TABLE).delete().eq("doc_name", doc_name).execute()
    logging.info("Готово. Теперь можно запускать upload.py с файлом новой редакции.")


def main():
    parser = argparse.ArgumentParser(
        description="Удаление чанков документа по doc_name перед загрузкой новой редакции"
    )
    parser.add_argument(
        "doc_name",
        nargs="?",
        default=None,
        help="Точное название документа (doc_name) для удаления. Без аргумента — просто список."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Реально удалить (по умолчанию — только проверка количества)."
    )
    args = parser.parse_args()

    if args.doc_name is None:
        list_documents()
    else:
        delete_document(args.doc_name, args.apply)


if __name__ == "__main__":
    main()
