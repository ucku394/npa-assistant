"""
Миграция doc_name в таблице npa_chunks (Supabase).

Проблема: один и тот же документ мог попасть в базу под разными
значениями doc_name (например, из-за отсутствия записи в DOC_NAME_MAP
на момент загрузки, или из-за загрузки под разными именами файлов).
Этот скрипт помогает найти такие дубли и привести их к одному
каноническому названию.

РЕЖИМЫ РАБОТЫ:

1. Аудит (по умолчанию, ничего не меняет в базе):
       python migrate_doc_names.py
   Выводит все уникальные doc_name с количеством чанков у каждого,
   отсортированные так, чтобы похожие названия оказались рядом —
   удобно на глаз найти дубли одного документа.

2. Применение маппинга (реально обновляет базу):
       python migrate_doc_names.py --apply
   Перед этим заполните словарь CANONICAL_MAP ниже: ключ — старое
   значение doc_name (как оно есть в базе, скопируйте из вывода
   аудита), значение — каноническое название, к которому всё нужно
   привести. Разные ключи могут указывать на один и тот же канон —
   это и есть объединение дублей.

Скрипт ничего не удаляет и не создаёт — только обновляет столбец
doc_name у существующих строк (UPDATE ... WHERE doc_name = <старое>).
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

# --------------------------------------------------------
# ЗАПОЛНИТЕ ПЕРЕД ЗАПУСКОМ С --apply
# Ключ — старое doc_name (точная строка из базы, скопируйте из аудита)
# Значение — каноническое название, к которому приводим
# --------------------------------------------------------
CANONICAL_MAP = {
    'Закон об охране труда от 23 июня 2008 г. № 356-З от 23.06.2008 № 356-З':
        'Закон об охране труда от 23 июня 2008 г. № 356-З',
    'Закон об охране труда от 23 июня 2008 г. № 356-З от 23.06.2008 № 356-З от 23.06.2008 № 356-З':
        'Закон об охране труда от 23 июня 2008 г. № 356-З',

    'О порядке обучения, стажировки, инструктажа и проверки знаний работающих по вопросам охраны труда № О порядке обучения, стажировки, инструктажа и проверки знаний работающих по вопросам охраны труда № 175 от 28 ноября 2008 г. от 28 ноября 2008 г.  ':
        'О порядке обучения, стажировки, инструктажа и проверки знаний работающих по вопросам охраны труда № 175 от 28 ноября 2008 г.',
    'Постановление\r\nМинистерства труда\r\nи социальной защиты\r\nРеспублики Беларусь\r\n28.11.2008 № О порядке обучения, стажировки, инструктажа и проверки знаний работающих по вопросам охраны труда № 175 от 28 ноября 2008 г.':
        'О порядке обучения, стажировки, инструктажа и проверки знаний работающих по вопросам охраны труда № 175 от 28 ноября 2008 г.',

    'Об утверждении Правил по охране труда при выполнении работ на высоте от 6 февраля 2025 г. № 11 ':
        'Об утверждении Правил по охране труда при выполнении работ на высоте от 6 февраля 2025 г. № 11',
}


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


def audit():
    names = fetch_all_doc_names()
    counts = Counter(names)
    logging.info(f"Всего чанков: {len(names)}, уникальных doc_name: {len(counts)}")
    print()

    # Сортируем по названию, чтобы похожие строки (частые кандидаты
    # на дубль одного документа) оказались рядом в выводе.
    for name, count in sorted(counts.items(), key=lambda x: x[0]):
        print(f"  [{count:>4}]  {name!r}")

    print()
    print("Просмотрите список выше, найдите документы, которые встречаются")
    print("под разными названиями, и заполните CANONICAL_MAP в этом файле.")
    print("Затем запустите: python migrate_doc_names.py --apply")


def apply():
    if not CANONICAL_MAP:
        logging.error("CANONICAL_MAP пуст — нечего применять. Заполните словарь в файле.")
        sys.exit(1)

    for old_name, new_name in CANONICAL_MAP.items():
        if old_name == new_name:
            continue

        # Считаем, сколько строк реально попадёт под замену
        check = (
            supabase
            .table(TABLE)
            .select("id", count="exact")
            .eq("doc_name", old_name)
            .execute()
        )
        affected = check.count or 0

        if affected == 0:
            logging.warning(f"Нет строк с doc_name={old_name!r} — пропуск.")
            continue

        logging.info(f"{old_name!r} -> {new_name!r} ({affected} чанков)")
        supabase.table(TABLE).update({"doc_name": new_name}).eq("doc_name", old_name).execute()

    print()
    logging.info("Готово. Рекомендуется повторно запустить скрипт без --apply,")
    logging.info("чтобы убедиться, что дублей больше нет.")


def main():
    parser = argparse.ArgumentParser(description="Аудит и объединение дублей doc_name в Supabase")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Реально применить CANONICAL_MAP к базе (по умолчанию — только аудит)."
    )
    args = parser.parse_args()

    if args.apply:
        apply()
    else:
        audit()


if __name__ == "__main__":
    main()