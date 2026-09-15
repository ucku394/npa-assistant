import os
import sys
import time
from dotenv import load_dotenv
from supabase import create_client
from sentence_transformers import SentenceTransformer


# ============================================================
# НАСТРОЙКИ
# ============================================================

MODEL_NAME = "intfloat/multilingual-e5-base"

# Размер пакета для генерации embeddings
BATCH_SIZE = 16

# Сколько строк получать из Supabase за один запрос
FETCH_BATCH_SIZE = 500

# ============================================================
# ЗАГРУЗКА .ENV
# ============================================================

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL:
    print("ОШИБКА: не найден SUPABASE_URL в .env")
    sys.exit(1)

if not SUPABASE_SERVICE_ROLE_KEY:
    print("ОШИБКА: не найден SUPABASE_SERVICE_ROLE_KEY в .env")
    sys.exit(1)


# ============================================================
# SUPABASE
# ============================================================

print("=" * 70)
print("ПЕРЕИНДЕКСАЦИЯ NPA_CHUNKS")
print("=" * 70)

print("\nПодключение к Supabase...")

supabase = create_client(
    SUPABASE_URL,
    SUPABASE_SERVICE_ROLE_KEY
)

print("Supabase: OK")


# ============================================================
# МОДЕЛЬ EMBEDDING
# ============================================================

print("\nЗагрузка модели:")
print(MODEL_NAME)

try:
    model = SentenceTransformer(MODEL_NAME)
except Exception as e:
    print("\nОШИБКА загрузки модели:")
    print(e)
    sys.exit(1)

print("Модель загружена.")

# Проверяем размерность
test_embedding = model.encode(
    ["passage: тест проверки размерности"],
    normalize_embeddings=True
)[0]

embedding_dimension = len(test_embedding)

print(f"Размерность embedding: {embedding_dimension}")

if embedding_dimension != 768:
    print(
        f"\nОШИБКА: ожидалась размерность 768, "
        f"получено {embedding_dimension}"
    )
    sys.exit(1)

print("Размерность: OK")


# ============================================================
# ПОЛУЧАЕМ ОБЩЕЕ КОЛИЧЕСТВО
# ============================================================

print("\nПолучение количества чанков...")

count_response = (
    supabase
    .table("npa_chunks")
    .select("id", count="exact")
    .execute()
)

total = count_response.count

if total is None:
    print("ОШИБКА: Supabase не вернул количество записей.")
    sys.exit(1)

print(f"Всего чанков: {total}")


if total == 0:
    print("Таблица npa_chunks пустая.")
    sys.exit(0)


# ============================================================
# ПЕРЕИНДЕКСАЦИЯ
# ============================================================

processed = 0
updated = 0
skipped = 0
errors = 0

start_time = time.time()


for offset in range(0, total, FETCH_BATCH_SIZE):

    print("\n" + "-" * 70)
    print(
        f"Получение записей "
        f"{offset + 1} - {min(offset + FETCH_BATCH_SIZE, total)}"
    )
    print("-" * 70)

    try:
        response = (
            supabase
            .table("npa_chunks")
            .select("id, content")
            .range(
                offset,
                min(offset + FETCH_BATCH_SIZE, total) - 1
            )
            .execute()
        )

        rows = response.data

    except Exception as e:
        print("\nОШИБКА получения данных:")
        print(e)

        print("\nОстанавливаем процесс.")
        print("Запустите reindex.py повторно после устранения проблемы.")

        sys.exit(1)


    if not rows:
        print("Записей больше нет.")
        break


    # --------------------------------------------------------
    # Разбиваем полученные записи на небольшие batch
    # --------------------------------------------------------

    for batch_start in range(0, len(rows), BATCH_SIZE):

        batch = rows[
            batch_start:
            batch_start + BATCH_SIZE
        ]

        valid_rows = []
        texts = []

        for row in batch:

            row_id = row.get("id")
            content = row.get("content")

            if not row_id:
                print("Пропуск записи без id.")
                skipped += 1
                continue

            if not content:
                print(
                    f"Пропуск {row_id}: пустой content."
                )
                skipped += 1
                continue

            content = str(content).strip()

            if not content:
                print(
                    f"Пропуск {row_id}: пустой content."
                )
                skipped += 1
                continue

            # Для E5 документы должны иметь prefix passage:
            texts.append(
                "passage: " + content
            )

            valid_rows.append(row)


        if not valid_rows:
            continue


        # ----------------------------------------------------
        # Генерация embeddings
        # ----------------------------------------------------

        try:

            embeddings = model.encode(
                texts,
                batch_size=BATCH_SIZE,
                normalize_embeddings=True,
                show_progress_bar=False
            )

        except Exception as e:

            print("\nОШИБКА генерации embedding:")
            print(e)

            errors += len(valid_rows)

            print(
                "\nПроцесс остановлен. "
                "Необходимо повторить запуск."
            )

            sys.exit(1)


        # ----------------------------------------------------
        # Проверка размерности
        # ----------------------------------------------------

        for embedding in embeddings:

            if len(embedding) != 768:

                print(
                    "\nКРИТИЧЕСКАЯ ОШИБКА:"
                )

                print(
                    f"Получена размерность {len(embedding)}, "
                    f"ожидалась 768."
                )

                sys.exit(1)


        # ----------------------------------------------------
        # Записываем embeddings в Supabase
        # ----------------------------------------------------

        for row, embedding in zip(valid_rows, embeddings):

            row_id = row["id"]

            try:

                vector = embedding.tolist()

                (
                    supabase
                    .table("npa_chunks")
                    .update({
                        "embedding": vector
                    })
                    .eq("id", row_id)
                    .execute()
                )

                updated += 1

            except Exception as e:

                errors += 1

                print(
                    f"\nОшибка обновления {row_id}:"
                )

                print(e)


        processed += len(batch)

        elapsed = time.time() - start_time

        if processed > 0:

            speed = processed / elapsed

            remaining = total - processed

            if speed > 0:
                eta = remaining / speed
            else:
                eta = 0

            print(
                f"Прогресс: "
                f"{processed}/{total} "
                f"({processed / total * 100:.1f}%) | "
                f"обновлено: {updated} | "
                f"пропущено: {skipped} | "
                f"ошибок: {errors} | "
                f"скорость: {speed:.1f}/сек | "
                f"осталось: {eta / 60:.1f} мин"
            )


# ============================================================
# ФИНАЛ
# ============================================================

elapsed = time.time() - start_time

print("\n")
print("=" * 70)
print("ПЕРЕИНДЕКСАЦИЯ ЗАВЕРШЕНА")
print("=" * 70)

print(f"Всего чанков:       {total}")
print(f"Обновлено:          {updated}")
print(f"Пропущено:          {skipped}")
print(f"Ошибок:             {errors}")
print(f"Время:              {elapsed / 60:.2f} мин")

print("=" * 70)

if errors > 0:

    print(
        "\nВНИМАНИЕ: были ошибки."
    )

    print(
        "Не запускайте бота до проверки базы."
    )

else:

    print(
        "\nВсе embeddings успешно обновлены."
    )

    print(
        "Теперь можно переходить к изменению bot.py."
    )
