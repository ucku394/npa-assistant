# Ассистент по ОТ и ПБ Республики Беларусь

RAG-ассистент по охране труда, пожарной и промышленной безопасности: отвечает на вопросы по НПА РБ со ссылками на конкретные пункты, проверяет фотографии на нарушения, формирует проекты предписаний, акта Н-1 и наряда-допуска.

- **Telegram-бот** (`bot.py`) и **Web API + UI** (`api.py`, `web/`) используют общий `core/chat_service.py`.
- Поиск: Supabase (pgvector + полнотекстовый), эмбеддинги `multilingual-e5` (ONNX/PyTorch), cross-encoder reranker.
- Генерация: Gemini → OpenRouter → DeepSeek (fallback с cooldown), проверка SOURCE_ID и Evidence Gate.

Структура проекта описана в [ARCHITECTURE.md](ARCHITECTURE.md).

## Локальный запуск

```bash
python -m venv .venv && . .venv/bin/activate
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt pytest pre-commit
pre-commit install
cp .env.example .env   # заполнить ключи
python run.py          # Web API (http://localhost:8000) + бот, если задан TELEGRAM_BOT_TOKEN
```

Только бот: `python bot.py`. Только API: `uvicorn api:app --port 8000`.

## Проверки

```bash
pytest -q                 # тесты (без сети)
ruff check .              # линтер
pre-commit run --all-files
python evaluation/rag_eval.py evaluation/questions.json   # оценка RAG (нужен доступ к Supabase и LLM)
```

## Деплой

Railway / Docker: `Dockerfile` и `Procfile` запускают `python run.py`, то есть API и бота в одном процессе. **Не запускайте `bot.py` отдельно рядом с `run.py`**: два процесса с polling одного токена получают от Telegram `409 Conflict`.

Миграции БД лежат в `supabase/migrations/`, применяются по порядку.

## Доступ и безопасность

| Переменная | По умолчанию | Назначение |
|---|---|---|
| `WEB_API_KEY` | пусто (API открыт) | Общий ключ для `/api/*`. Web UI один раз запрашивает его и хранит в браузере; запросы передают его в заголовке `X-API-Key` или `Authorization: Bearer`. |
| `WEB_RATE_LIMIT_PER_MINUTE` | 20 | Лимит запросов с одного IP (0 — без лимита). |
| `WEB_MAX_UPLOAD_MB` | 12 | Максимальный размер фото. |
| `WEB_CORS_ORIGINS` | пусто | Разрешённые сторонние origin (по умолчанию только same-origin). |
| `FORWARDED_ALLOW_IPS` | `*` | Каким прокси доверять `X-Forwarded-For` (нужно для лимита по IP за прокси Railway). |
| `TELEGRAM_ALLOWED_USER_IDS` | пусто (все) | Список разрешённых Telegram user ID через запятую. |
| `TELEGRAM_RATE_LIMIT_PER_MINUTE` | 10 | Лимит сообщений от одного пользователя (`/start` не учитывается). |

Остальные переменные перечислены в [.env.example](.env.example). Не коммитьте `.env` и реальные ключи: в CI и pre-commit работает gitleaks.
