# RAG Evaluation Framework

## Наборы

\`dataset.py\` содержит 24 сценария x 10 контролируемых формулировок = 240 regression cases.

Реальные вопросы хранятся отдельно в \`eval/data/production.jsonl\` после обезличивания и ручной разметки.

Формат строки JSONL:
\`{"id":"prod-0001","question":"...","expected_documents":["..."],"expected_points":["..."],"forbidden_documents":[],"forbidden_phrases":[],"tags":["ppe"]}\`

## Метрики

- Recall@3 / Recall@5
- MRR
- Precision@5
- forbidden_rate
- empty_rate
- by_tag

## Запуск

\`python -m eval.run_eval --limit 240\`

\`python -m eval.run_eval --dataset eval/data/production.jsonl --json reports/production_eval.json\`

Regression gate:

\`python -m eval.run_eval --dataset eval/data/production.jsonl --baseline eval/baseline.json --max-regression 0.02\`

Если baseline отсутствует, текущий результат просто сохраняется. После первого стабильного запуска baseline фиксируется и дальше используется как regression guard.

## Получение 200-300 реальных вопросов

1. Экспортировать обезличенные Telegram-вопросы.
2. Удалить ФИО, телефоны, названия конкретных работников и другую PII.
3. Дедуплицировать почти одинаковые вопросы.
4. Разметить правильный НПА и пункт/статью.
5. Для конфликтных случаев заполнить forbidden_*.
6. Отделить immutable evaluation set от набора, на котором настраивается scoring.

Синтетические 240 кейсов не выдаются за реальные production queries.
