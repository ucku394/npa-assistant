# RAG Evaluation Framework

The corpus contains **240 curated/production-like Belarus OHS/PB question variants** across 12 domains. They are regression cases, not a dump of actual production logs.

## Gold labels

`gold_source_ids` now contain a **document-level gold baseline** derived from the current Supabase corpus.

`gold_chunk_ids` are intentionally more conservative. Exact chunk labels have been manually verified for the **knowledge-testing** domain first, because this is the area currently under active RAG refactoring.

Important: gold labels are evidence expectations, not generated from the RAG result itself. They should be reviewed when an NPA edition is replaced.

## Offline evaluation

```bash
python -m pytest -q tests
python -m eval.evaluate_rag
```

The offline evaluator measures:

- topic accuracy
- legal-query coverage
- knowledge-testing → briefing leakage
- briefing → knowledge-testing leakage

## Live retrieval evaluation

`eval/evaluate_retrieval.py` runs the actual retrieval pipeline against the live Supabase corpus and does **not** call Gemini/OpenRouter. This isolates retrieval quality from answer-generation quality.

Required environment:

```text
SUPABASE_URL
SUPABASE_SERVICE_ROLE_KEY
```

Run:

```bash
RAG_EVAL_LIMIT=240 python -m eval.evaluate_retrieval
```

or start the manual GitHub Actions workflow: **Live RAG Retrieval Evaluation**.

Metrics:

- Recall@5
- Recall@10
- MRR
- Precision@5
- wrong-domain retrieval rate @10
- forbidden-source rate @10
- evidence sufficiency rate

The report also includes per-topic metrics and failed cases.

### Why source-level gold comes first

The assistant's `SOURCE_ID` is document/point based, for example `NPA_175_P54`. Document-level gold remains stable when chunk UUIDs are regenerated during an NPA replacement. Exact `gold_chunk_ids` can then be added to high-value cases after manual verification.

## Next evaluation layer

After retrieval metrics are stable, add:

1. `SOURCE_ID` accuracy against exact chunk golds.
2. answer-support evaluation.
3. unsupported-answer rate.
4. contradiction detection.
5. temporal/current-edition validation.
