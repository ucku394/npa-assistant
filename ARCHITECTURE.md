# Project structure

The runtime entry points remain at the repository root (bot.py, api.py) so the existing Railway/Procfile startup commands do not change.

- rag_engine/ — retrieval, query classification and expansion, legal target discovery, and RAG orchestration.
- services/ — AI provider routing, photo inspection, vision analysis, and prescription document generation.
- scripts/ — operational and maintenance commands (NPA corpus audit, document deletion, retrieval evaluation, reindexing, upload, embedding check).
- core/ — transport-neutral chat orchestration.
- tests/ — automated regression tests.
- evaluation/ — retrieval evaluation datasets and benchmark runner.
- supabase/migrations/ — database migrations.

## Compatibility

Legacy root-level imports and script names are retained as thin compatibility shims. Existing commands such as `python upload.py` and imports such as `from rag import retrieve_context` continue to route to the package implementation. New code should import from `rag_engine` or `services` directly.

## Common commands

- Run bot: `python bot.py`
- Run web API: `uvicorn api:app --host 0.0.0.0 --port $PORT`
- Run tests: `pytest -q`
- Evaluate retrieval: `python -m scripts.evaluate_rag_retrieval`
- Upload/update NPA documents: `python upload.py` (compatibility entry point)
