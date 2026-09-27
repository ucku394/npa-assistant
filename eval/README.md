# RAG Evaluation Framework

The corpus contains 240 curated Belarus OHS/PB question variants across 12 domains. It is a regression corpus separate from unit tests.

Metrics: topic accuracy, legal-query coverage, knowledge-testing to briefing leakage, and briefing to knowledge-testing leakage.

Every case already contains empty gold_source_ids and gold_chunk_ids. After mapping questions to actual Supabase NPA chunks, the evaluator can add Recall@5/10, MRR, Precision@5, wrong-domain retrieval, forbidden-source rate, evidence sufficiency, SOURCE_ID accuracy and unsupported-answer rate.

Run: python -m pytest -q tests

Evaluation: python -m eval.evaluate_rag
