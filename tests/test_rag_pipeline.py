import asyncio
from rag_engine.pipeline import RagPipeline, RagPipelineDependencies

def test_pipeline_orchestration_keeps_result_shape():
    def embeddings(queries):
        return [[0.0] * 384 for _ in queries]

    def search(*args):
        return [{
            "doc_name": "НПА №1",
            "point_num": "11",
            "content": "Норма по вопросу СИЗ",
            "legal_domain": "occupational_safety",
            "topic": "ppe_nonprovision",
            "similarity": 0.9,
        }]

    def merge(groups, roles):
        result, seen = [], set()
        for group in groups:
            for item in group:
                key = (item.get("doc_name"), item.get("point_num"))
                if key not in seen:
                    seen.add(key)
                    result.append(item)
        return result

    def score(*args, **kwargs):
        return 1.0

    def select(chunks, *args, **kwargs):
        return chunks[:5]

    async def targeted(*args):
        return []

    deps = RagPipelineDependencies(
        detect_legal_domain=lambda q: "occupational_safety",
        detect_topic=lambda q: "ppe_nonprovision",
        extract_query_terms=lambda q: ["сиз"],
        detect_query_intents=lambda q: ["employee_right"],
        is_cross_reference_query=lambda i: False,
        detect_primary_intent=lambda i, q: "employee_right",
        is_labor_code_query=lambda q: False,
        detect_special_category=lambda q: None,
        minor_special_issue=lambda q: None,
        build_query_profile=lambda q: {"question_type": "whether"},
        build_search_queries=lambda q, t, d, i: [q],
        get_query_embeddings=embeddings,
        search_chunks=search,
        merge_search_results=merge,
        get_targeted_chunks=targeted,
        legal_relevance_score=score,
        select_legal_diverse_chunks=select,
        accident_query_mode=lambda q: None,
        build_source_id=lambda c, i: "NPA_1_P11",
        get_document_name=lambda c: c["doc_name"],
        get_point_number=lambda c: c["point_num"],
        build_retrieved_text=lambda c: "TEXT",
        safe_float=lambda v, default=0.0: float(v or default),
        final_count=5,
        candidate_count=50,
    )

    result = asyncio.run(
        RagPipeline(deps).retrieve(
            "Можно ли отказаться от работы без СИЗ?", object()
        )
    )
    assert result["found"] is True
    assert result["final_count"] == 1
    assert result["chunks"][0]["_source_id"] == "NPA_1_P11"
