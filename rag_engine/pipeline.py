from __future__ import annotations
import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class RagPipelineDependencies:
    detect_legal_domain: Callable[[str], str]
    detect_topic: Callable[[str], str]
    extract_query_terms: Callable[[str], List[str]]
    detect_query_intents: Callable[[str], List[str]]
    is_cross_reference_query: Callable[[List[str]], bool]
    detect_primary_intent: Callable[[List[str], str], Optional[str]]
    is_labor_code_query: Callable[[str], bool]
    detect_special_category: Callable[[str], Optional[str]]
    minor_special_issue: Callable[[str], Optional[str]]
    build_query_profile: Callable[[str], Dict[str, Any]]
    build_search_queries: Callable[[str, str, str, List[str]], List[str]]
    get_query_embeddings: Callable[[List[str]], List[List[float]]]
    search_chunks: Callable[..., List[Dict[str, Any]]]
    merge_search_results: Callable[[List[List[Dict[str, Any]]], List[str]], List[Dict[str, Any]]]
    get_targeted_chunks: Callable[..., Awaitable[List[Dict[str, Any]]]]
    legal_relevance_score: Callable[..., float]
    select_legal_diverse_chunks: Callable[..., List[Dict[str, Any]]]
    accident_query_mode: Callable[[str], Optional[str]]
    build_source_id: Callable[[Dict[str, Any], int], str]
    get_document_name: Callable[[Dict[str, Any]], str]
    get_point_number: Callable[[Dict[str, Any]], str]
    build_retrieved_text: Callable[[List[Dict[str, Any]]], str]
    safe_float: Callable[[Any, float], float]
    final_count: int
    candidate_count: int

class RagPipeline:
    """Orchestrates retrieval while legal heuristics stay outside the pipeline."""

    def __init__(self, d: RagPipelineDependencies):
        self.d = d

    async def retrieve(self, user_query: str, supabase) -> Dict[str, Any]:
        query = str(user_query or "").strip()
        d = self.d
        if not query:
            return self._empty("", "general", [], False)

        legal_domain = d.detect_legal_domain(query)
        topic = d.detect_topic(query)
        query_terms = d.extract_query_terms(query)
        intents = d.detect_query_intents(query)
        cross_reference = d.is_cross_reference_query(intents)
        primary_intent = d.detect_primary_intent(intents, query)
        labor_code_query = d.is_labor_code_query(query)
        special_category = d.detect_special_category(query)
        special_issue = d.minor_special_issue(query) if special_category == "minor" else None
        query_profile = d.build_query_profile(query)
        search_queries = d.build_search_queries(query, topic, legal_domain, intents)

        logger.info(
            "RAG_PIPELINE classify domain=%s topic=%s intents=%s primary=%s cross_reference=%s queries=%s",
            legal_domain, topic, intents, primary_intent, cross_reference, len(search_queries),
        )

        query_vectors = await asyncio.to_thread(d.get_query_embeddings, search_queries)
        valid_pairs = [(q, v) for q, v in zip(search_queries, query_vectors) if v]
        if not valid_pairs:
            logger.warning("RAG_PIPELINE no valid embeddings")
            return self._empty(legal_domain, topic, intents, cross_reference,
                               query_profile=query_profile, primary_intent=primary_intent)

        valid_queries = [q for q, _ in valid_pairs]
        vectors = [v for _, v in valid_pairs]
        dimensions = sorted(set(len(v) for v in vectors))
        if dimensions != [384]:
            raise ValueError(f"Unexpected embedding dimensions: {dimensions}. Expected [384].")

        async def run_search(vector, search_query, remove_domain_filter=False):
            return await asyncio.to_thread(
                d.search_chunks, supabase, vector, search_query,
                None if remove_domain_filter else legal_domain, None,
            )

        raw_groups = await asyncio.gather(
            *[
                run_search(v, q, remove_domain_filter=(cross_reference and i == 0))
                for i, (q, v) in enumerate(valid_pairs)
            ],
            return_exceptions=True,
        )

        groups = []
        for i, result in enumerate(raw_groups):
            if isinstance(result, Exception):
                logger.warning("RAG_PIPELINE search failed query=%r error=%s", valid_queries[i], result)
                groups.append([])
            else:
                groups.append(result or [])

        roles = ["main" if i == 0 else "expanded" for i in range(len(groups))]
        candidate_chunks = d.merge_search_results(groups, roles)

        targeted = await d.get_targeted_chunks(supabase, topic, query)
        if targeted:
            accident_mode = d.accident_query_mode(query) if topic == "accident_investigation" else None
            if accident_mode == "worker_did_not_report":
                for chunk in targeted:
                    chunk["_accident_worker_not_report_targeted"] = True
            targeted_merged = d.merge_search_results([targeted], ["targeted"])
            candidate_chunks = d.merge_search_results(
                [candidate_chunks, targeted_merged], ["merged", "targeted"]
            )

        candidate_count = len(candidate_chunks)
        if not candidate_chunks:
            return self._empty(legal_domain, topic, intents, cross_reference,
                               query_profile=query_profile, primary_intent=primary_intent)

        domain_specific_count = sum(c.get("legal_domain") == legal_domain for c in candidate_chunks)
        topic_specific_count = sum(c.get("topic") == topic for c in candidate_chunks)

        for chunk in candidate_chunks:
            chunk["_user_query_for_scoring"] = query
            chunk["_combined_score"] = d.legal_relevance_score(
                chunk, query_terms, topic, intents, cross_reference,
                primary_intent=primary_intent, labor_code_query=labor_code_query,
                user_query=query, special_category=special_category,
                special_issue=special_issue, query_profile=query_profile,
            )

        ranked = sorted(
            candidate_chunks,
            key=lambda c: d.safe_float(c.get("_combined_score")),
            reverse=True,
        )

        final_limit = max(d.final_count, 7) if cross_reference else d.final_count
        accident_mode = d.accident_query_mode(query) if topic == "accident_investigation" else None
        final_chunks = d.select_legal_diverse_chunks(
            ranked, final_limit, topic, intents, cross_reference,
            primary_intent=primary_intent, labor_code_query=labor_code_query,
            special_category=special_category, special_issue=special_issue,
            accident_mode=accident_mode, query_profile=query_profile, user_query=query,
        )

        source_references = []
        for index, chunk in enumerate(final_chunks, start=1):
            source_id = d.build_source_id(chunk, index)
            chunk["_source_id"] = source_id
            document_name = d.get_document_name(chunk)
            point = d.get_point_number(chunk)
            source_references.append({
                "source_id": source_id,
                "reference": f"{document_name} — пункт/статья {point}" if point else document_name,
            })

        return {
            "chunks": final_chunks,
            "retrieved_text": d.build_retrieved_text(final_chunks),
            "found": bool(final_chunks),
            "candidate_count": candidate_count,
            "final_count": len(final_chunks),
            "source_references": source_references,
            "legal_domain": legal_domain,
            "topic": topic,
            "intents": intents,
            "primary_intent": primary_intent,
            "query_profile": query_profile,
            "cross_reference": cross_reference,
            "domain_specific_count": domain_specific_count,
            "topic_specific_count": topic_specific_count,
        }

    @staticmethod
    def _empty(legal_domain, topic, intents, cross_reference, *, query_profile=None, primary_intent=None):
        return {
            "chunks": [], "retrieved_text": "", "found": False,
            "candidate_count": 0, "final_count": 0, "source_references": [],
            "legal_domain": legal_domain, "topic": topic, "intents": intents,
            "primary_intent": primary_intent, "query_profile": query_profile or {},
            "cross_reference": cross_reference, "domain_specific_count": 0,
            "topic_specific_count": 0,
        }
