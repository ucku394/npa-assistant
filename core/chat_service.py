"""Shared text-chat service for Telegram and the future web API.

This module contains only transport-neutral orchestration:
question -> RAG -> legal prompt -> AI -> structured result.

Telegram formatting, typing indicators and photo analysis stay in bot.py.
"""

import asyncio
import logging
import re
from typing import Any, Dict, List

from supabase import create_client

from ai_router import generate_answer
from config import SUPABASE_SERVICE_ROLE_KEY, SUPABASE_URL
from prompts import LEGAL_ASSISTANT_PROMPT
from rag import build_source_id, retrieve_context

logger = logging.getLogger(__name__)


class ChatService:
    """Common text-question pipeline used by Telegram and Web."""

    def __init__(self, supabase_client=None):
        self.supabase = supabase_client or create_client(
            SUPABASE_URL,
            SUPABASE_SERVICE_ROLE_KEY,
        )

    @staticmethod
    def _extract_source_ids(answer: str) -> List[str]:
        if not answer:
            return []

        result: List[str] = []
        seen = set()

        pattern = (
            r"\[SOURCE:([A-Za-zА-Яа-яЁё0-9_./-]+)\]"
            r"|\b(NPA_[A-Za-zА-Яа-яЁё0-9_./-]+)\b"
        )

        for match in re.finditer(pattern, answer):
            source_id = (match.group(1) or match.group(2) or "").strip()
            if source_id and source_id not in seen:
                seen.add(source_id)
                result.append(source_id)

        return result

    @staticmethod
    def _build_sources(chunks: List[Dict[str, Any]], used_ids: List[str]) -> List[Dict[str, Any]]:
        """Build sources in the same order in which the AI cited them.

        This keeps the web UI's source numbering meaningful: Source 1 is the
        first validated SOURCE_ID mentioned by the model, not merely the first
        chunk returned by RAG.
        """
        if not chunks or not used_ids:
            return []

        chunk_by_source_id: Dict[str, Dict[str, Any]] = {}
        for index, chunk in enumerate(chunks, start=1):
            source_id = chunk.get("_source_id") or build_source_id(chunk, index)
            if not source_id:
                continue
            source_id = str(source_id).strip()
            chunk["_source_id"] = source_id
            chunk_by_source_id.setdefault(source_id, chunk)

        sources: List[Dict[str, Any]] = []
        seen = set()

        for citation_order, raw_source_id in enumerate(used_ids, start=1):
            source_id = str(raw_source_id or "").strip()
            if not source_id or source_id in seen:
                continue

            chunk = chunk_by_source_id.get(source_id)
            if not chunk:
                continue

            document = (
                chunk.get("document")
                or chunk.get("document_name")
                or chunk.get("doc_name")
                or chunk.get("title")
                or chunk.get("npa_name")
                or chunk.get("source")
                or "Неизвестный НПА"
            )
            point = (
                chunk.get("point_num")
                or chunk.get("point")
                or chunk.get("point_number")
                or chunk.get("article")
                or chunk.get("article_number")
                or chunk.get("paragraph")
                or chunk.get("section")
                or ""
            )
            source_url = chunk.get("source_url") or chunk.get("url") or ""

            item_key = (source_id, str(document).strip(), str(point).strip())
            if item_key in seen:
                continue
            seen.add(item_key)

            sources.append(
                {
                    "source_id": source_id,
                    "document": str(document).strip(),
                    "point": str(point).strip(),
                    "source_url": str(source_url).strip(),
                    "citation_order": citation_order,
                }
            )

        return sources

    @staticmethod
    def _build_citation_answer(
        answer: str,
        sources: List[Dict[str, Any]],
    ) -> str:
        """Replace validated SOURCE markers with compact [1], [2] citations."""
        result = str(answer or "")
        order_by_source_id = {
            str(source.get("source_id")).strip(): int(source.get("citation_order", index))
            for index, source in enumerate(sources, start=1)
            if source.get("source_id")
        }

        def replace_marker(match):
            source_id = match.group(1).strip()
            order = order_by_source_id.get(source_id)
            return f"[{order}]" if order else ""

        result = re.sub(
            r"\[SOURCE:([A-Za-zА-Яа-яЁё0-9_./-]+)\]",
            replace_marker,
            result,
        )

        for source_id, order in sorted(
            order_by_source_id.items(),
            key=lambda item: len(item[0]),
            reverse=True,
        ):
            result = re.sub(
                rf"(?<![A-Za-zА-Яа-яЁё0-9_]){re.escape(source_id)}"
                rf"(?![A-Za-zА-Яа-яЁё0-9_])",
                f"[{order}]",
                result,
            )

        result = re.sub(r"[ \t]+([,.;:])", r"\1", result)
        result = re.sub(r"[ \t]+\n", "\n", result)
        result = re.sub(r"\n{3,}", "\n\n", result)
        return result.strip()

    @staticmethod
    def _remove_source_markers(answer: str, valid_ids: List[str]) -> str:
        answer = str(answer or "")

        answer = re.sub(
            r"\[SOURCE:[A-Za-zА-Яа-яЁё0-9_./-]+\]",
            "",
            answer,
        )

        for source_id in sorted(
            {str(x).strip() for x in valid_ids if x},
            key=len,
            reverse=True,
        ):
            answer = re.sub(
                rf"(?<![A-Za-zА-Яа-яЁё0-9_]){re.escape(source_id)}"
                rf"(?![A-Za-zА-Яа-яЁё0-9_])",
                "",
                answer,
            )

        answer = re.sub(r"[ \t]+([,.;:])", r"\1", answer)
        answer = re.sub(r"[ \t]+\n", "\n", answer)
        answer = re.sub(r"\n{3,}", "\n\n", answer)
        return answer.strip()

    @staticmethod
    def _build_evidence_map(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build a compact evidence map from the exact RAG chunks."""
        evidence = []
        for index, chunk in enumerate(chunks, start=1):
            source_id = str(
                chunk.get("_source_id") or build_source_id(chunk, index) or ""
            ).strip()
            if not source_id:
                continue

            document = str(
                chunk.get("document")
                or chunk.get("document_name")
                or chunk.get("doc_name")
                or chunk.get("title")
                or chunk.get("npa_name")
                or chunk.get("source")
                or "Неизвестный НПА"
            ).strip()
            point = str(
                chunk.get("point_num")
                or chunk.get("point")
                or chunk.get("point_number")
                or chunk.get("article")
                or chunk.get("article_number")
                or chunk.get("paragraph")
                or chunk.get("section")
                or ""
            ).strip()
            text = str(
                chunk.get("content")
                or chunk.get("text")
                or chunk.get("chunk_text")
                or ""
            ).strip()

            evidence.append({
                "index": index,
                "source_id": source_id,
                "document": document,
                "point": point,
                "legal_domain": str(chunk.get("legal_domain") or "").strip(),
                "topic": str(chunk.get("topic") or "").strip(),
                "source_url": str(
                    chunk.get("source_url") or chunk.get("url") or ""
                ).strip(),
                "excerpt": text[:700],
            })
        return evidence

    @staticmethod
    def _format_evidence_map(evidence: List[Dict[str, Any]]) -> str:
        if not evidence:
            return "EVIDENCE MAP: отсутствует."

        blocks = ["EVIDENCE MAP:"]
        for item in evidence:
            blocks.append(
                f"[ИСТОЧНИК {item['index']}] "
                f"SOURCE_ID={item['source_id']}\n"
                f"НПА: {item['document']}\n"
                f"Пункт/статья: {item['point'] or 'не указан'}\n"
                f"Область: {item['legal_domain'] or 'не указана'}\n"
                f"Тема: {item['topic'] or 'не указана'}\n"
                f"Фрагмент: {item['excerpt']}"
            )
        return "\n\n".join(blocks)

    async def process_text(self, question: str) -> Dict[str, Any]:
        """Process one legal text question and return a transport-neutral result."""
        question = str(question or "").strip()
        if not question:
            return {
                "success": False,
                "error": "empty_question",
                "question": "",
                "answer": "",
                "sources": [],
                "rag": {},
            }

        rag_result = await retrieve_context(question, self.supabase)

        chunks = rag_result.get("chunks") or []
        npa_context = rag_result.get("retrieved_text") or ""

        evidence_map = self._build_evidence_map(chunks)
        evidence_map_text = self._format_evidence_map(evidence_map)

        logger.info(
            "CHAT | RAG candidates=%s | final=%s | domain=%s | topic=%s",
            rag_result.get("candidate_count"),
            rag_result.get("final_count"),
            rag_result.get("legal_domain"),
            rag_result.get("topic"),
        )

        rag_meta = {
            "found": bool(rag_result.get("found") and npa_context),
            "candidate_count": rag_result.get("candidate_count", 0),
            "final_count": rag_result.get("final_count", 0),
            "legal_domain": rag_result.get("legal_domain"),
            "topic": rag_result.get("topic"),
            "evidence_count": len(evidence_map),
        }

        if not rag_meta["found"]:
            return {
                "success": False,
                "error": "no_relevant_context",
                "question": question,
                "answer": "",
                "sources": [],
                "rag": rag_meta,
                "evidence_map": evidence_map,
                "evidence_map_text": evidence_map_text,
            }

        prompt = LEGAL_ASSISTANT_PROMPT.format(
            retrieved_text=npa_context,
            evidence_map=evidence_map_text,
            user_query=question,
        )

        answer = await asyncio.to_thread(generate_answer, prompt)
        if not answer:
            raise RuntimeError("AI returned empty answer.")

        answer = str(answer).strip()

        valid_source_ids: List[str] = []
        for index, chunk in enumerate(chunks, start=1):
            source_id = chunk.get("_source_id") or build_source_id(chunk, index)
            if source_id:
                chunk["_source_id"] = source_id
                valid_source_ids.append(str(source_id))

        used_source_ids = self._extract_source_ids(answer)
        sources = self._build_sources(chunks, used_source_ids)
        answer_with_citations = self._build_citation_answer(
            answer=answer,
            sources=sources,
        )

        answer = self._remove_source_markers(answer, valid_source_ids)

        return {
            "success": True,
            "error": None,
            "question": question,
            "answer": answer,
            "answer_with_citations": answer_with_citations,
            "sources": sources,
            "rag": rag_meta,
            "evidence_map": evidence_map,
            "evidence_map_text": evidence_map_text,
        }


chat_service = ChatService()
