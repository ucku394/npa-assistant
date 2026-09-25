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

        patterns = (
            r"\[SOURCE:([A-Za-zА-Яа-яЁё0-9_./-]+)\]",
            r"\bNPA_[A-Za-zА-Яа-яЁё0-9_./-]+\b",
        )

        for pattern in patterns:
            for source_id in re.findall(pattern, answer):
                source_id = source_id.strip()
                if source_id and source_id not in seen:
                    seen.add(source_id)
                    result.append(source_id)

        return result

    @staticmethod
    def _build_sources(chunks: List[Dict[str, Any]], used_ids: List[str]) -> List[Dict[str, str]]:
        if not chunks or not used_ids:
            return []

        used = {str(x).strip() for x in used_ids if x}
        sources: List[Dict[str, str]] = []
        seen = set()

        for index, chunk in enumerate(chunks, start=1):
            source_id = chunk.get("_source_id") or build_source_id(chunk, index)
            if not source_id:
                continue

            source_id = str(source_id).strip()
            if source_id not in used:
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

            item_key = (source_id, str(document).strip(), str(point).strip())
            if item_key in seen:
                continue
            seen.add(item_key)

            sources.append(
                {
                    "source_id": source_id,
                    "document": str(document).strip(),
                    "point": str(point).strip(),
                }
            )

        return sources

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
        }

        if not rag_meta["found"]:
            return {
                "success": False,
                "error": "no_relevant_context",
                "question": question,
                "answer": "",
                "sources": [],
                "rag": rag_meta,
            }

        prompt = LEGAL_ASSISTANT_PROMPT.format(
            retrieved_text=npa_context,
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

        answer = self._remove_source_markers(answer, valid_source_ids)

        return {
            "success": True,
            "error": None,
            "question": question,
            "answer": answer,
            "sources": sources,
            "rag": rag_meta,
        }


chat_service = ChatService()
