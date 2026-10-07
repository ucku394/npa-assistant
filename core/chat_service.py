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
    def _grounding_check(
        answer: str,
        valid_source_ids: List[str],
    ) -> Dict[str, Any]:
        """Evidence Gate: legal claims must be traceable to returned RAG sources."""
        text = str(answer or "").strip()
        valid = {str(x).strip() for x in valid_source_ids if str(x).strip()}
        cited = ChatService._extract_source_ids(text)
        valid_cited = [source_id for source_id in cited if source_id in valid]

        legal_markers = (
            "обязан", "должен", "необходимо", "запрещ", "допуска", "вправе",
            "имеет право", "не реже", "не позднее", "срок", "периодич",
            "пункт", "статья", "ответствен",
        )
        legal_claims = sum(1 for marker in legal_markers if marker in text.lower())
        citation_ratio = len(valid_cited) / max(len(cited), 1)

        passed = bool(valid_cited) and citation_ratio >= 1.0
        if legal_claims >= 2 and not valid_cited:
            passed = False

        return {
            "passed": passed,
            "cited_source_ids": valid_cited,
            "unknown_source_ids": [source_id for source_id in cited if source_id not in valid],
            "legal_claim_markers": legal_claims,
            "citation_ratio": round(citation_ratio, 3),
        }

    @staticmethod
    def _claim_evidence_check(
        answer: str,
        evidence_map: List[Dict[str, Any]],
        query_profile: Dict[str, Any],
        topic: str = "",
    ) -> Dict[str, Any]:
        """Проверяет не только наличие цитаты, но и содержательную опору утверждения на НПА."""
        text = str(answer or "").strip().lower()
        qtype = str((query_profile or {}).get("question_type") or "general").strip().lower()

        cited_ids = set(ChatService._extract_source_ids(answer))
        cited_evidence = [
            item for item in evidence_map
            if str(item.get("source_id") or "").strip() in cited_ids
        ]

        frequency_markers = (
            "не реже", "не чаще", "один раз", "раза в", "раз в ",
            "каждые", "периодически", "срок", "не позднее", "до ",
            "в течение", "ежегодно", "ежемесячно", "ежеквартально",
        )
        responsibility_markers = (
            "ответствен", "отвечает", "ответственными являются",
            "обязан", "должен", "назнач",
        )
        prohibition_markers = (
            "запрещается", "запрещено", "не допускается", "не допускается",
        )
        permission_markers = (
            "допускается", "разрешается", "вправе", "имеет право",
        )

        excerpts = " ".join(
            str(item.get("excerpt") or "").lower()
            for item in cited_evidence
        )

        contradiction_absent = bool(
            re.search(
                r"(?:не\s+установлен\w*|не\s+определен\w*|"
                r"не\s+раскрыт\w*|отсутств\w*\s+(?:требован\w*|норм\w*|"
                r"периодичност\w*|срок\w*))",
                text,
            )
        )

        if qtype == "frequency":
            evidence_has_frequency = any(marker in excerpts for marker in frequency_markers)
            answer_has_frequency = any(marker in text for marker in frequency_markers)

            # Критическая защита: нельзя утверждать отсутствие периодичности,
            # если процитированный фрагмент содержит прямую временную норму.
            if contradiction_absent and evidence_has_frequency:
                return {
                    "passed": False,
                    "reason": "answer_denies_frequency_but_cited_evidence_contains_frequency",
                    "question_type": qtype,
                    "cited_evidence_count": len(cited_evidence),
                }

            # Если ассистент сообщает периодичность, она должна быть подтверждена
            # хотя бы одним процитированным фрагментом.
            if answer_has_frequency and not evidence_has_frequency:
                return {
                    "passed": False,
                    "reason": "frequency_claim_has_no_temporal_marker_in_cited_evidence",
                    "question_type": qtype,
                    "cited_evidence_count": len(cited_evidence),
                }

        elif qtype in {"who", "responsibility"}:
            evidence_has_actor = any(marker in excerpts for marker in responsibility_markers)
            if evidence_has_actor and contradiction_absent:
                # Отрицательные ответы о наличии ответственного лица также
                # запрещены, когда цитата прямо содержит норму об ответственности.
                return {
                    "passed": False,
                    "reason": "answer_denies_responsibility_but_cited_evidence_contains_actor",
                    "question_type": qtype,
                    "cited_evidence_count": len(cited_evidence),
                }

        elif qtype == "whether":
            evidence_has_legal_rule = any(
                marker in excerpts
                for marker in prohibition_markers + permission_markers
            )
            if contradiction_absent and evidence_has_legal_rule:
                return {
                    "passed": False,
                    "reason": "answer_denies_rule_but_cited_evidence_contains_permission_or_prohibition",
                    "question_type": qtype,
                    "cited_evidence_count": len(cited_evidence),
                }

        return {
            "passed": True,
            "reason": "supported_or_not_applicable",
            "question_type": qtype,
            "cited_evidence_count": len(cited_evidence),
            "topic": topic,
        }

    @staticmethod
    def _build_grounding_retry_prompt(
        base_prompt: str,
        reason: Dict[str, Any],
    ) -> str:
        return (
            base_prompt
            + "\\n\\n============================================================\\n"
            + "СТРОГИЙ РЕЖИМ ДОКАЗАТЕЛЬНОСТИ\\n"
            + "============================================================\\n"
            + "Предыдущий вариант не прошёл Evidence Gate. Переформулируй ответ. "
            + "Каждое юридически значимое утверждение должно иметь [SOURCE:SOURCE_ID]. "
            + "Используй только SOURCE_ID из EVIDENCE MAP и фактический текст RAG. "
            + "Не добавляй нормы по памяти. Если доказательств недостаточно, прямо укажи, "
            + "что в предоставленных фрагментах НПА требование не раскрыто.\\n"
            + f"Причина проверки: {reason}"
        )

    @staticmethod
    def _format_telegram_answer(
        answer: str,
        sources: List[Dict[str, Any]],
    ) -> str:
        """Компактный формат ответа для Telegram без потери юридических ссылок."""
        result = str(answer or "").strip()
        result = re.sub(r"\n{3,}", "\n\n", result)
        result = re.sub(r"[ \t]+", " ", result)
        result = re.sub(r"\n +", "\n", result)

        if sources:
            source_lines = []
            for source in sources[:6]:
                document = str(source.get("document") or "Неизвестный НПА").strip()
                point = str(source.get("point") or "").strip()
                source_lines.append(
                    f"• {document}" + (f" — пункт/статья {point}" if point else "")
                )
            result += "\n\n📎 <b>Источники</b>\n" + "\n".join(source_lines)
        return result.strip()

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
            "query_profile": rag_result.get("query_profile") or {},
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

        grounding = self._grounding_check(answer, valid_source_ids)
        claim_evidence = self._claim_evidence_check(
            answer=answer,
            evidence_map=evidence_map,
            query_profile=rag_result.get("query_profile") or {},
            topic=str(rag_result.get("topic") or ""),
        )

        if not grounding["passed"] or not claim_evidence["passed"]:
            retry_reason = {
                "grounding": grounding,
                "claim_evidence": claim_evidence,
            }
            retry_prompt = self._build_grounding_retry_prompt(prompt, retry_reason)
            retry_prompt += (
                "\\n\\nДОПОЛНИТЕЛЬНОЕ ПРАВИЛО ПРОВЕРКИ СОДЕРЖАНИЯ:\\n"
                "Если в EVIDENCE MAP есть прямая норма по вопросу, нельзя утверждать, "
                "что такая норма отсутствует. Для вопросов о периодичности укажи только "
                "срок/периодичность, которая прямо следует из процитированного фрагмента. "
                "Для вопросов о том, кто отвечает, укажи только лицо/субъект, прямо "
                "названный в норме. Для вопросов 'можно ли' различай прямой запрет и "
                "прямое разрешение.\\n"
            )
            retry_answer = await asyncio.to_thread(generate_answer, retry_prompt)
            if retry_answer:
                answer = str(retry_answer).strip()
                grounding = self._grounding_check(answer, valid_source_ids)
                claim_evidence = self._claim_evidence_check(
                    answer=answer,
                    evidence_map=evidence_map,
                    query_profile=rag_result.get("query_profile") or {},
                    topic=str(rag_result.get("topic") or ""),
                )

        rag_meta["grounding"] = grounding
        rag_meta["claim_evidence"] = claim_evidence

        if not grounding["passed"] or not claim_evidence["passed"]:
            logger.warning(
                "LEGAL | Evidence Gate failed | cited=%s | unknown=%s | claims=%s",
                grounding.get("cited_source_ids"),
                grounding.get("unknown_source_ids"),
                grounding.get("legal_claim_markers"),
            )
            return {
                "success": False,
                "error": "grounding_failed",
                "question": question,
                "answer": "",
                "sources": [],
                "rag": {**rag_meta, "grounding": grounding, "claim_evidence": claim_evidence},
                "evidence_map": evidence_map,
                "evidence_map_text": evidence_map_text,
            }

        used_source_ids = self._extract_source_ids(answer)
        sources = self._build_sources(chunks, used_source_ids)
        answer_with_citations = self._build_citation_answer(
            answer=answer,
            sources=sources,
        )
        telegram_answer = self._format_telegram_answer(answer_with_citations, sources)

        answer = self._remove_source_markers(answer, valid_source_ids)

        return {
            "success": True,
            "error": None,
            "question": question,
            "answer": answer,
            "answer_with_citations": answer_with_citations,
            "telegram_answer": telegram_answer,
            "sources": sources,
            "rag": rag_meta,
            "evidence_map": evidence_map,
            "evidence_map_text": evidence_map_text,
        }


chat_service = ChatService()
