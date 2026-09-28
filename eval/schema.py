from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class EvalCase:
    id: str
    question: str
    expected_documents: List[str] = field(default_factory=list)
    expected_points: List[str] = field(default_factory=list)
    forbidden_documents: List[str] = field(default_factory=list)
    forbidden_phrases: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)


def _match(value: str, patterns: List[str]) -> bool:
    value = value.lower()
    return any(p.lower() in value for p in patterns)


def chunk_document_matches(case: EvalCase, chunk: dict) -> bool:
    document = str(chunk.get("doc_name") or chunk.get("document") or "")
    return bool(case.expected_documents) and _match(document, case.expected_documents)


def chunk_point_matches(case: EvalCase, chunk: dict) -> bool:
    point = str(chunk.get("point_num") or chunk.get("point") or chunk.get("article") or "")
    return bool(case.expected_points) and _match(point, case.expected_points)


def chunk_matches(case: EvalCase, chunk: dict) -> bool:
    document_ok = chunk_document_matches(case, chunk) if case.expected_documents else True
    point_ok = chunk_point_matches(case, chunk) if case.expected_points else True
    return bool(case.expected_documents or case.expected_points) and document_ok and point_ok


def chunk_is_forbidden(case: EvalCase, chunk: dict) -> bool:
    document = str(chunk.get("doc_name") or chunk.get("document") or "")
    content = str(chunk.get("content") or chunk.get("text") or "")
    return _match(document, case.forbidden_documents) or _match(content, case.forbidden_phrases)
