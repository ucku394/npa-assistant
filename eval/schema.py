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

def chunk_matches(case: EvalCase, chunk: dict) -> bool:
    document = str(chunk.get("doc_name") or chunk.get("document") or "")
    point = str(chunk.get("point_num") or chunk.get("point") or chunk.get("article") or "")
    if case.expected_documents and not _match(document, case.expected_documents):
        return False
    if case.expected_points and not _match(point, case.expected_points):
        return False
    return bool(case.expected_documents or case.expected_points)

def chunk_is_forbidden(case: EvalCase, chunk: dict) -> bool:
    document = str(chunk.get("doc_name") or chunk.get("document") or "")
    content = str(chunk.get("content") or chunk.get("text") or "")
    return _match(document, case.forbidden_documents) or _match(content, case.forbidden_phrases)
