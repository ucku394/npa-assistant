"""FastAPI web API for the Belarus OHS / industrial safety assistant.

The API uses the shared ChatService, so Telegram and Web have the same
RAG, prompt, AI fallback and SOURCE_ID validation pipeline.
"""

import logging
import os
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from core.chat_service import chat_service


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)


app = FastAPI(
    title="Ассистент по ОТ и ПБ РБ API",
    version="1.0.0",
    description=(
        "API ассистента по охране труда, пожарной и промышленной "
        "безопасности в Республике Беларусь."
    ),
)


def _cors_origins() -> List[str]:
    raw = os.getenv("WEB_CORS_ORIGINS", "*").strip()
    if not raw:
        return ["*"]
    return [item.strip() for item in raw.split(",") if item.strip()] or ["*"]


WEB_DIR = Path(__file__).resolve().parent / "web"


app.mount("/web", StaticFiles(directory=WEB_DIR), name="web")


@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(WEB_DIR / "index.html")


app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


class ChatRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="Вопрос пользователя.",
    )


class SourceItem(BaseModel):
    source_id: str
    document: str
    point: str = ""
    source_url: str = ""
    citation_order: int = 0


class RagMeta(BaseModel):
    found: bool = False
    candidate_count: int = 0
    final_count: int = 0
    legal_domain: str | None = None
    topic: str | None = None


class ChatResponse(BaseModel):
    success: bool
    error: str | None = None
    question: str
    answer: str
    sources: list[SourceItem] = []
    rag: RagMeta = RagMeta()


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "service": "npa-assistant-api",
        "version": "1.0.0",
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Вопрос не может быть пустым.",
        )

    try:
        result = await chat_service.process_text(question)
    except Exception:
        logger.exception("WEB API | chat processing failed")
        raise HTTPException(
            status_code=500,
            detail="Не удалось обработать вопрос.",
        )

    if result.get("error") == "empty_question":
        raise HTTPException(
            status_code=400,
            detail="Вопрос не может быть пустым.",
        )

    if result.get("error") == "no_relevant_context":
        return ChatResponse(
            success=False,
            error="no_relevant_context",
            question=question,
            answer=(
                "Я не нашёл достаточно релевантных фрагментов НПА "
                "в базе, поэтому не буду придумывать нормативное требование."
            ),
            sources=[],
            rag=result.get("rag") or {},
        )

    if not result.get("success"):
        raise HTTPException(
            status_code=500,
            detail="Не удалось обработать вопрос.",
        )

    return ChatResponse(
        success=True,
        error=None,
        question=question,
        answer=result.get("answer") or "",
        sources=result.get("sources") or [],
        rag=result.get("rag") or {},
    )
