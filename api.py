"""FastAPI web API for the Belarus OHS / industrial safety assistant.

The API uses the shared ChatService, so Telegram and Web have the same
RAG, prompt, AI fallback and SOURCE_ID validation pipeline.
"""

import logging
import os
import asyncio
from io import BytesIO
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from core.chat_service import chat_service
from inspection_service import verify_findings
from prescription_service import build_draft_prescription
from vision_service import analyze_image, VisionUnavailableError


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

# Prevent HTTP client logs from emitting credential-bearing Telegram API URLs.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

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
    answer_with_citations: str = ""
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
            answer_with_citations="",
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
        answer_with_citations=result.get("answer_with_citations") or result.get("answer") or "",
        sources=result.get("sources") or [],
        rag=result.get("rag") or {},
    )


class PrescriptionRequest(BaseModel):
    findings: list[dict] = Field(default_factory=list)
    enterprise: str = ""
    subdivision: str = ""
    workplace: str = ""
    recipient: str = ""
    prescription_number: str = ""
    deadline: str = ""
    issuer_name: str = ""
    issuer_position: str = ""
    recipient_name: str = ""
    recipient_position: str = ""


@app.post("/api/inspect")
async def inspect_photo(file: UploadFile = File(...)):
    allowed = {"image/jpeg", "image/png", "image/webp"}
    mime = file.content_type or "image/jpeg"
    if mime not in allowed:
        raise HTTPException(status_code=400, detail="Поддерживаются JPG, PNG и WEBP.")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Файл фотографии пуст.")
    if len(data) > 12 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Размер фотографии не должен превышать 12 МБ.")
    try:
        vision = await asyncio.to_thread(analyze_image, data, mime, "")
        findings = await verify_findings(vision.get("potential_findings") or [], chat_service.supabase)
        return {"success": True, "scene": vision.get("scene",""), "category": vision.get("category",""),
                "observations": vision.get("observations", []), "findings": findings,
                "needs_review": vision.get("needs_review", False)}
    except VisionUnavailableError as exc:
        logger.warning("WEB API | photo inspection unavailable | error=%s", exc)
        return {
            "success": False,
            "error": "vision_unavailable",
            "scene": "",
            "category": "unknown",
            "observations": [],
            "findings": [],
            "needs_review": True,
            "message": "Автоматический анализ фотографии временно недоступен. Повторите попытку позже.",
        }
    except Exception:
        logger.exception("WEB API | photo inspection failed")
        raise HTTPException(status_code=500, detail="Не удалось выполнить фотоинспекцию.")


@app.post("/api/prescription")
async def create_prescription(request: PrescriptionRequest):
    confirmed = [x for x in request.findings if x.get("status") == "confirmed" and x.get("legal_basis")]
    if not confirmed:
        raise HTTPException(status_code=400, detail="Нет подтверждённых нарушений с проверенным нормативным основанием.")
    try:
        content = build_draft_prescription(
            findings=confirmed, enterprise=request.enterprise, subdivision=request.subdivision,
            workplace=request.workplace, recipient=request.recipient,
            prescription_number=request.prescription_number, deadline=request.deadline or None,
            issuer_name=request.issuer_name, issuer_position=request.issuer_position,
            recipient_name=request.recipient_name, recipient_position=request.recipient_position,
        )
        return StreamingResponse(BytesIO(content),
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={"Content-Disposition": 'attachment; filename="proekt_predpisaniya.docx"'})
    except Exception:
        logger.exception("WEB API | prescription generation failed")
        raise HTTPException(status_code=500, detail="Не удалось сформировать проект предписания.")
