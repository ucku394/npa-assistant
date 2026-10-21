"""FastAPI web API for the Belarus OHS / industrial safety assistant.

The API uses the shared ChatService, so Telegram and Web have the same
RAG, prompt, AI fallback and SOURCE_ID validation pipeline.
"""

import logging
import os
import asyncio
import time
from collections import defaultdict, deque
from io import BytesIO
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from core.chat_service import chat_service
from services.inspection_service import verify_findings
from services.prescription_service import build_draft_prescription
from services.document_generator import (
    ActN1Request,
    WorkPermitRequest,
    auto_suggest_act_n1_rules,
    auto_suggest_work_permit_measures,
    generate_act_n1_docx,
    generate_work_permit_docx,
)
from services.vision_service import analyze_image, VisionUnavailableError


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
    raw = os.getenv("WEB_CORS_ORIGINS", "").strip()
    # Same-origin browser requests do not need CORS. Cross-origin access must
    # be explicitly allowlisted by the operator.
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


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

# Lightweight process-local abuse controls. For multi-replica deployments,
# enforce equivalent limits at the gateway/WAF as well.
_RATE_WINDOWS = defaultdict(deque)
_RATE_LIMITS = {
    "/api/chat": (20, 60),
    "/api/inspect": (5, 60),
    "/api/prescription": (5, 60),
    "/api/documents": (10, 60),
}
_RATE_DEFAULT = (60, 60)
_RATE_CLEANUP_COUNTER = 0


@app.middleware("http")
async def api_abuse_controls(request, call_next):
    global _RATE_CLEANUP_COUNTER
    path = request.url.path
    if not path.startswith("/api/") or path == "/api/health":
        return await call_next(request)

    # Reject oversized requests before FastAPI reads/parses the body.
    try:
        declared_length = int(request.headers.get("content-length", "0") or 0)
    except ValueError:
        return JSONResponse(status_code=400, content={"detail": "Некорректный Content-Length."})
    body_limit = 13 * 1024 * 1024 if path == "/api/inspect" else 256 * 1024
    if declared_length > body_limit:
        return JSONResponse(status_code=413, content={"detail": "Размер запроса превышает допустимый лимит."})

    client_ip = request.client.host if request.client else "unknown"
    route_key = next((key for key in _RATE_LIMITS if path == key or path.startswith(key + "/")), None)
    limit, window_seconds = _RATE_LIMITS.get(route_key, _RATE_DEFAULT)
    bucket_key = (client_ip, route_key or path)
    now = time.monotonic()
    events = _RATE_WINDOWS[bucket_key]
    while events and now - events[0] >= window_seconds:
        events.popleft()
    if len(events) >= limit:
        retry_after = max(1, int(window_seconds - (now - events[0])))
        return JSONResponse(
            status_code=429,
            headers={"Retry-After": str(retry_after)},
            content={"detail": "Слишком много запросов. Повторите попытку позже."},
        )
    events.append(now)

    _RATE_CLEANUP_COUNTER += 1
    if _RATE_CLEANUP_COUNTER % 500 == 0:
        for key in list(_RATE_WINDOWS):
            queue = _RATE_WINDOWS[key]
            while queue and now - queue[0] >= window_seconds:
                queue.popleft()
            if not queue:
                _RATE_WINDOWS.pop(key, None)

    return await call_next(request)


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

    if result.get("error") == "grounding_failed":
        return ChatResponse(
            success=False,
            error="grounding_failed",
            question=question,
            answer=(
                "Не удалось подтвердить ответ найденными фрагментами нормативных "
                "правовых актов. Чтобы не вводить вас в заблуждение, я не буду "
                "выдавать неподтверждённое предположение за требование законодательства. "
                "Попробуйте уточнить вопрос или указать конкретный нормативный акт."
            ),
            answer_with_citations="",
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
    # Never trust status/legal_basis supplied by the browser. Re-run legal
    # verification on the server and use only the fresh, source-validated result.
    candidates = []
    for item in request.findings:
        if not isinstance(item, dict):
            continue
        description = str(item.get("violation") or item.get("description") or "").strip()
        if not description:
            continue
        candidates.append({
            "description": description,
            "visual_evidence": str(item.get("evidence") or item.get("visual_evidence") or "").strip(),
            "risk_level": item.get("risk_level", "medium"),
        })
    if not candidates:
        raise HTTPException(status_code=400, detail="Нет наблюдений для повторной юридической проверки.")
    try:
        reverified = await verify_findings(candidates, chat_service.supabase)
    except Exception:
        logger.exception("WEB API | prescription legal re-verification failed")
        raise HTTPException(status_code=503, detail="Не удалось повторно проверить нормативные основания. Проект не сформирован.")

    confirmed = [
        item for item in reverified
        if item.get("status") == "confirmed" and item.get("legal_basis")
    ]
    if not confirmed:
        raise HTTPException(
            status_code=400,
            detail="Повторная проверка не подтвердила нарушения с нормативным основанием. Проект не сформирован.",
        )
    try:
        content = await asyncio.to_thread(
            build_draft_prescription,
            findings=confirmed,
            enterprise=request.enterprise,
            subdivision=request.subdivision,
            workplace=request.workplace,
            recipient=request.recipient,
            prescription_number=request.prescription_number,
            deadline=request.deadline or None,
            issuer_name=request.issuer_name,
            issuer_position=request.issuer_position,
            recipient_name=request.recipient_name,
            recipient_position=request.recipient_position,
        )
        return StreamingResponse(BytesIO(content),
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={"Content-Disposition": 'attachment; filename="proekt_predpisaniya.docx"'})
    except Exception:
        logger.exception("WEB API | prescription generation failed")
        raise HTTPException(status_code=500, detail="Не удалось сформировать проект предписания.")


# ============================================================
# DOCUMENT GENERATION ENDPOINTS (ACT H-1 & WORK PERMIT)
# ============================================================

@app.post("/api/documents/act-n1")
async def create_act_n1(request: ActN1Request):
    """Validate and auto-enrich Act H-1 (несчастный случай на производстве)."""
    auto = auto_suggest_act_n1_rules(request.incident_description)
    data = request.model_dump()
    if not data.get("violated_npa"):
        data["violated_npa"] = auto["violated_npa"]
    if not data.get("causes"):
        data["causes"] = auto["causes"]
    if not data.get("corrective_measures"):
        data["corrective_measures"] = auto["corrective_measures"]
    return {
        "status": "success",
        "document_type": "act_n1",
        "title": "Акт о несчастном случае на производстве формы Н-1",
        "data": data,
    }


@app.post("/api/documents/act-n1/download")
async def download_act_n1(request: ActN1Request):
    """Generate and stream official Belarus Act H-1 as DOCX."""
    try:
        content_io = await asyncio.to_thread(generate_act_n1_docx, request)
        return StreamingResponse(
            content_io,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={"Content-Disposition": 'attachment; filename="Akt_formy_H1.docx"'},
        )
    except Exception:
        logger.exception("WEB API | Act H-1 generation failed")
        raise HTTPException(status_code=500, detail="Не удалось сформировать Акт формы Н-1.")


@app.post("/api/documents/work-permit")
async def create_work_permit(request: WorkPermitRequest):
    """Validate and auto-enrich Work Permit (наряд-допуск на работы повышенной опасности)."""
    auto = auto_suggest_work_permit_measures(request.work_type)
    data = request.model_dump()
    if not data.get("safety_measures"):
        data["safety_measures"] = auto["safety_measures"]
    if not data.get("required_ppe"):
        data["required_ppe"] = auto["required_ppe"]
    return {
        "status": "success",
        "document_type": "work_permit",
        "title": "Наряд-допуск на выполнение работ с повышенной опасностью",
        "data": data,
    }


@app.post("/api/documents/work-permit/download")
async def download_work_permit(request: WorkPermitRequest):
    """Generate and stream official Belarus Work Permit as DOCX."""
    try:
        content_io = await asyncio.to_thread(generate_work_permit_docx, request)
        return StreamingResponse(
            content_io,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={"Content-Disposition": 'attachment; filename="Naryad_dopusk.docx"'},
        )
    except Exception:
        logger.exception("WEB API | Work Permit generation failed")
        raise HTTPException(status_code=500, detail="Не удалось сформировать наряд-допуск.")
