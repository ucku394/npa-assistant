"""
Telegram bot for Belarus occupational / industrial safety questions.

TEXT:
Telegram
    -> local multilingual-e5-small query embedding
    -> Supabase vector search
    -> semantic similarity sorting
    -> TOP-N RAG chunks
    -> Gemini
    -> OpenRouter fallback
    -> SOURCE_ID validation
    -> Telegram

PHOTO:
Telegram
    -> OpenRouter free multimodal vision
    -> structured visual findings
    -> Belarus NPA RAG verification
    -> human-confirmed draft prescription
"""

import asyncio
import html
import logging
import re
from contextlib import asynccontextmanager
from io import BytesIO

from supabase import create_client

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)


from config import (
    SUPABASE_SERVICE_ROLE_KEY,
    SUPABASE_URL,
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_MESSAGE_LIMIT,
)


from rag import build_source_id
from core.chat_service import chat_service
from vision_service import VisionUnavailableError, analyze_image
from inspection_service import verify_findings
from prescription_service import build_draft_prescription


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(name)s | "
        "%(message)s"
    ),
)

logger = logging.getLogger(__name__)


# ============================================================
# CONCURRENT ACTION MANAGER
# ============================================================

@asynccontextmanager
async def continuous_typing(chat, interval: float = 4.0):
    """
    Фоновая задача, которая периодически обновляет статус 'печатает' в чате,
    пока выполняются операции поиска в базе и генерации ответа.
    """
    async def _send_action():
        try:
            while True:
                await chat.send_action(ChatAction.TYPING)
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            pass

    task = asyncio.create_task(_send_action())
    try:
        yield
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


# ============================================================
# SOURCE ID FUNCTIONS
# ============================================================

def extract_used_source_ids(answer: str) -> list[str]:
    """
    Извлекает SOURCE_ID из ответа модели.
    """
    if not answer:
        return []

    result = []
    seen = set()

    marked_matches = re.findall(
        r"\[SOURCE:([A-Za-zА-Яа-яЁё0-9_./-]+)\]",
        answer,
    )

    for source_id in marked_matches:
        source_id = source_id.strip()
        if not source_id:
            continue
        if source_id not in seen:
            seen.add(source_id)
            result.append(source_id)

    raw_matches = re.findall(
        r"\bNPA_[A-Za-zА-Яа-яЁё0-9_./-]+\b",
        answer,
    )

    for source_id in raw_matches:
        source_id = source_id.strip()
        if not source_id:
            continue
        if source_id not in seen:
            seen.add(source_id)
            result.append(source_id)

    return result


def build_used_source_references(
    chunks,
    used_source_ids,
) -> list[str]:
    """
    Возвращает только те источники, которые присутствуют среди chunks
    и были процитированы моделью.
    """
    if not chunks or not used_source_ids:
        return []

    used = {
        str(source_id).strip()
        for source_id in used_source_ids
        if source_id
    }

    references = []
    seen = set()

    for index, chunk in enumerate(chunks, start=1):
        source_id = (
            chunk.get("_source_id")
            or build_source_id(chunk, index)
        )

        if not source_id or source_id not in used:
            continue

        document_name = (
            chunk.get("document")
            or chunk.get("document_name")
            or chunk.get("doc_name")
            or chunk.get("title")
            or chunk.get("npa_name")
            or chunk.get("source")
            or "Неизвестный НПА"
        )
        document_name = str(document_name).strip()

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
        point = str(point).strip()

        if point:
            reference = f"{document_name} — пункт/статья {point}"
        else:
            reference = document_name

        if reference not in seen:
            seen.add(reference)
            references.append(reference)

    return references


def remove_source_markers(
    answer: str,
    valid_source_ids=None,
) -> str:
    """
    Удаляет технические SOURCE-маркеры.
    """
    if not answer:
        return ""

    answer = re.sub(
        r"\[SOURCE:[A-Za-zА-Яа-яЁё0-9_./-]+\]",
        "",
        answer,
    )

    if valid_source_ids:
        valid_ids = sorted(
            {
                str(source_id).strip()
                for source_id in valid_source_ids
                if source_id
            },
            key=len,
            reverse=True,
        )

        for source_id in valid_ids:
            answer = re.sub(
                rf"(?<![A-Za-zА-Яа-яЁё0-9_])"
                rf"{re.escape(source_id)}"
                rf"(?![A-Za-zА-Яа-яЁё0-9_])",
                "",
                answer,
            )

    answer = re.sub(r"[ \t]+([,.;:])", r"\1", answer)
    answer = re.sub(r"[ \t]+\n", "\n", answer)
    answer = re.sub(r"\n{3,}", "\n\n", answer)

    return answer.strip()


# ============================================================
# CONFIG VALIDATION
# ============================================================

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN is not configured.")

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    raise RuntimeError("SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY are not configured.")


# ============================================================
# ============================================================
# CLIENTS
# ============================================================

supabase = create_client(
    SUPABASE_URL,
    SUPABASE_SERVICE_ROLE_KEY,
)

# TEXT UTILITIES
# ============================================================

def normalize_whitespace(text: str) -> str:
    text = str(text or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_ai_markup(text: str) -> str:
    text = str(text or "")
    replacements = {
        "<br>": "\n",
        "<br/>": "\n",
        "<br />": "\n",
        "</p>": "\n\n",
        "<p>": "",
        "<strong>": "",
        "</strong>": "",
        "<b>": "",
        "</b>": "",
        "<i>": "",
        "</i>": "",
        "<em>": "",
        "</em>": "",
    }

    for old, new in replacements.items():
        text = re.sub(re.escape(old), new, text, flags=re.IGNORECASE)

    text = re.sub(r"```(?:html|markdown|text)?", "", text, flags=re.IGNORECASE)
    text = text.replace("```", "")
    return normalize_whitespace(text)


def ensure_numbered_list_spacing(text: str) -> str:
    text = normalize_whitespace(text)
    text = re.sub(r"(?m)(^|\n)(\s*)(\d{1,2})[.)]\s+", r"\1\2\3. ", text)
    text = re.sub(r"(?m)([^\n])\n(\s*\d{1,2}\.\s+)", r"\1\n\n\2", text)
    return text


def to_telegram_html(text: str) -> str:
    text = clean_ai_markup(text)
    text = ensure_numbered_list_spacing(text)
    text = html.escape(text, quote=False)

    text = re.sub(r"(?m)^\s*#{1,6}\s*(.+?)\s*$", r"<b>\1</b>", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text, flags=re.DOTALL)
    text = re.sub(r"(?m)^\s*[-*]\s+", "• ", text)
    text = re.sub(r"(?m)^\s*([-_])(?:\s*\1){2,}\s*$", "────────", text)
    text = re.sub(r"(?<!\*)\*([^*\n]+?)\*(?!\*)", r"<i>\1</i>", text)

    return text.strip()


# ============================================================
# TELEGRAM MESSAGE SPLITTING
# ============================================================

def split_text_smart(
    text: str,
    limit: int = TELEGRAM_MESSAGE_LIMIT,
) -> list[str]:
    text = str(text or "").strip()
    if len(text) <= limit:
        return [text]

    parts = []
    remaining = text

    while len(remaining) > limit:
        cut = remaining.rfind("\n\n", 0, limit)
        if cut < limit // 2:
            cut = remaining.rfind("\n", 0, limit)
        if cut < limit // 2:
            cut = remaining.rfind(" ", 0, limit)
        if cut < limit // 2:
            cut = limit

        part = remaining[:cut].strip()
        if part:
            parts.append(part)
        remaining = remaining[cut:].strip()

    if remaining:
        parts.append(remaining)

    return parts


async def send_long_message(
    update: Update,
    text: str,
    use_html: bool = True,
):
    if not update.effective_message:
        return

    for chunk in split_text_smart(text):
        if use_html:
            rendered = to_telegram_html(chunk)
            try:
                await update.effective_message.reply_text(
                    rendered,
                    parse_mode="HTML",
                    disable_web_page_preview=True,
                )
                continue
            except Exception:
                logger.exception("Telegram HTML send failed; retrying plain text.")

        await update.effective_message.reply_text(
            html.unescape(str(chunk)),
            disable_web_page_preview=True,
        )


def split_html_message(
    text: str,
    limit: int = TELEGRAM_MESSAGE_LIMIT,
) -> list[str]:
    """
    Разбивает уже подготовленный Telegram HTML на безопасные по длине
    блоки. Старается резать только по строкам/абзацам и не оставляет
    незакрытые HTML-теги.
    """
    text = str(text or "").strip()
    if len(text) <= limit:
        return [text]

    lines = text.splitlines()
    parts = []
    current = ""

    def flush():
        nonlocal current
        if current.strip():
            parts.append(current.strip())
        current = ""

    for line in lines:
        candidate = line if not current else current + "\n" + line

        if len(candidate) <= limit:
            current = candidate
            continue

        if current:
            flush()

        if len(line) <= limit:
            current = line
            continue

        remaining = line
        while len(remaining) > limit:
            cut = remaining.rfind(" ", 0, limit)
            if cut < limit // 2:
                cut = limit

            chunk = remaining[:cut].strip()
            if chunk:
                parts.append(chunk)
            remaining = remaining[cut:].strip()

        if remaining:
            current = remaining

    flush()
    return parts


async def send_inspection_message(
    update: Update,
    text: str,
    reply_markup=None,
):
    """
    Отправляет профессиональный результат фотоинспекции без риска
    Telegram BadRequest: Message is too long.

    Кнопки прикрепляются только к последней части.
    """
    if not update.effective_message:
        return

    parts = split_html_message(text)

    for index, part in enumerate(parts):
        markup = reply_markup if index == len(parts) - 1 else None

        try:
            await update.effective_message.reply_text(
                part,
                parse_mode="HTML",
                reply_markup=markup,
                disable_web_page_preview=True,
            )
        except Exception as exc:
            logger.warning(
                "Inspection HTML send failed for part %s/%s: %s. "
                "Retrying plain text.",
                index + 1,
                len(parts),
                exc,
            )
            await update.effective_message.reply_text(
                re.sub(r"<[^>]+>", "", html.unescape(part)),
                reply_markup=markup,
                disable_web_page_preview=True,
            )


# ============================================================
# DEBUG UPDATE
# ============================================================

async def debug_update(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    logger.info(
        "TELEGRAM UPDATE | update_id=%s | user_id=%s | chat_id=%s | text=%r | photo=%s",
        update.update_id,
        update.effective_user.id if update.effective_user else None,
        update.effective_chat.id if update.effective_chat else None,
        update.effective_message.text if update.effective_message else None,
        bool(update.effective_message and update.effective_message.photo),
    )


# ============================================================
# /START
# ============================================================

async def start(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    if not update.effective_message:
        return

    await update.effective_message.reply_text(
        "Здравствуйте! Я помощник по охране труда и "
        "промышленной безопасности в Республике Беларусь.\n\n"
        "Задайте вопрос текстом или отправьте фотографию — "
        "я помогу разобрать ситуацию."
    )


# ============================================================
# TEXT HANDLER
# ============================================================

async def text_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    if not update.effective_message:
        return

    question = (update.effective_message.text or "").strip()
    if not question:
        return

    status_message = None
    try:
        status_message = await update.effective_message.reply_text(
            "🔍 <i>Ищу в базе НПА Республики Беларусь...</i>",
            parse_mode="HTML",
        )
    except Exception as e:
        logger.warning("Не удалось отправить статусное сообщение: %s", e)

    try:
        async with continuous_typing(update.effective_chat):
            # Единый AI Core: RAG -> prompt -> AI -> SOURCE_ID.
            result = await chat_service.process_text(question)

            if not result.get("success"):
                if status_message:
                    try:
                        await status_message.delete()
                    except Exception:
                        pass

                error = result.get("error")
                if error == "no_relevant_context":
                    await update.effective_message.reply_text(
                        "Я не нашёл достаточно релевантных фрагментов НПА в базе, "
                        "поэтому не буду придумывать нормативное требование."
                    )
                else:
                    await update.effective_message.reply_text(
                        "Не удалось обработать вопрос."
                    )
                return

            if status_message:
                try:
                    await status_message.edit_text(
                        "⚖️ <i>Анализирую требования законодательства...</i>",
                        parse_mode="HTML",
                    )
                except Exception as e:
                    logger.debug("Не удалось обновить статус анализа: %s", e)

            answer = clean_ai_markup(result.get("answer") or "")
            answer = ensure_numbered_list_spacing(answer)

            sources = result.get("sources") or []
            if sources:
                answer += "\n\n📎 ИСТОЧНИКИ\n\n"
                answer += "\n\n".join(
                    (
                        f"• {source.get('document', 'Неизвестный НПА')}"
                        + (
                            f" — пункт/статья {source.get('point')}"
                            if source.get("point")
                            else ""
                        )
                    )
                    for source in sources
                )
            else:
                logger.warning(
                    "LEGAL | AI did not provide valid SOURCE_IDs. "
                    "No automatic sources will be added."
                )

            answer = clean_ai_markup(answer)
            answer = ensure_numbered_list_spacing(answer)

        if status_message:
            try:
                await status_message.delete()
            except Exception:
                pass

        await send_long_message(
            update,
            answer,
            use_html=True,
        )

    except Exception:
        logger.exception("Text handler failed.")

        if status_message:
            try:
                await status_message.delete()
            except Exception:
                pass

        await update.effective_message.reply_text(
            "Произошла ошибка при обработке запроса. "
            "Попробуйте ещё раз через несколько секунд."
        )


# ============================================================
# ============================================================
# PHOTO INSPECTION / PRESCRIPTION
# ============================================================

def _format_inspection_result(vision: dict, verified: list[dict]) -> str:
    """
    Профессиональное представление результата фотоинспекции.

    Принцип:
    1. Что объективно видно на фотографии.
    2. Какие признаки требуют внимания.
    3. Какие нарушения подтверждены конкретным НПА.
    4. Какие факты необходимо проверить на месте.

    Технические поля AI/RAG пользователю не показываются.
    """
    category_map = {
        "occupational_safety": "Охрана труда",
        "fire_safety": "Пожарная безопасность",
        "industrial_safety": "Промышленная безопасность",
        "radiation_safety": "Радиационная безопасность",
        "unknown": "Требует определения",
    }

    def esc(value) -> str:
        return html.escape(str(value or "").strip())

    def first_text(item: dict, *keys: str) -> str:
        for key in keys:
            value = str(item.get(key) or "").strip()
            if value:
                return value
        return ""

    category_raw = str(vision.get("category") or "").strip()
    category = category_map.get(category_raw, category_raw or "Требует определения")
    scene = str(vision.get("scene") or "").strip()
    observations = vision.get("observations") or []

    confirmed = [
        item for item in verified
        if item.get("status") == "confirmed"
    ]
    potential = [
        item for item in verified
        if item.get("status") == "potential"
    ]
    not_confirmed = [
        item for item in verified
        if item.get("status") == "not_confirmed"
    ]

    lines = [
        "📷 <b>Фотоинспекция</b>",
        "",
        f"🏷️ <b>Область:</b> {esc(category)}",
    ]

    # --------------------------------------------------------
    # 1. OBJECTIVE VISUAL DESCRIPTION
    # --------------------------------------------------------
    lines.extend(["", "👁️ <b>Что видно на фото</b>"])

    if scene:
        lines.append(esc(scene))

    visible_observations = []
    for item in observations[:5]:
        if not isinstance(item, dict):
            continue

        description = first_text(item, "description", "observation")
        evidence = first_text(item, "evidence", "visual_evidence")

        if description:
            visible_observations.append((description, evidence))

    if visible_observations:
        for description, evidence in visible_observations:
            lines.append(f"• {esc(description)}")
            if evidence and evidence != description:
                lines.append(f"  <i>Визуальный признак: {esc(evidence)}</i>")
    elif not scene:
        lines.append(
            "• На фотографии недостаточно данных для уверенного "
            "описания обстановки."
        )

    # --------------------------------------------------------
    # 2. CONFIRMED VIOLATIONS
    # --------------------------------------------------------
    if confirmed:
        lines.extend(["", "🚨 <b>Подтверждённые нарушения</b>"])

        for index, item in enumerate(confirmed, 1):
            violation = first_text(
                item,
                "violation",
                "description",
            ) or "Подтверждённое нарушение"

            lines.append(f"<b>{index}. {esc(violation)}</b>")

            evidence = first_text(
                item,
                "evidence",
                "visual_evidence",
            )
            if evidence:
                lines.append(
                    f"   👁️ <b>Визуальный признак:</b> {esc(evidence)}"
                )

            legal_basis = item.get("legal_basis") or []
            if legal_basis:
                lines.append("   📚 <b>Нормативное основание:</b>")
                for ref in legal_basis[:3]:
                    if not isinstance(ref, dict):
                        continue

                    document = first_text(
                        ref,
                        "document",
                        "doc_name",
                    ) or "НПА Республики Беларусь"
                    point = first_text(
                        ref,
                        "point",
                        "article",
                    )

                    reference = esc(document)
                    if point:
                        reference += f", {esc(point)}"

                    lines.append(f"   • {reference}")

            action = first_text(
                item,
                "corrective_action",
                "action",
            )
            if action:
                lines.append(
                    f"   🛠️ <b>Рекомендуемое действие:</b> {esc(action)}"
                )

    # --------------------------------------------------------
    # 3. POTENTIAL / NEEDS ON-SITE VERIFICATION
    # --------------------------------------------------------
    if potential:
        lines.extend(["", "⚠️ <b>Признаки, требующие проверки</b>"])

        for index, item in enumerate(potential, 1):
            description = first_text(
                item,
                "violation",
                "description",
            ) or "Возможный риск или признак несоответствия"

            lines.append(f"<b>{index}. {esc(description)}</b>")

            evidence = first_text(
                item,
                "evidence",
                "visual_evidence",
            )
            if evidence:
                lines.append(
                    f"   👁️ <b>Визуальный признак:</b> {esc(evidence)}"
                )

            checks = [
                str(value).strip()
                for value in (item.get("verification_needed") or [])
                if str(value).strip()
            ]

            if checks:
                lines.append("   🔎 <b>Что проверить на месте:</b>")
                for check in checks[:4]:
                    lines.append(f"   • {esc(check)}")

            legal_basis = item.get("legal_basis") or []
            if legal_basis:
                lines.append("   📚 <b>Нормативная проверка:</b>")
                for ref in legal_basis[:3]:
                    if not isinstance(ref, dict):
                        continue

                    document = first_text(
                        ref,
                        "document",
                        "doc_name",
                    ) or "НПА Республики Беларусь"
                    point = first_text(
                        ref,
                        "point",
                        "article",
                    )

                    reference = esc(document)
                    if point:
                        reference += f", {esc(point)}"

                    lines.append(f"   • {reference}")

    # --------------------------------------------------------
    # 4. NOT CONFIRMED / NO FINDINGS
    # --------------------------------------------------------
    if not confirmed and not potential:
        if not_confirmed:
            lines.extend([
                "",
                "ℹ️ <b>Результат проверки</b>",
                "По отдельным визуальным признакам фотография "
                "не позволяет достоверно установить нарушение.",
                "",
                "📌 <i>Для окончательной оценки необходимо проверить "
                "фактические условия выполнения работ на месте.</i>",
            ])
        else:
            lines.extend([
                "",
                "✅ <b>Явных признаков нарушений не обнаружено</b>",
                "",
                "По фотографии не выявлено визуальных признаков, "
                "которые можно обоснованно отнести к нарушению "
                "требований безопасности.",
                "",
                "📌 <i>Фотоанализ не заменяет проверку условий "
                "выполнения работ на месте.</i>",
            ])
    elif potential and not confirmed:
        lines.extend([
            "",
            "📌 <i>Выявленные признаки не считаются подтверждённым "
            "нарушением до проверки факта и применимого нормативного "
            "требования.</i>",
        ])
    elif confirmed and potential:
        lines.extend([
            "",
            "📌 <i>Часть выявленных признаков подтверждена нормативно, "
            "остальные требуют дополнительной проверки на месте.</i>",
        ])

    return "\n".join(lines)

async def photo_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    if not update.effective_message or not update.effective_message.photo:
        return

    status_message = None
    try:
        status_message = await update.effective_message.reply_text(
            "📷 <i>1/3 Анализирую фотографию...</i>",
            parse_mode="HTML",
        )

        async with continuous_typing(update.effective_chat):
            photo = update.effective_message.photo[-1]
            telegram_file = await context.bot.get_file(photo.file_id)

            buffer = BytesIO()
            await telegram_file.download_to_memory(out=buffer)
            image_bytes = buffer.getvalue()

            if len(image_bytes) > 12 * 1024 * 1024:
                raise ValueError("Фотография слишком большая для анализа (максимум 12 МБ).")

            caption = (update.effective_message.caption or "").strip()

            vision = await asyncio.to_thread(
                analyze_image,
                image_bytes,
                "image/jpeg",
                caption,
            )

            if status_message:
                await status_message.edit_text(
                    "⚖️ <i>2/3 Проверяю потенциальные нарушения по базе НПА РБ...</i>",
                    parse_mode="HTML",
                )

            verified = await verify_findings(vision.get("potential_findings") or [], supabase)

            context.user_data["pending_inspection"] = {
                "vision": vision,
                "verified": verified,
                "photo_bytes": image_bytes,
                "caption": caption,
            }

            answer = _format_inspection_result(vision, verified)
            confirmed = [
                item for item in verified
                if item.get("status") == "confirmed" and item.get("legal_basis")
            ]

            if confirmed:
                keyboard = InlineKeyboardMarkup([
                    [InlineKeyboardButton(
                        "📄 Оформить ПРОЕКТ предписания",
                        callback_data="inspection:prescription",
                    )],
                    [InlineKeyboardButton(
                        "❌ Отклонить результаты",
                        callback_data="inspection:reject",
                    )],
                ])
            else:
                keyboard = InlineKeyboardMarkup([
                    [InlineKeyboardButton(
                        "❌ Закрыть результат",
                        callback_data="inspection:reject",
                    )],
                ])

            if status_message:
                await status_message.delete()

            await send_inspection_message(
                update,
                answer,
                reply_markup=keyboard,
            )

    except VisionUnavailableError:
        logger.warning("Photo inspection temporarily unavailable: vision providers are unavailable")
        if status_message:
            try:
                await status_message.delete()
            except Exception:
                pass
        await update.effective_message.reply_text(
            "⚠️ Фото получено, но сейчас недоступен сервис визуального анализа, временно перегружен.\n\n"
            "Попробуйте отправить фото ещё раз немного позже."
        )
    except Exception as exc:
        logger.exception("Photo inspection failed: %s", exc)
        if status_message:
            try:
                await status_message.delete()
            except Exception:
                pass
        await update.effective_message.reply_text(
            "Не удалось выполнить фотоинспекцию. Проверьте доступность сервиса и повторите отправку фотографии."
        )


async def inspection_callback(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    query = update.callback_query
    if not query:
        return

    await query.answer()

    pending = context.user_data.get("pending_inspection")
    if not pending:
        await query.edit_message_text(
            "Результат фотоинспекции больше недоступен. Отправьте фотографию заново."
        )
        return

    if query.data == "inspection:reject":
        context.user_data.pop("pending_inspection", None)
        await query.edit_message_text("Результат фотоинспекции отклонён.")
        return

    if query.data != "inspection:prescription":
        return

    confirmed = [
        item for item in pending.get("verified", [])
        if item.get("status") == "confirmed" and item.get("legal_basis")
    ]
    if not confirmed:
        await query.edit_message_text(
            "Нет подтверждённых нарушений с валидным нормативным основанием."
        )
        return

    await query.edit_message_text(
        "📄 <b>Формирую проект предписания...</b>",
        parse_mode="HTML",
    )

    try:
        docx_bytes = await asyncio.to_thread(
            build_draft_prescription,
            confirmed,
            pending.get("photo_bytes"),
        )
        await query.message.reply_document(
            document=BytesIO(docx_bytes),
            filename="proekt_predpisaniya_po_foto.docx",
            caption=(
                "📄 Проект предписания сформирован. "
                "Перед официальным применением проверьте факты, "
                "сроки, ответственных и нормативное основание."
            ),
        )
        context.user_data.pop("pending_inspection", None)
    except Exception:
        logger.exception("Prescription generation failed.")
        await query.message.reply_text(
            "Не удалось сформировать DOCX проекта предписания."
        )


# GLOBAL ERROR HANDLER
# ============================================================

async def error_handler(
    update: object,
    context: ContextTypes.DEFAULT_TYPE,
):
    logger.error(
        "UNHANDLED TELEGRAM ERROR: %s",
        context.error,
        exc_info=context.error,
    )


# ============================================================
# MAIN
# ============================================================

def main():
    logger.info("Bot started.")

    application = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .build()
    )

    application.add_handler(
        MessageHandler(filters.ALL, debug_update),
        group=-100,
    )
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.PHOTO, photo_handler))
    application.add_handler(CallbackQueryHandler(inspection_callback, pattern=r"^inspection:"))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_handler))
    application.add_error_handler(error_handler)

    application.run_polling(
        drop_pending_updates=False,
        allowed_updates=Update.ALL_TYPES,
        close_loop=False,
    )


if __name__ == "__main__":
    main()
