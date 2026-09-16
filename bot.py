import asyncio
import base64
import html
import logging
import re
from io import BytesIO
from openai import OpenAI
from supabase import create_client
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
from ai_router import generate_answer
from config import DEEPSEEK_API_KEY, DEEPSEEK_VISION_MODEL, SUPABASE_SERVICE_ROLE_KEY, SUPABASE_URL, TELEGRAM_BOT_TOKEN, TELEGRAM_MESSAGE_LIMIT
from prompts import LEGAL_ASSISTANT_PROMPT, VISION_ANALYSIS_PROMPT
from rag import get_source_references, retrieve_context

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger=logging.getLogger(__name__)

if not TELEGRAM_BOT_TOKEN: raise RuntimeError("TELEGRAM_BOT_TOKEN is not configured.")
if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY: raise RuntimeError("SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY are not configured.")
supabase=create_client(SUPABASE_URL,SUPABASE_SERVICE_ROLE_KEY)
deepseek_client=OpenAI(api_key=DEEPSEEK_API_KEY,base_url="https://api.deepseek.com") if DEEPSEEK_API_KEY else None

def clean_ai_markup(text):
    text=re.sub(r"```(?:\w+)?\s*|```","",str(text or ""))
    text=re.sub(r"</?(?:b|strong|i|em|u|s|del|code|pre|br)\b[^>]*>","",text,flags=re.I)
    return re.sub(r"\n{3,}","\n\n",html.unescape(text.replace("\r\n","\n").replace("\r","\n"))).strip()

def ensure_numbered_list_spacing(text):
    return re.sub(r"(?<!^)\s+(?=\d{1,2}[.)]\s)", "\n\n", clean_ai_markup(text))

def to_telegram_html(text):
    text=html.escape(ensure_numbered_list_spacing(text),quote=False)
    text=re.sub(r"(?m)^\s*#{1,6}\s*(.+?)\s*$",r"<b>\1</b>",text)
    text=re.sub(r"\*\*(.+?)\*\*",r"<b>\1</b>",text,flags=re.S)
    text=re.sub(r"(?m)^\s*[-*]\s+","• ",text)
    return re.sub(r"(?<!\*)\*([^*\n]+?)\*(?!\*)",r"<i>\1</i>",text).strip()

def split_text_smart(text,limit=TELEGRAM_MESSAGE_LIMIT):
    text=str(text or "").strip()
    if len(text)<=limit: return [text] if text else []
    parts=[]
    while len(text)>limit:
        cut=text.rfind("\n\n",0,limit)
        if cut<limit//2: cut=text.rfind("\n",0,limit)
        if cut<limit//2: cut=text.rfind(" ",0,limit)
        if cut<limit//2: cut=limit
        parts.append(text[:cut].strip())
        text=text[cut:].strip()
    if text: parts.append(text)
    return parts

async def send_long_message(update,text,use_html=True):
    if not update.effective_message: return
    for chunk in split_text_smart(text):
        if use_html:
            try:
                await update.effective_message.reply_text(to_telegram_html(chunk),parse_mode="HTML",disable_web_page_preview=True)
                continue
            except Exception as exc:
                logger.warning("Telegram HTML send failed; plain fallback | error=%s",exc)
        await update.effective_message.reply_text(html.unescape(clean_ai_markup(chunk)),disable_web_page_preview=True)

async def start(update,context):
    await update.effective_message.reply_text("Здравствуйте! Я помощник по охране труда и промышленной безопасности в Республике Беларусь.\n\nЗадайте вопрос текстом или отправьте фотографию.")

async def text_handler(update,context):
    if not update.effective_message: return
    question=(update.effective_message.text or "").strip()
    if not question: return
    try:
        await update.effective_chat.send_action(ChatAction.TYPING)
        result=await retrieve_context(question,supabase)
        chunks=result["chunks"]
        if not chunks:
            await update.effective_message.reply_text("Я не нашёл достаточно релевантных фрагментов НПА в базе, поэтому не буду придумывать нормативное требование.")
            return
        prompt=LEGAL_ASSISTANT_PROMPT.format(context=result["retrieved_text"],question=question)
        answer=await asyncio.to_thread(generate_answer,prompt)
        answer=ensure_numbered_list_spacing(answer)
        sources=get_source_references(chunks)
        if sources: answer += "\n\n📎 ИСТОЧНИКИ\n\n" + "\n\n".join(sources)
        await send_long_message(update,answer)
    except RuntimeError as exc:
        logger.warning("Text handler provider error | error=%s",exc)
        await update.effective_message.reply_text("⚠️ Сейчас не удалось получить ответ от AI-модели. Поиск по базе НПА выполнен, но генератор ответа временно недоступен. Попробуйте повторить вопрос немного позже.")
    except Exception:
        logger.exception("Text handler failed.")
        await update.effective_message.reply_text("Произошла ошибка при обработке запроса. Попробуйте ещё раз через несколько секунд.")

def _vision_text(response):
    if not response.choices: return ""
    content=response.choices[0].message.content
    if isinstance(content,str): return content.strip()
    if isinstance(content,list):
        return "\n".join(str(x.get("text") or "") for x in content if isinstance(x,dict) and x.get("type")=="text").strip()
    return str(content or "").strip()

async def photo_handler(update,context):
    if not update.effective_message or not update.effective_message.photo: return
    if deepseek_client is None:
        await update.effective_message.reply_text("Анализ фотографий сейчас недоступен: не настроен DEEPSEEK_API_KEY.")
        return
    try:
        await update.effective_chat.send_action(ChatAction.TYPING)
        photo=update.effective_message.photo[-1]
        file=await context.bot.get_file(photo.file_id)
        buf=BytesIO()
        await file.download_to_memory(out=buf)
        data=buf.getvalue()
        if len(data)>32*1024*1024:
            await update.effective_message.reply_text("Фотография слишком большая для анализа.")
            return
        b64=base64.b64encode(data).decode("ascii")
        response=await asyncio.to_thread(lambda: deepseek_client.chat.completions.create(model=DEEPSEEK_VISION_MODEL,messages=[{"role":"user","content":[{"type":"text","text":VISION_ANALYSIS_PROMPT},{"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{b64}"}}]}],temperature=0.1,max_tokens=1200))
        result=_vision_text(response)
        if not result: raise RuntimeError("Vision model returned an empty response.")
        await send_long_message(update,result,use_html=False)
    except Exception:
        logger.exception("Photo handler failed.")
        await update.effective_message.reply_text("Не удалось проанализировать фотографию. Попробуйте отправить её ещё раз.")

async def error_handler(update,context):
    logger.error("UNHANDLED TELEGRAM ERROR: %s",context.error,exc_info=context.error)

def main():
    app=Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start",start))
    app.add_handler(MessageHandler(filters.PHOTO,photo_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND,text_handler))
    app.add_error_handler(error_handler)
    logger.info("Bot started.")
    app.run_polling(drop_pending_updates=True,allowed_updates=Update.ALL_TYPES,close_loop=False)

if __name__=="__main__": main()
