import asyncio
import html
import logging
import re
from typing import Any
from supabase import Client
from config import RAG_MATCH_COUNT, RAG_MATCH_THRESHOLD, RAG_FINAL_COUNT
from embedding import get_query_embedding

logger = logging.getLogger(__name__)
WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
RU_STOPWORDS = {"и","в","во","не","что","он","она","они","мы","вы","это","как","а","но","или","ли","из","к","ко","у","за","по","для","на","с","со","от","до","о","об","про","при","же","бы","быть","есть","так","такой","такие","также","можно","нужно","должен","должны","является","какие","какой","какая","какое","когда","где","кто","чем","если","после","перед","через","согласно","требования","требование","правила","порядок"}

def _clean(value: Any) -> str:
    text = html.unescape(str(value or ""))
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def _tokens(text: str) -> set[str]:
    return {x.lower() for x in WORD_RE.findall(text.lower()) if len(x) >= 3 and x.lower() not in RU_STOPWORDS}

def _bigrams(text: str) -> set[tuple[str,str]]:
    words = [x.lower() for x in WORD_RE.findall(text.lower()) if len(x) >= 3 and x.lower() not in RU_STOPWORDS]
    return set(zip(words, words[1:]))

def _semantic(chunk: dict[str,Any]) -> float:
    try:
        return max(0.0, min(1.0, float(chunk.get("similarity", chunk.get("score", 0.0)))))
    except (TypeError, ValueError):
        return 0.0

def _lexical(query: str, text: str) -> float:
    q, c = _tokens(query), _tokens(text)
    if not q or not c:
        return 0.0
    overlap = len(q & c) / len(q)
    qb, cb = _bigrams(query), _bigrams(text)
    bigram = len(qb & cb) / len(qb) if qb else 0.0
    nq = re.sub(r"\s+", " ", query.lower()).strip()
    nc = re.sub(r"\s+", " ", text.lower())
    phrase = 1.0 if len(nq) >= 12 and nq in nc else 0.0
    return max(0.0, min(1.0, 0.70*overlap + 0.20*bigram + 0.10*phrase))

def _format(chunk: dict[str,Any]) -> str:
    doc = _clean(chunk.get("doc_name")) or "НПА"
    point = _clean(chunk.get("point_num") or chunk.get("article") or chunk.get("section") or "-")
    content = _clean(chunk.get("content"))
    return f"Документ: {doc}\nПункт/статья: {point}\nТекст:\n{content}"

def _dedupe(chunks):
    out, seen = [], set()
    for chunk in chunks:
        key = (_clean(chunk.get("doc_name")), _clean(chunk.get("point_num") or chunk.get("article") or chunk.get("section")), _clean(chunk.get("content")))
        if key[2] and key not in seen:
            seen.add(key)
            out.append(chunk)
    return out

def _local_rerank(query, chunks):
    ranked = []
    for i, chunk in enumerate(chunks):
        item = dict(chunk)
        sem = _semantic(item)
        lex = _lexical(query, _format(item))
        item["_local_score"] = 0.80*sem + 0.20*lex
        item["_semantic_score"] = sem
        item["_lexical_score"] = lex
        item["_original_index"] = i
        ranked.append(item)
    ranked.sort(key=lambda x: (x["_local_score"], x["_semantic_score"], -x["_original_index"]), reverse=True)
    return ranked[:RAG_FINAL_COUNT]

def _search(supabase: Client, vector):
    return supabase.rpc("match_npa_chunks", {"query_embedding": vector, "match_threshold": RAG_MATCH_THRESHOLD, "match_count": RAG_MATCH_COUNT}).execute().data or []

def _context(chunks):
    return "\n\n".join(f"===== ИСТОЧНИК {i} =====\n{_format(c)}" for i,c in enumerate(chunks,1))

async def retrieve_context(user_query: str, supabase: Client) -> dict[str,Any]:
    if not user_query.strip():
        return {"chunks":[],"retrieved_text":"","found":False,"candidate_count":0,"final_count":0}
    logger.info("RAG | query=%s", user_query)
    vector = await asyncio.to_thread(get_query_embedding, user_query)
    logger.info("RAG | embedding dimension=%d", len(vector))
    candidates = await asyncio.to_thread(_search, supabase, vector)
    logger.info("RAG | candidates=%d", len(candidates))
    if not candidates:
        return {"chunks":[],"retrieved_text":"","found":False,"candidate_count":0,"final_count":0}
    candidates = _dedupe(candidates)
    final = _local_rerank(user_query, candidates)
    logger.info("RAG | local reranking complete | final chunks=%d", len(final))
    return {"chunks":final,"retrieved_text":_context(final),"found":bool(final),"candidate_count":len(candidates),"final_count":len(final)}

def get_source_names(chunks):
    out=[]
    for c in chunks:
        name=_clean(c.get("doc_name")) or "НПА"
        if name not in out: out.append(name)
    return out

def get_source_references(chunks):
    out=[]; seen=set()
    for c in chunks:
        doc=_clean(c.get("doc_name")) or "НПА"
        point=_clean(c.get("point_num") or c.get("article") or c.get("section"))
        key=(doc,point)
        if key in seen: continue
        seen.add(key)
        out.append(f"• {doc} — пункт {point}" if point else f"• {doc}")
    return out
