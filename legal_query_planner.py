"""Universal legal query planner for the Belarusian legal RAG."""
from __future__ import annotations
import json, logging, re
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

def _clean(values: Any, limit: int = 8) -> List[str]:
    if not isinstance(values, list): return []
    out, seen = [], set()
    for value in values:
        value = re.sub(r"\\s+", " ", str(value or "")).strip()
        if value and value.lower() not in seen:
            seen.add(value.lower()); out.append(value)
        if len(out) >= limit: break
    return out

def _json(text: str) -> Dict[str, Any]:
    text = str(text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\\s*", "", text, flags=re.I)
        text = re.sub(r"\\s*```$", "", text)
    try: return json.loads(text)
    except Exception: pass
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start: return json.loads(text[start:end+1])
    raise ValueError("planner returned invalid JSON")

def fallback_plan(original: str, profile: Dict[str, Any], topic: str, domain: str) -> Dict[str, Any]:
    qualifiers = profile.get("qualifiers") or []
    qtype = profile.get("question_type") or "general"
    intent_map = {"frequency":"frequency","who":"responsibility","what_to_do":"procedure","whether":"permission","responsibility":"responsibility","term":"deadline","kind":"training","document":"documentation"}
    intent = "grounds" if any(x in qualifiers for x in ("suspension_for_osh_violation","suspension_for_unpassed_osh_training")) else "rights" if any(x in qualifiers for x in ("refusal_due_to_no_ppe","ppe_nonprovision_action")) else intent_map.get(qtype,"general")
    action = "suspend" if intent == "grounds" else "refuse_work" if intent == "rights" else profile.get("action") or "none"
    targets = []
    if profile.get("target_document") or profile.get("target_article"):
        targets.append({"document":profile.get("target_document") or "","article":profile.get("target_article") or "","point":""})
    phrases = _clean(profile.get("legal_phrases"), 10)
    concepts = _clean([profile.get("event"),topic,profile.get("subject"),profile.get("object"),*phrases[:4]],10)
    return {"version":"1.0","original_query":original,"jurisdiction":"BY","domain":domain or "other","intent":intent,"action":action,"subject":profile.get("subject"),"actor":profile.get("actor"),"object":profile.get("object"),"condition":profile.get("action_state"),"question_type":qtype,"legal_concepts":concepts,"legal_phrases":phrases,"search_queries":[],"keywords":_clean(concepts+phrases,12),"legal_targets":targets,"confidence":0.55,"planner_source":"deterministic"}

def _prompt(original: str, profile: Dict[str, Any], topic: str, domain: str) -> str:
    return ("Ты внутренний Legal Query Planner для законодательства Республики Беларусь.\n"
            "Не отвечай пользователю. Определи юридический смысл и поисковый план.\n"
            "Только Беларусь. Не придумывай статьи/пункты/НПА. Неизвестные legal_targets оставь пустыми.\n"
            "Верни только JSON. search_queries — поисковые фразы, legal_phrases — язык НПА.\n\n"
            f"Вопрос: {original}\nПрофиль: {json.dumps(profile, ensure_ascii=False)}\n"
            f"Тема: {topic}\nДомен: {domain}\n\n"
            "Поля: domain, intent, action, subject, actor, object, condition, question_type, legal_concepts, legal_phrases, search_queries, keywords, legal_targets, confidence.")

def build_legal_query_plan(original: str, profile: Dict[str, Any], topic: str, legal_domain: str) -> Dict[str, Any]:
    original = re.sub(r"\\s+", " ", str(original or "").strip())
    fallback = fallback_plan(original, profile, topic, legal_domain)
    if not original: return fallback
    try:
        from ai_router import generate_answer
        raw = generate_answer(_prompt(original, profile, topic, legal_domain), enforce_source_grounding=False)
        plan = _json(raw)
        allowed_domains = {"occupational_safety","fire_safety","industrial_safety","labor_law","other"}
        allowed_intents = {"requirement","prohibition","permission","grounds","procedure","responsibility","rights","deadline","frequency","training","documentation","inspection","medical","certification","comparison","definition","general"}
        allowed_actions = {"allow","prohibit","suspend","train","inspect","certify","investigate","document","report","refuse_work","verify","apply","none"}
        domain = str(plan.get("domain") or "other").lower(); intent = str(plan.get("intent") or "general").lower(); action = str(plan.get("action") or "none").lower()
        if domain not in allowed_domains: domain = "other"
        if intent not in allowed_intents: intent = "general"
        if action not in allowed_actions: action = "none"
        normalized = {"version":"1.0","original_query":original,"jurisdiction":"BY","domain":domain,"intent":intent,"action":action,"subject":plan.get("subject"),"actor":plan.get("actor"),"object":plan.get("object"),"condition":plan.get("condition"),"question_type":plan.get("question_type") or "general","legal_concepts":_clean(plan.get("legal_concepts"),10),"legal_phrases":_clean(plan.get("legal_phrases"),10),"search_queries":_clean(plan.get("search_queries"),8),"keywords":_clean(plan.get("keywords"),12),"legal_targets":plan.get("legal_targets") if isinstance(plan.get("legal_targets"),list) else [],"confidence":max(0,min(1,float(plan.get("confidence",0) or 0))),"planner_source":"llm"}
        if not fallback.get("legal_targets"): normalized["legal_targets"] = []
        if not normalized["search_queries"]: normalized["search_queries"] = _clean(normalized["legal_phrases"]+normalized["legal_concepts"],8)
        logger.info("LEGAL_QUERY_PLAN | source=llm | domain=%s | intent=%s | action=%s | targets=%s | queries=%s",domain,intent,action,normalized["legal_targets"],len(normalized["search_queries"]))
        return normalized
    except Exception as exc:
        logger.warning("LEGAL_QUERY_PLAN | LLM unavailable; deterministic fallback | error=%s", exc)
        return fallback