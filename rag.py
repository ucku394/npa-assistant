import asyncio
import hashlib
import logging
import os
import re
from typing import Any, Dict, List, Optional

from embedding import get_query_embeddings

logger = logging.getLogger(__name__)


# ============================================================
# НАСТРОЙКИ
# ============================================================

RAG_FINAL_COUNT = int(os.getenv("RAG_FINAL_COUNT", "5"))

RAG_CANDIDATE_COUNT = max(
    RAG_FINAL_COUNT * 10,
    50,
)

RAG_MIN_CANDIDATES = 10
TARGETED_SEARCH_LIMIT = 50


# ============================================================
# ОПРЕДЕЛЕНИЕ ОБЛАСТИ ЗАПРОСА
# ============================================================

def detect_legal_domain(user_query: str) -> str:
    """
    Определяет основную правовую область запроса.
    """
    query = str(user_query or "").strip().lower()

    if not query:
        return "general"

    industrial_patterns = [
        r"\bпромышленн\w*\s+безопасност\w*",
        r"\bопасн\w*\s+производственн\w*\s+объект\w*",
        r"\bпотенциально\s+опасн\w*\s+объект\w*",
        r"\bпоо\b",
        r"\bопо\b",
        r"\bавари\w*\s+на\s+опасн\w*\s+производственн\w*\s+объект\w*",
        r"\bтехническ\w*\s+расследован\w*\s+авари\w*",
        r"\bрегистрац\w*\s+опасн\w*\s+производственн\w*\s+объект\w*",
        r"\bэкспертиз\w*\s+промышленн\w*\s+безопасност\w*",
        r"\bэксплуатац\w*\s+опасн\w*\s+производственн\w*\s+объект\w*",
        r"\bлиценз\w*\s+промышленн\w*\s+безопасност\w*",
        r"\bпроизводственн\w*\s+контрол\w*",
        r"\bтехническ\w*\s+устройств\w*",
        r"\bдеклараци\w*\s+промышленн\w*\s+безопасност\w*",
    ]

    if any(re.search(pattern, query, flags=re.IGNORECASE) for pattern in industrial_patterns):
        return "industrial_safety"

    fire_patterns = [
        r"\bпожарн\w*\s+безопасност\w*",
        r"\bпротивопожарн\w*",
        r"\bпожар\w*",
        r"\bогнетушител\w*",
        r"\bэвакуац\w*",
        r"\bпожарн\w*\s+сигнализац\w*",
        r"\bсистем\w*\s+оповещен\w*\s+о\s+пожар\w*",
        r"\bвзрывопожароопасн\w*",
        r"\bпожароопасн\w*",
    ]

    if any(re.search(pattern, query, flags=re.IGNORECASE) for pattern in fire_patterns):
        return "fire_safety"

    electrical_patterns = [
        r"\bэлектробезопасност\w*",
        r"\bгрупп\w*\s+по\s+электробезопасност\w*",
        r"\bэлектроустановк\w*",
        r"\bэлектроустановк\w*\s+до\s+\d+\s*кВ",
        r"\bэлектроустановк\w*\s+свыше\s+\d+\s*кВ",
        r"\bэлектротехническ\w*\s+персонал\w*",
        r"\bэлектротехнологическ\w*\s+персонал\w*",
    ]

    if any(re.search(pattern, query, flags=re.IGNORECASE) for pattern in electrical_patterns):
        return "electrical_safety"

    occupational_patterns = [
        r"\bакт\w*\s+н[-–—]?\s*1\b",
        r"\bформа\w*\s+н[-–—]?\s*1\b",
        r"\bн[-–—]?\s*1\b",
        r"\bпострадавш\w*",
        r"\bпотерпевш\w*",
        r"\bродственник\w*",
        r"\bвруч\w*\s+акт\w*",
        r"\bутвержденн?\w*\s+акт\w*",
        r"\bокончан\w*\s+расследован\w*",
        r"\bпосле\s+окончан\w*\s+расследован\w*",
        r"\bрасследован\w*\s+на\s+производств\w*",
        r"\bучет\w*\s+несчастн\w*",
        r"\bохран\w*\s+труд\w*",
        r"\bохране\s+труда\b",
        r"\bинструктаж\w*",
        r"\bстажировк\w*",
        r"\bсредств\w*\s+индивидуальн\w*\s+защит\w*",
        r"\bсиз\b",
        r"\bспецодежд\w*",
        r"\bспецобув\w*",
        r"\bкостюм\w*",
        r"\bперчат\w*",
        r"\bвыдач\w*\s+сиз\b",
        r"\bпровер\w*\s+знани\w*\s+по\s+охран\w*\s+труд\w*",
        r"\bобучен\w*\s+по\s+охран\w*\s+труд\w*",
        r"\bинструкци\w*\s+по\s+охран\w*\s+труд\w*",
        r"\bнесчастн\w*\s+случа\w*",
        r"\bпрофессиональн\w*\s+заболеван\w*",
        r"\bмедицинск\w*\s+осмотр\w*",
        r"\bмедосмотр\w*",
        r"\bработ\w*\s+на\s+высот\w*",
        r"\bпогрузочно-разгрузочн\w*",
        r"\bбывш\w*\s+в\s+употреблен\w*",
        r"\bпериод\w*\s+использован\w*",
        r"\bсрок\w*\s+носк\w*",
        r"\bаттестаци\w*\s+рабоч\w*\s+мест\w*",
        r"\bуслов\w*\s+труд\w*",
    ]

    if any(re.search(pattern, query, flags=re.IGNORECASE) for pattern in occupational_patterns):
        return "occupational_safety"

    sanitary_patterns = [
        r"\bсанитар\w*",
        r"\bсанитарно-эпидемиологическ\w*",
        r"\bгигиен\w*",
        r"\bмикроклимат\w*",
        r"\bсанитарн\w*\s+норм\w*",
    ]

    if any(re.search(pattern, query, flags=re.IGNORECASE) for pattern in sanitary_patterns):
        return "sanitary"

    return "general"


# ============================================================
# ОПРЕДЕЛЕНИЕ ТЕМЫ
# ============================================================

def detect_topic(user_query: str) -> str:
    query = str(user_query or "").strip().lower()

    if not query:
        return "general"

    ppe_nonprovision_patterns = [
        r"\bсиз\b.*\bне\s+выдан\w*",
        r"\bне\s+выдан\w*.*\bсиз\b",
        r"\bсиз\b.*\bповрежден\w*",
        r"\bповрежден\w*.*\bсиз\b",
        r"\bсиз\b.*\bнеисправн\w*",
        r"\bнеисправн\w*.*\bсиз\b",
        r"\bне\s+обеспечен\w*.*\bсиз\b",
        r"\bсиз\b.*\bне\s+обеспечен\w*",
        r"\bбез\s+сиз\b",
        r"\bбез\s+средств\w*\s+индивидуальн\w*\s+защит\w*",
        r"\bотказ\w*.*\bработ\w*.*\bсиз\b",
        r"\bотказ\w*.*\bвыполнен\w*.*\bработ\w*",
        r"\bотказ\w*.*\bвыполнен\w*.*\bзадан\w*",
        r"\bработник\w*.*\bотказ\w*.*\bработ\w*",
        r"\bработник\w*.*\bне\s+может\s+приступ\w*",
        r"\bне\s+приступ\w*.*\bработ\w*",
        r"\bприостанов\w*.*\bработ\w*",
        r"\bправ\w*.*\bотказ\w*.*\bработ\w*",
        r"\bимеет\s+ли\s+прав\w*.*\bотказ\w*",
        r"\bперв\w*\s+шаг\w*.*\bработник\w*",
        r"\bдейств\w*.*\bработник\w*.*\bсиз\b",
        r"\bповрежд\w*\s+средств\w*\s+индивидуальн\w*\s+защит\w*",
        r"\bневыдач\w*.*\bсиз\b",
    ]

    if any(re.search(pattern, query, flags=re.IGNORECASE) for pattern in ppe_nonprovision_patterns):
        return "ppe_nonprovision"

    medical_exam_patterns = [
        r"\bмедицинск\w*\s+осмотр\w*",
        r"\bмедицинск\w*\s+освидетельствован\w*",
        r"\bмедосмотр\w*",
        r"\bпредварительн\w*\s+медицинск\w*",
        r"\bпериодическ\w*\s+медицинск\w*",
        r"\bвнеочередн\w*\s+медицинск\w*",
        r"\bобязательн\w*\s+медицинск\w*",
        r"\bосмотр\w*\s+при\s+поступлен\w*\s+на\s+работ\w*",
        r"\bмедицинск\w*\s+осмотр\w*\s+работник\w*",
        r"\bмедицинск\w*\s+осмотр\w*\s+работающ\w*",
    ]

    if any(re.search(pattern, query, flags=re.IGNORECASE) for pattern in medical_exam_patterns):
        return "medical_examinations"

    workplace_attestation_patterns = [
        r"\bаттестаци\w*\s+рабоч\w*\s+мест\w*",
        r"\bоценк\w*\s+услов\w*\s+труд\w*",
        r"\bуслов\w*\s+труд\w*\s+при\s+аттестаци\w*",
        r"\bаттестаци\w*\s+по\s+услов\w*\s+труд\w*",
        r"\bоснован\w*\s+для\s+проведен\w*\s+аттестаци\w*",
        r"\bпроведен\w*\s+аттестаци\w*\s+рабоч\w*\s+мест\w*",
        r"\bпериодичност\w*\s+аттестаци\w*",
        r"\bкомисс\w*\s+по\s+аттестаци\w*",
        r"\bрезультат\w*\s+аттестаци\w*\s+рабоч\w*\s+мест\w*",
    ]

    if any(re.search(pattern, query, flags=re.IGNORECASE) for pattern in workplace_attestation_patterns):
        return "workplace_attestation"

    occupational_briefing_patterns = [
        r"\bвводн\w*\s+инструктаж\w*",
        r"\bкто\s+провод\w*.*\bинструктаж\w*",
        r"\bпровод\w*.*\bвводн\w*\s+инструктаж\w*",
        r"\bлиц\w*.*\bпровод\w*.*\bвводн\w*\s+инструктаж\w*",
        r"\bответственн\w*.*\bвводн\w*\s+инструктаж\w*",
        r"\bспециалист\w*\s+по\s+охран\w*\s+труд\w*.*\bинструктаж\w*",
        r"\bразов\w*\s+работ\w*.*\bне\s+связан\w*.*\bпрям\w*\s+обязанност\w*",
        r"\bне\s+связан\w*\s+с\s+прям\w*\s+обязанност\w*",
        r"\bпрям\w*\s+обязанност\w*.*\bразов\w*\s+работ\w*",
        r"\bнаряд\w*[-–—]?\s*допуск\w*",
        r"\bнаряд\w*\s+допуск\w*",
        r"\bцелев\w*\s+инструктаж\w*",
    ]

    if any(re.search(pattern, query, flags=re.IGNORECASE) for pattern in occupational_briefing_patterns):
        return "occupational_briefing"

    occupational_training_patterns = [
        r"\bстажиров\w*",
        r"\bпродолжительност\w*\s+стажиров\w*",
        r"\bсрок\w*\s+стажиров\w*",
        r"\bминимальн\w*.*\bстажиров\w*",
        r"\bне\s+менее\s+двух\b.*\bрабоч\w*",
        r"\bрабоч\w*\s+дн\w*.*\bстажиров\w*",
        r"\bрабоч\w*\s+смен\w*.*\bстажиров\w*",
        r"\bповышенн\w*\s+опасност\w*.*\bстажиров\w*",
        r"\bстажиров\w*.*\bповышенн\w*\s+опасност\w*",
        r"\bдопуск\w*\s+к\s+самостоятельн\w*\s+работ\w*.*\bстажиров\w*",
        r"\bстажиров\w*.*\bсамостоятельн\w*\s+работ\w*",
        r"\b№\s*175\b",
        r"\bинструкци\w*\s*№?\s*175\b",
    ]

    if any(re.search(pattern, query, flags=re.IGNORECASE) for pattern in occupational_training_patterns):
        return "occupational_training"

    accident_patterns = [
        r"\bнесчастн\w*\s+случа\w*",
        r"\bгруппов\w*\s+несчастн\w*\s+случа\w*",
        r"\bгруппов\w*\s+несчастн\w*",
        r"\bрасследован\w*\s+несчастн\w*",
        r"\bучет\w*\s+несчастн\w*",
        r"\bпотерпевш\w*",
        r"\bтравм\w*\s+на\s+производств\w*",
        r"\bтравм\w*\s+работник\w*",
        r"\bпроисшеств\w*\s+на\s+производств\w*",
        r"\bакт\w*\s+н[-–—]?\s*1\b",
        r"\bформа\w*\s+н[-–—]?\s*1\b",
        r"\bн[-–—]?\s*1\b",
        r"\bпострадавш\w*",
        r"\bродственник\w*",
        r"\bвруч\w*\s+акт\w*",
        r"\bутвержденн?\w*\s+акт\w*",
        r"\bокончан\w*\s+расследован\w*",
        r"\bпосле\s+окончан\w*\s+расследован\w*",
    ]

    if any(re.search(pattern, query, flags=re.IGNORECASE) for pattern in accident_patterns):
        return "accident_investigation"

    return "general"


# ============================================================
# СПЕЦИАЛЬНЫЕ КАТЕГОРИИ РАБОТНИКОВ
# ============================================================

def detect_special_category(user_query: str) -> Optional[str]:
    """Определяет специальную категорию работника для точечного поиска."""
    query = re.sub(r"\s+", " ", str(user_query or "").strip().lower())
    if not query:
        return None

    minor_patterns = [
        r"\bнесовершеннолетн\w*",
        r"\bлиц\w*\s+до\s+18\s+лет\b",
        r"\bлиц\w*\s+моложе\s+18\s+лет\b",
        r"\bработник\w*\s+моложе\s+18\s+лет\b",
        r"\bработник\w*\s+моложе\s+восемнадцат\w*\s+лет\b",
        r"\bдо\s+восемнадцат\w*\s+лет\b",
        r"\bмоложе\s+восемнадцат\w*\s+лет\b",
        r"\b\d{1,2}[-–—]?летн\w*\s+работник\w*",
        r"\bработник\w*\s+\d{1,2}[-–—]?летн\w*",
        r"\b\d{1,2}\s*летн\w*\b",
    ]
    if any(re.search(pattern, query, flags=re.IGNORECASE) for pattern in minor_patterns):
        return "minor"
    return None


def _minor_special_issue(user_query: str) -> Optional[str]:
    """Определяет конкретное ограничение для несовершеннолетнего работника."""
    query = re.sub(r"\s+", " ", str(user_query or "").strip().lower())

    holiday_markers = (
        "государственн",
        "праздничн",
        "нерабоч",
        "выходн",
    )
    if any(marker in query for marker in holiday_markers) and (
        "дн" in query or "праздник" in query or "выходн" in query
    ):
        return "holiday_weekend"

    if re.search(r"\bночн\w*|\bсверхурочн\w*", query, flags=re.IGNORECASE):
        return "night_overtime"

    if re.search(
        r"\bтяжел\w*\s+работ\w*|\bвредн\w*\s+услов\w*|\bопасн\w*\s+услов\w*|\bподземн\w*|\bгорн\w*\s+работ\w*",
        query,
        flags=re.IGNORECASE,
    ):
        return "prohibited_work"

    if re.search(r"\bмедицинск\w*\s+осмотр\w*|\bмедосмотр\w*", query, flags=re.IGNORECASE):
        return "medical"

    if re.search(r"\bотпуск\w*", query, flags=re.IGNORECASE):
        return "leave"

    if re.search(r"\bрабоч\w*\s+врем\w*|\bсокращенн\w*\s+продолжительност\w*", query, flags=re.IGNORECASE):
        return "work_time"

    return None


# ============================================================
# КЛЮЧЕВЫЕ СЛОВА
# ============================================================

def _extract_query_terms(user_query: str) -> List[str]:
    query = str(user_query or "").strip().lower()
    terms: List[str] = []

    keyword_patterns = [
        r"\bсиз\b",
        r"\bсредств\w*\s+индивидуальн\w*\s+защит\w*",
        r"\bспецодежд\w*",
        r"\bспецобув\w*",
        r"\bкостюм\w*",
        r"\bперчат\w*",
        r"\bне\s+выдан\w*",
        r"\bневыдач\w*",
        r"\bповрежден\w*",
        r"\bповрежд\w*",
        r"\bнеисправн\w*",
        r"\bне\s+обеспечен\w*",
        r"\bбез\s+сиз\b",
        r"\bбез\s+средств\w*\s+индивидуальн\w*\s+защит\w*",
        r"\bотказ\w*",
        r"\bне\s+приступ\w*",
        r"\bприостанов\w*",
        r"\bработник\w*",
        r"\bдейств\w*",
        r"\bперв\w*\s+шаг\w*",
        r"\bправ\w*",
        r"\bобеспечен\w*",
        r"\bвыдач\w*",
        r"\bисправн\w*",
        r"\bзащитн\w+\s+свойств\w*",
        r"\bмедицинск\w*\s+осмотр\w*",
        r"\bмедицинск\w*",
        r"\bмедосмотр\w*",
        r"\bпредварительн\w*",
        r"\bпериодическ\w*",
        r"\bвнеочередн\w*",
        r"\bобязательн\w*",
        r"\bработающ\w*",
        r"\bза\s+чей\s+счет\b",
        r"\bза\s+чей\s+сч[её]т\b",
        r"\bза\s+сч[её]т\b",
        r"\bсчет\b",
        r"\bоплат\w*",
        r"\bфинанс\w*",
        r"\bрасход\w*",
        r"\bзатрат\w*",
        r"\bсредств\w*",
        r"\bиспользован\w*",
        r"\bбывш\w*\s+в\s+употреблен\w*",
        r"\bпериод\w*\s+использован\w*",
        r"\bсрок\w*\s+носк\w*",
        r"\bаттестаци\w*",
        r"\bрабоч\w*\s+мест\w*",
        r"\bуслов\w*\s+труд\w*",
        r"\bоснован\w*",
        r"\bпроведен\w*",
        r"\bпериодичност\w*",
        r"\bрезультат\w*",
        r"\bкомисс\w*",
        r"\bвредн\w*",
        r"\bопасн\w*",
        r"\bнесчастн\w*\s+случа\w*",
        r"\bрасследован\w*",
        r"\bучет\w*",
        r"\bакт\w*\s+н[-–—]?\s*1\b",
        r"\bформа\w*\s+н[-–—]?\s*1\b",
        r"\bн[-–—]?\s*1\b",
        r"\bпострадавш\w*",
        r"\bпотерпевш\w*",
        r"\bродственник\w*",
        r"\bвруч\w*",
        r"\bутвержденн?\w*",
        r"\bокончан\w*\s+расследован\w*",
        r"\bрабоч\w*\s+дн\w*",
        r"\bсрок\w*",
        r"\bстажиров\w*",
        r"\bпродолжительност\w*\s+стажиров\w*",
        r"\bне\s+менее\s+двух\b",
        r"\bрабоч\w*\s+смен\w*",
        r"\bповышенн\w*\s+опасност\w*",
        r"\bсамостоятельн\w*\s+работ\w*",
        r"\bдопуск\w*",
        r"\bпровер\w*\s+знан\w*",
        r"\b№\s*175\b",
        r"\bпромышленн\w*\s+безопасност\w*",
        r"\bопасн\w*\s+производственн\w*\s+объект\w*",
        r"\bопо\b",
        r"\bпоо\b",
        r"\bпроизводственн\w*\s+контрол\w*",
        r"\bтехническ\w*\s+устройств\w*",
        r"\bавари\w*",
        r"\bинцидент\w*",
        r"\bпожарн\w*\s+безопасност\w*",
        r"\bпожар\w*",
        r"\bогнетушител\w*",
        r"\bэвакуац\w*",
    ]

    for pattern in keyword_patterns:
        matches = re.findall(pattern, query, flags=re.IGNORECASE)
        for match in matches:
            value = str(match).strip().lower()
            if value and value not in terms:
                terms.append(value)

    return terms


def _keyword_score(
    chunk: Dict[str, Any],
    query_terms: List[str],
) -> float:
    if not query_terms:
        return 0.0

    document_name = str(chunk.get("doc_name") or chunk.get("document") or "").lower()
    content = str(chunk.get("content") or chunk.get("text") or "").lower()
    point = str(chunk.get("point_num") or "").lower()

    text = f"{document_name} {content} {point}"
    matched = sum(1 for term in query_terms if term in text)
    return matched / len(query_terms)


# ============================================================
# СПЕЦИАЛЬНЫЙ SCORE ДЛЯ ЮРИДИЧЕСКОЙ ТЕМЫ
# ============================================================

def _topic_relevance_score(
    chunk: Dict[str, Any],
    topic: str,
) -> float:
    if not topic or topic == "general":
        return 0.0

    document_name = str(chunk.get("doc_name") or chunk.get("document") or "").lower()
    content = str(chunk.get("content") or chunk.get("text") or "").lower()
    db_topic = str(chunk.get("topic") or "").lower()

    score = 0.0

    if topic == "occupational_briefing":
        if db_topic == "occupational_briefing":
            score += 1.0

        target_instruction_markers = [
            "целевой инструктаж",
            "разовых работ",
            "не связанных с прямыми обязанностями",
            "прямыми обязанностями",
            "наряд-допуск",
            "наряду-допуску",
        ]
        target_matches = sum(1 for marker in target_instruction_markers if marker in content)
        if target_matches:
            score += min(target_matches * 0.45, 1.80)

        briefing_markers = [
            "вводный инструктаж",
            "проводит инструктаж",
            "специалист по охране труда",
            "уполномоченное должностное лицо",
            "руководитель организации",
            "руководитель структурного подразделения",
        ]
        matches = sum(1 for marker in briefing_markers if marker in content or marker in document_name)
        if matches:
            score += min(matches * 0.30, 1.20)

        if "175" in document_name:
            score += 0.80

    elif topic == "ppe_nonprovision":
        if db_topic == "ppe_nonprovision":
            score += 1.0

        is_npa_209 = bool(
            re.search(r"№\s*209\b", document_name, flags=re.IGNORECASE)
            or re.search(r"\b209\b", document_name, flags=re.IGNORECASE)
        )
        if is_npa_209:
            score += 1.20

        if "сиз" in document_name or (
            "средств" in document_name
            and "индивидуальн" in document_name
            and "защит" in document_name
        ):
            score += 0.70

        if "сиз" in content:
            score += 0.35
        if "средств" in content and "индивидуальн" in content and "защит" in content:
            score += 0.35

        problem_markers = [
            "не выдан",
            "невыдач",
            "поврежден",
            "поврежд",
            "неисправн",
            "не обеспечен",
            "отказ",
            "не приступ",
            "приостанов",
        ]
        problem_matches = sum(1 for marker in problem_markers if marker in content)
        if problem_matches:
            score += min(problem_matches * 0.25, 0.75)

    elif topic == "medical_examinations":
        if db_topic == "medical_examinations":
            score += 1.0

        if (
            re.search(r"№\s*74\b", document_name, flags=re.IGNORECASE)
            or re.search(r"\b74\b", document_name, flags=re.IGNORECASE)
        ):
            score += 1.20

        if "медицинск" in document_name and "осмотр" in document_name:
            score += 0.80

        if "медицинск" in content:
            score += 0.30
        if "осмотр" in content:
            score += 0.30

        financing_markers = ["за счет", "за счёт", "оплата", "расход", "средств", "финанс", "затрат"]
        if any(marker in content for marker in financing_markers):
            score += 0.60

    elif topic == "workplace_attestation":
        if db_topic == "workplace_attestation":
            score += 1.0
        if "аттестаци" in document_name:
            score += 0.80
        if "аттестаци" in content:
            score += 0.35
        if "рабоч" in content and "мест" in content:
            score += 0.15

        attestation_mode = _attestation_query_mode(
            str(chunk.get("_user_query_for_scoring") or "")
        )
        if attestation_mode == "periodicity":
            if "один раз в пять лет" in content:
                score += 1.20
            if "срок действия результатов аттестации" in content:
                score += 0.90
            if "очередн" in content and "аттестаци" in content:
                score += 0.35
            if re.search(r"\b19\b", str(chunk.get("point_num") or "")):
                score += 0.90

        if "253" in document_name and (
            "аттестаци" in document_name or "аттестаци" in content or "рабоч" in content
        ):
            score += 1.50

    elif topic == "accident_investigation":
        if db_topic == "accident_investigation":
            score += 1.0
        if "несчаст" in document_name:
            score += 0.80
        if "расследован" in document_name:
            score += 0.45
        if "расследован" in content:
            score += 0.35
        if "несчаст" in content:
            score += 0.25

        if re.search(r"\bн[-–—]?\s*1\b", content, re.IGNORECASE):
            score += 0.90
        if "акт" in content and re.search(r"\bн[-–—]?\s*1\b", content, re.IGNORECASE):
            score += 0.55

        accident_markers = [
            "пострадавш",
            "потерпевш",
            "родственник",
            "вруч",
            "утвержден",
            "утверждён",
            "рабочих дней",
            "рабочие дни",
            "окончани",
        ]
        marker_matches = sum(1 for marker in accident_markers if marker in content)
        if marker_matches:
            score += min(marker_matches * 0.15, 0.75)

    return min(max(score, 0.0), 2.5)


# ============================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_identifier(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^A-Za-zА-Яа-яЁё0-9_./-]+", "", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def _normalize_point_identifier(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    text = re.sub(r"[.,;:]+$", "", text)
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^A-Za-zА-Яа-яЁё0-9_.-]+", "", text)
    text = re.sub(r"_+", "_", text)
    text = re.sub(r"[.,;:]+$", "", text)
    return text.strip("_")


def _normalize_source_id(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    text = re.sub(r"[.,;:]+$", "", text)
    return _normalize_identifier(text)


def _get_document_name(chunk: Dict[str, Any]) -> str:
    for key in (
        "doc_name",
        "document",
        "document_name",
        "doc",
        "title",
        "npa_name",
        "source",
    ):
        value = chunk.get(key)
        if value:
            return str(value).strip()
    return "Неизвестный НПА"


def _get_point_number(chunk: Dict[str, Any]) -> str:
    for key in (
        "point_num",
        "point",
        "point_number",
        "article",
        "article_number",
        "paragraph",
        "section",
    ):
        value = chunk.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()

    text = chunk.get("content") or chunk.get("text") or chunk.get("chunk_text") or ""
    text = str(text).strip()
    if not text:
        return ""

    patterns = [
        r"\bпункт(?:а|ом)?\s+([0-9]+(?:\.[0-9]+)*)\b",
        r"\bп\.\s*([0-9]+(?:\.[0-9]+)*)\b",
        r"\bстатья\s+([0-9]+(?:\.[0-9]+)*)\b",
        r"\bст\.\s*([0-9]+(?:\.[0-9]+)*)\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1)

    return ""


def _extract_npa_number(document_name: str) -> str:
    if not document_name:
        return ""
    patterns = [
        r"№\s*([0-9]+(?:[-/][A-Za-zА-Яа-я0-9]+)*)",
        r"N\s*([0-9]+(?:[-/][A-Za-zА-Яа-я0-9]+)*)",
    ]
    for pattern in patterns:
        match = re.search(pattern, document_name, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return ""


def build_source_id(
    chunk: Dict[str, Any],
    index: int = 0,
) -> str:
    existing_source_id = chunk.get("source_id") or chunk.get("_source_id")
    if existing_source_id:
        return _normalize_source_id(existing_source_id)

    document_name = _get_document_name(chunk)
    point = _get_point_number(chunk)
    npa_number = _extract_npa_number(document_name)

    if npa_number:
        base = f"NPA_{npa_number}"
    else:
        normalized_document = _normalize_identifier(document_name)
        base = (
            f"NPA_{normalized_document[:80]}"
            if normalized_document
            else "NPA_UNKNOWN"
        )

    if point:
        normalized_point = _normalize_point_identifier(point)
        if normalized_point:
            return f"{base}_P{normalized_point}"

    text = chunk.get("content") or chunk.get("text") or chunk.get("chunk_text") or ""
    stable_string = f"{document_name}|{str(text).strip()}"
    digest = hashlib.sha1(stable_string.encode("utf-8")).hexdigest()[:12]

    return f"{base}_H{digest}"


# ============================================================
# ФОРМИРОВАНИЕ КОНТЕКСТА
# ============================================================

def _format_chunk(
    chunk: Dict[str, Any],
    index: int,
) -> str:
    document_name = _get_document_name(chunk)
    point = _get_point_number(chunk)
    text = chunk.get("content") or chunk.get("text") or chunk.get("chunk_text") or ""

    source_id = build_source_id(chunk, index)
    chunk["_source_id"] = source_id

    legal_domain = chunk.get("legal_domain") or ""
    topic = chunk.get("topic")
    semantic = _semantic_score(chunk)
    combined = chunk.get("_combined_score")

    lines = [
        f"SOURCE_ID: {source_id}",
        f"DOCUMENT: {document_name}",
    ]

    if legal_domain:
        lines.append(f"LEGAL_DOMAIN: {legal_domain}")
    if topic:
        lines.append(f"TOPIC: {topic}")
    if point:
        lines.append(f"POINT_OR_ARTICLE: {point}")

    lines.append(f"SEMANTIC_SCORE: {semantic:.4f}")
    if combined is not None:
        lines.append(f"RAG_SCORE: {float(combined):.4f}")

    lines.append("TEXT:")
    lines.append(str(text).strip())

    return "\n".join(lines)


def _build_retrieved_text(
    chunks: List[Dict[str, Any]],
) -> str:
    blocks = []
    for index, chunk in enumerate(chunks, start=1):
        source_id = build_source_id(chunk, index)
        chunk["_source_id"] = source_id
        block = [
            f"===== RAG SOURCE {index} =====",
            _format_chunk(chunk, index),
            f"===== END RAG SOURCE {index} =====",
        ]
        blocks.append("\n".join(block))
    return "\n\n".join(blocks)


# ============================================================
# SUPABASE — VECTOR SEARCH
# ============================================================

def _search_chunks(
    supabase,
    query_vector: List[float],
    search_query: str,
    legal_domain: Optional[str],
    topic_filter: Optional[str],
) -> List[Dict[str, Any]]:
    try:
        response = (
            supabase.rpc(
                "hybrid_search_npa_chunks",
                {
                    "query_embedding": query_vector,
                    "search_query": search_query,
                    "match_count": RAG_CANDIDATE_COUNT,
                    "domain_filter": legal_domain,
                    "topic_filter": topic_filter,
                    "semantic_limit": max(RAG_CANDIDATE_COUNT, 40),
                    "lexical_limit": max(RAG_CANDIDATE_COUNT, 40),
                    "rrf_k": 60,
                },
            ).execute()
        )

        data = response.data or []
        for chunk in data:
            chunk["similarity"] = _safe_float(
                chunk.get("semantic_score"),
                _safe_float(chunk.get("similarity")),
            )
            chunk["_hybrid_rrf_score"] = _safe_float(chunk.get("rrf_score"))
            chunk["_hybrid_final_score"] = _safe_float(chunk.get("final_score"))

        return data

    except Exception as exc:
        logger.warning(
            "RAG | hybrid search failed; fallback to semantic_search_npa_chunks | error=%s",
            exc,
        )

        response = (
            supabase.rpc(
                "semantic_search_npa_chunks",
                {
                    "query_embedding": query_vector,
                    "match_count": RAG_CANDIDATE_COUNT,
                    "domain_filter": legal_domain,
                    "topic_filter": topic_filter,
                },
            ).execute()
        )

        data = response.data or []
        for chunk in data:
            chunk["similarity"] = _safe_float(chunk.get("similarity"))
            chunk["_hybrid_rrf_score"] = 0.0
            chunk["_hybrid_final_score"] = 0.0

        return data


# ============================================================
# TARGETED SEARCH (СГРУППИРОВАННЫЙ В ОДИН ЗАПРОС)
# ============================================================

def _execute_combined_targeted_search(
    supabase,
    filters_list: List[str],
    domain: str = "occupational_safety",
    limit: int = TARGETED_SEARCH_LIMIT,
) -> List[Dict[str, Any]]:
    """
    Выполняет один сгруппированный запрос через .or_()
    вместо десятка последовательных вызовов.
    """
    if not filters_list:
        return []

    combined_query = ",".join(filters_list)

    try:
        response = (
            supabase.table("npa_chunks")
            .select(
                "doc_name,"
                "doc_type,"
                "point_num,"
                "content,"
                "legal_domain,"
                "topic,"
                "source_url"
            )
            .eq("legal_domain", domain)
            .or_(combined_query)
            .limit(limit)
            .execute()
        )
        return response.data or []
    except Exception as exc:
        logger.warning("RAG | combined targeted search failed: %s", exc)
        return []


def _attestation_query_mode(user_query: str) -> str:
    """
    Определяет узкий тип вопроса внутри темы аттестации рабочих мест.

    Это не новая правовая тема. Это поисковый intent, позволяющий не
    смешивать вопрос о периодичности с вопросами о результатах,
    комиссии, основаниях и внеочередной аттестации.
    """
    query = re.sub(r"\s+", " ", str(user_query or "").strip().lower())

    if not query:
        return "general"

    periodicity_patterns = [
        r"\bс\s+какой\s+периодичност\w*\b",
        r"\bкак\s+часто\b",
        r"\bпериодичност\w*\b",
        r"\bодин\s+раз\s+в\s+пять\s+лет\b",
        r"\bсрок\s+действ\w*\s+результат\w*\s+аттестаци\w*\b",
        r"\bсрок\w*\s+действ\w*\s+результат\w*\b",
        r"\bочередн\w*\s+аттестаци\w*\b",
    ]
    if any(re.search(pattern, query, flags=re.IGNORECASE) for pattern in periodicity_patterns):
        return "periodicity"

    extraordinary_patterns = [
        r"\bвнеочередн\w*\s+аттестаци\w*\b",
        r"\bпереаттестаци\w*\b",
        r"\bв\s+течение\s+шести\s+месяц\w*\b",
    ]
    if any(re.search(pattern, query, flags=re.IGNORECASE) for pattern in extraordinary_patterns):
        return "extraordinary"

    results_patterns = [
        r"\bрезультат\w*\s+аттестаци\w*\b",
        r"\bдополнительн\w*\s+отпуск\w*\b",
        r"\bдоплат\w*\b",
        r"\bпенсион\w*\s+страхован\w*\b",
    ]
    if any(re.search(pattern, query, flags=re.IGNORECASE) for pattern in results_patterns):
        return "results"

    commission_patterns = [
        r"\bкомисс\w*\s+по\s+аттестаци\w*\b",
        r"\bсостав\w*\s+комисс\w*\b",
        r"\bкто\s+входит\s+в\s+комисс\w*\b",
    ]
    if any(re.search(pattern, query, flags=re.IGNORECASE) for pattern in commission_patterns):
        return "commission"

    return "general"


def _targeted_attestation_search(
    supabase,
    user_query: str = "",
) -> List[Dict[str, Any]]:
    """
    Узкий lexical-search для аттестации рабочих мест.

    Для вопроса о периодичности специально ищем норму п. 19 Положения
    о порядке проведения аттестации рабочих мест по условиям труда.
    """
    mode = _attestation_query_mode(user_query)

    if mode == "periodicity":
        # Для периодичности нельзя использовать общий OR-запрос с
        # doc_name.ilike.%253%: он совпадает со всеми chunks НПА №253
        # и может вытеснить п. 19 из лимита. Сначала ограничиваемся №253,
        # затем ищем внутри него признаки нормы о пятилетнем сроке.
        try:
            response = (
                supabase.table("npa_chunks")
                .select(
                    "doc_name,"
                    "doc_type,"
                    "point_num,"
                    "content,"
                    "legal_domain,"
                    "topic,"
                    "source_url"
                )
                .eq("legal_domain", "occupational_safety")
                .ilike("doc_name", "%253%")
                .or_(
                    "content.ilike.%срок действия результатов аттестации составляет пять лет%,"
                    "content.ilike.%один раз в пять лет%,"
                    "content.ilike.%приказ об утверждении очередной аттестации%,"
                    "point_num.ilike.%19%"
                )
                .limit(TARGETED_SEARCH_LIMIT)
                .execute()
            )
            return _deduplicate_chunks(response.data or [])
        except Exception as exc:
            logger.warning(
                "RAG | attestation periodicity targeted search failed: %s",
                exc,
            )
            return []
    elif mode == "extraordinary":
        queries = [
            "doc_name.ilike.%253%",
            "content.ilike.%внеочередная аттестация%",
            "content.ilike.%переаттестация%",
            "content.ilike.%в течение шести месяцев%",
            "point_num.ilike.%17%",
        ]
    elif mode == "results":
        queries = [
            "doc_name.ilike.%253%",
            "content.ilike.%результаты аттестации%",
            "content.ilike.%дополнительный отпуск%",
            "content.ilike.%профессиональное пенсионное страхование%",
        ]
    elif mode == "commission":
        queries = [
            "doc_name.ilike.%253%",
            "content.ilike.%комиссия по проведению аттестации%",
            "content.ilike.%комиссия%",
        ]
    else:
        queries = [
            "doc_name.ilike.%253%",
            "doc_name.ilike.%аттестаци%",
            "content.ilike.%аттестаци%",
        ]

    results = _execute_combined_targeted_search(supabase, queries)
    return _deduplicate_chunks(results)


def _targeted_medical_exam_search(supabase) -> List[Dict[str, Any]]:
    queries = [
        "doc_name.ilike.%74%",
        "doc_name.ilike.%медицинск%",
        "doc_name.ilike.%осмотр%",
        "content.ilike.%медицинск%",
        "content.ilike.%медосмотр%",
    ]
    results = _execute_combined_targeted_search(supabase, queries)
    return _deduplicate_chunks(results)


def _targeted_ppe_nonprovision_search(supabase) -> List[Dict[str, Any]]:
    queries = [
        "doc_name.ilike.%209%",
        "doc_name.ilike.%СИЗ%",
        "doc_name.ilike.%средств%индивидуальн%защит%",
        "content.ilike.%не выдан%",
        "content.ilike.%невыдач%",
        "content.ilike.%поврежден%",
        "content.ilike.%поврежд%",
        "content.ilike.%неисправн%",
        "content.ilike.%не обеспечен%",
        "content.ilike.%отказ%",
        "content.ilike.%не приступ%",
        "content.ilike.%приостанов%",
        "content.ilike.%работник%",
    ]
    results = _execute_combined_targeted_search(supabase, queries)
    return _deduplicate_chunks(results)


def _targeted_occupational_training_search(supabase) -> List[Dict[str, Any]]:
    queries = [
        "doc_name.ilike.%175%",
        "doc_name.ilike.%Инструкци%",
        "content.ilike.%стажиров%",
        "content.ilike.%продолжительн%стажиров%",
        "content.ilike.%срок%стажиров%",
        "content.ilike.%не менее двух%",
        "content.ilike.%рабочих дней%",
        "content.ilike.%рабочих смен%",
        "content.ilike.%повышенной опасностью%",
        "content.ilike.%допуск к самостоятельной работе%",
        "content.ilike.%самостоятельной работе%",
        "content.ilike.%проверка знаний%",
    ]
    results = _execute_combined_targeted_search(supabase, queries)
    return _deduplicate_chunks(results)


def _targeted_occupational_briefing_search(
    supabase,
    user_query: str = "",
) -> List[Dict[str, Any]]:
    target_mode = _is_target_briefing_query(user_query)
    responsible_mode = not target_mode and _is_responsible_briefing_query(user_query)

    if target_mode:
        queries = [
            "doc_name.ilike.%175%",
            "doc_name.ilike.%Инструкци%",
            "content.ilike.%целевой инструктаж%",
            "content.ilike.%разовых работ%",
            "content.ilike.%не связанных с прямыми обязанностями%",
            "content.ilike.%прямыми обязанностями%",
            "content.ilike.%наряд-допуск%",
            "content.ilike.%наряду-допуску%",
            "content.ilike.%наряд допуск%",
        ]
    elif responsible_mode:
        queries = [
            "doc_name.ilike.%175%",
            "doc_name.ilike.%Инструкци%",
            "content.ilike.%вводный инструктаж%",
            "content.ilike.%проводит специалист по охране труда%",
            "content.ilike.%специалист по охране труда%",
            "content.ilike.%уполномоченное должностное лицо%",
            "content.ilike.%руководитель организации%",
            "content.ilike.%руководитель структурного подразделения%",
        ]
    else:
        queries = [
            "doc_name.ilike.%175%",
            "doc_name.ilike.%Инструкци%",
            "content.ilike.%инструктаж%",
        ]

    results = _execute_combined_targeted_search(supabase, queries)
    return _deduplicate_chunks(results)


def _accident_query_mode(user_query: str) -> str:
    """Определяет узкий поисковый intent внутри темы расследования НС."""
    query = re.sub(r"\s+", " ", str(user_query or "").strip().lower())

    if not query:
        return "general"

    worker_did_not_report_patterns = [
        r"\bне\s+сообщил\w*\b.*\bнесчастн\w*\s+случа\w*\b",
        r"\bнесчастн\w*\s+случа\w*\b.*\bне\s+сообщил\w*\b",
        r"\bне\s+сообщил\w*\b.*\bруководител\w*\b",
        r"\bне\s+сообщил\w*\b.*\bнанимател\w*\b",
        r"\bне\s+сообщил\w*\b.*\bначальник\w*\b",
        r"\bпотерпевш\w*\b.*\bне\s+сообщил\w*\b",
        r"\bработник\w*\b.*\bне\s+сообщил\w*\b",
        r"\bнесообщен\w*\b.*\bнесчастн\w*\b",
    ]
    if any(re.search(pattern, query, flags=re.IGNORECASE) for pattern in worker_did_not_report_patterns):
        return "worker_did_not_report"

    group_patterns = [
        r"\bгруппов\w*\s+несчастн\w*\s+случа\w*\b",
        r"\bдвух\s+и\s+более\b.*\bпострадавш\w*\b",
    ]
    if any(re.search(pattern, query, flags=re.IGNORECASE) for pattern in group_patterns):
        return "group_accident"

    n1_patterns = [
        r"\bакт\w*\s+н[-–—]?\s*1\b",
        r"\bформа\w*\s+н[-–—]?\s*1\b",
        r"\bвруч\w*\s+н[-–—]?\s*1\b",
    ]
    if any(re.search(pattern, query, flags=re.IGNORECASE) for pattern in n1_patterns):
        return "n1"

    return "general"


def _targeted_accident_worker_not_report_search(supabase) -> List[Dict[str, Any]]:
    """Точечный lexical-search для ситуации, когда работник не сообщил о НС."""
    try:
        response = (
            supabase.table("npa_chunks")
            .select(
                "doc_name,"
                "doc_type,"
                "point_num,"
                "content,"
                "legal_domain,"
                "topic,"
                "source_url"
            )
            .eq("legal_domain", "occupational_safety")
            .or_("doc_name.ilike.%30%,doc_name.ilike.%81 144%")
            .or_(
                "content.ilike.%не сообщил%,"
                "content.ilike.%не сообщила%,"
                "content.ilike.%сообщить о несчастном случае%,"
                "content.ilike.%сообщить о происшедшем несчастном случае%,"
                "content.ilike.%сообщить непосредственному руководителю%,"
                "content.ilike.%сообщить руководителю%,"
                "content.ilike.%немедленно сообщить%,"
                "content.ilike.%обязан сообщить%"
            )
            .limit(TARGETED_SEARCH_LIMIT)
            .execute()
        )
        return _deduplicate_chunks(response.data or [])
    except Exception as exc:
        logger.warning(
            "RAG | accident worker-not-report targeted search failed: %s",
            exc,
        )
        return []


def _targeted_accident_search(supabase, user_query: str = "") -> List[Dict[str, Any]]:
    mode = _accident_query_mode(user_query)

    if mode == "worker_did_not_report":
        return _targeted_accident_worker_not_report_search(supabase)

    queries = [
        "doc_name.ilike.%Правила%",
        "doc_name.ilike.%30%",
        "content.ilike.%группов%",
        "content.ilike.%одновременно с двумя%",
        "content.ilike.%два и более%",
        "content.ilike.%специальному расследованию%",
        "content.ilike.%немедленно сообщает%",
        "content.ilike.%сообщает%",
        "content.ilike.%уведом%",
        "content.ilike.%прокуратур%",
        "content.ilike.%государственной инспекции труда%",
        "doc_name.ilike.%несчаст%",
        "doc_name.ilike.%расслед%",
        "content.ilike.%Н-1%",
        "content.ilike.%н-1%",
        "content.ilike.%форма Н-1%",
        "content.ilike.%форма н-1%",
        "content.ilike.%акт Н-1%",
        "content.ilike.%акт н-1%",
        "content.ilike.%пострадавш%",
        "content.ilike.%потерпевш%",
        "content.ilike.%родственник%",
        "content.ilike.%вруч%",
        "content.ilike.%утвержден%",
        "content.ilike.%утверждён%",
        "content.ilike.%рабочих дней%",
        "content.ilike.%окончани%расследован%",
        "content.ilike.%расследован%несчаст%",
    ]
    results = _execute_combined_targeted_search(supabase, queries)
    return _deduplicate_chunks(results)


def _targeted_minor_search(supabase, user_query: str = "") -> List[Dict[str, Any]]:
    """
    Прямой lexical search для специальных гарантий несовершеннолетних.
    Нужен как страховка: vector search не должен быть единственным способом
    найти статью 276, если запрос явно содержит возраст работника.
    """
    issue = _minor_special_issue(user_query)
    if not issue:
        return []

    base_queries = [
        "doc_name.ilike.%Трудовой кодекс%",
        "content.ilike.%моложе восемнадцати лет%",
        "content.ilike.%несовершеннолетн%",
    ]

    article_queries = {
        "holiday_weekend": [
            "point_num.ilike.%276%",
            "content.ilike.%статья 276%",
            "content.ilike.%государственные праздники%",
            "content.ilike.%праздничные дни%",
            "content.ilike.%выходные дни%",
        ],
        "night_overtime": [
            "point_num.ilike.%276%",
            "content.ilike.%статья 276%",
            "content.ilike.%ночным%",
            "content.ilike.%сверхурочным%",
        ],
        "prohibited_work": [
            "point_num.ilike.%274%",
            "content.ilike.%статья 274%",
            "content.ilike.%тяжелых работах%",
            "content.ilike.%вредными%",
        ],
        "medical": [
            "point_num.ilike.%275%",
            "content.ilike.%статья 275%",
            "content.ilike.%медицинских осмотров%",
        ],
        "leave": [
            "point_num.ilike.%277%",
            "content.ilike.%статья 277%",
            "content.ilike.%трудовые отпуска%",
        ],
        "work_time": [
            "point_num.ilike.%278%",
            "point_num.ilike.%279%",
            "content.ilike.%сокращенной продолжительности%",
        ],
    }

    results = _execute_combined_targeted_search(
        supabase,
        base_queries + article_queries.get(issue, []),
    )
    return _deduplicate_chunks(results)


def _deduplicate_chunks(
    chunks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    unique: List[Dict[str, Any]] = []
    seen = set()

    for chunk in chunks:
        document = _get_document_name(chunk)
        point = _get_point_number(chunk)
        content = str(chunk.get("content") or "").strip()

        key = (
            document.lower(),
            point.lower(),
            content[:300].lower(),
        )

        if key in seen:
            continue

        seen.add(key)
        unique.append(chunk)

    return unique


async def _get_targeted_chunks(
    supabase,
    topic: str,
    user_query: str = "",
) -> List[Dict[str, Any]]:
    if detect_special_category(user_query) == "minor":
        minor_results = await asyncio.to_thread(
            _targeted_minor_search,
            supabase,
            user_query,
        )
        if minor_results:
            return minor_results

    if topic == "ppe_nonprovision":
        return await asyncio.to_thread(_targeted_ppe_nonprovision_search, supabase)
    if topic == "medical_examinations":
        return await asyncio.to_thread(_targeted_medical_exam_search, supabase)
    if topic == "workplace_attestation":
        return await asyncio.to_thread(
            _targeted_attestation_search,
            supabase,
            user_query,
        )
    if topic == "occupational_briefing":
        return await asyncio.to_thread(_targeted_occupational_briefing_search, supabase, user_query)
    if topic == "accident_investigation":
        return await asyncio.to_thread(
            _targeted_accident_search,
            supabase,
            user_query,
        )
    if topic == "occupational_training":
        return await asyncio.to_thread(_targeted_occupational_training_search, supabase)
    return []


# ============================================================
# SEMANTIC SCORE
# ============================================================

def _semantic_score(chunk: Dict[str, Any]) -> float:
    if "similarity" in chunk:
        return _safe_float(chunk["similarity"])
    if "score" in chunk:
        return _safe_float(chunk["score"])
    if "distance" in chunk:
        return 1.0 - _safe_float(chunk["distance"])
    return 0.0


# ============================================================
# DIVERSIFICATION
# ============================================================

def _get_document_key(chunk: Dict[str, Any]) -> str:
    return _get_document_name(chunk).strip().lower()


# ============================================================
# ПРОВЕРКА НУЖНОГО НПА
# ============================================================

def _is_occupational_training_document(chunk: Dict[str, Any]) -> bool:
    document_name = _get_document_name(chunk).lower()
    content = str(chunk.get("content") or "").lower()

    is_npa_175 = bool(
        re.search(r"№\s*175\b", document_name, flags=re.IGNORECASE)
        or ("инструкци" in document_name and re.search(r"\b175\b", document_name, flags=re.IGNORECASE))
    )

    has_internship = "стажиров" in content
    has_duration = "продолжительн" in content or "срок" in content
    has_minimum_two = "не менее двух" in content
    has_working_period = "рабочих дней" in content or "рабочих смен" in content

    return is_npa_175 or (has_internship and (has_minimum_two or has_working_period or has_duration))


def _is_accident_investigation_document(chunk: Dict[str, Any]) -> bool:
    document_name = _get_document_name(chunk).lower()
    content = str(chunk.get("content") or "").lower()

    has_n1 = bool(re.search(r"\bн[-–—]?\s*1\b", document_name + " " + content, re.IGNORECASE))
    has_accident = "несчаст" in document_name or "несчаст" in content
    has_investigation = "расследован" in document_name or "расследован" in content
    has_victim = "пострадавш" in content or "потерпевш" in content or "родственник" in content

    return has_n1 or (has_accident and has_investigation) or (has_accident and has_victim)


def _is_attestation_document(chunk: Dict[str, Any]) -> bool:
    document_name = _get_document_name(chunk).lower()
    content = str(chunk.get("content") or "").lower()
    return "253" in document_name and ("аттестаци" in document_name or "аттестаци" in content)


def _is_medical_exam_document(chunk: Dict[str, Any]) -> bool:
    document_name = _get_document_name(chunk).lower()
    content = str(chunk.get("content") or "").lower()

    is_npa_74 = bool(
        re.search(r"№\s*74\b", document_name, flags=re.IGNORECASE)
        or re.search(r"\b74\b", document_name, flags=re.IGNORECASE)
    )
    is_medical_document = "медицинск" in document_name and "осмотр" in document_name
    return is_npa_74 or is_medical_document


def _is_ppe_nonprovision_document(chunk: Dict[str, Any]) -> bool:
    document_name = _get_document_name(chunk).lower()
    content = str(chunk.get("content") or "").lower()

    is_npa_209 = bool(
        re.search(r"№\s*209\b", document_name, flags=re.IGNORECASE)
        or re.search(r"\b209\b", document_name, flags=re.IGNORECASE)
    )
    is_ppe_document = "сиз" in document_name or (
        "средств" in document_name and "индивидуальн" in document_name and "защит" in document_name
    )
    has_problem_context = any(
        marker in content
        for marker in (
            "не выдан",
            "невыдач",
            "поврежден",
            "поврежд",
            "неисправн",
            "не обеспечен",
            "отказ",
            "не приступ",
            "приостанов",
        )
    )
    return is_npa_209 or (is_ppe_document and has_problem_context)


# ============================================================
# МНОГОЗАПРОСНЫЙ ЮРИДИЧЕСКИЙ ПОИСК
# ============================================================

def detect_query_intents(user_query: str) -> List[str]:
    query = str(user_query or "").strip().lower()

    patterns = {
        "refusal": [
            r"\bотказ\w*",
            r"\bне\s+приступ\w*",
            r"\bприостанов\w*",
            r"\bне\s+выполнять\b",
        ],
        "danger": [
            r"\bопасн\w*",
            r"\bугроз\w*",
            r"\bриск\w*",
            r"\bаварийн\w*",
        ],
        "employee_right": [
            r"\bимеет\s+ли\s+прав\w*",
            r"\bправ\w*.*\bработник\w*",
            r"\bвправ\w*",
            r"\bможет\s+ли\s+работник\w*",
        ],
        "responsible_person": [
            r"\bкто\s+провод\w*",
            r"\bкто\s+долж\w*\s+провод\w*",
            r"\bкто\s+ответствен\w*",
            r"\bкто\s+назнач\w*",
            r"\bкакое\s+лицо\s+провод\w*",
            r"\bкакой\s+специалист\w*\s+провод\w*",
            r"\bлиц\w*.*\bпровод\w*.*\bинструктаж\w*",
        ],
        "employer_duty": [
            r"\bобязанност\w*\s+нанимател\w*",
            r"\bобязанност\w*.*\bнанимател\w*",
            r"\bобязан\w*.*\bнанимател\w*",
            r"\bнанимател\w*.*\bобязан\w*",
            r"\bнанимател\w*.*\bобеспеч\w*",
            r"\bобеспеч\w*.*\bнанимател\w*",
            r"\bобязанност\w*\s+работодател\w*",
            r"\bобязан\w*.*\bработодател\w*",
            r"\bобеспеч\w*.*\bработник\w*",
        ],
        "procedure": [
            r"\bперв\w*\s+шаг\w*",
            r"\bчто\s+делать\b",
            r"\bпорядок\w*",
            r"\bдейств\w*.*\bработник\w*",
            r"\bсначала\b",
        ],
        "liability": [
            r"\bответственност\w*",
            r"\bштраф\w*",
            r"\bнаказан\w*",
            r"\bвзыскан\w*",
        ],
    }

    return [
        intent
        for intent, pats in patterns.items()
        if any(re.search(p, query, flags=re.IGNORECASE) for p in pats)
    ]


def detect_primary_intent(
    intents: List[str],
    user_query: str = "",
) -> Optional[str]:
    """
    Определяет главный intent не по фиксированному приоритету,
    а по форме реального вопроса. Это важно для составных запросов:
    «кто обязан...», «имеет ли право...», «что должен сделать...».
    """
    if not intents:
        return None

    query = re.sub(r"\s+", " ", str(user_query or "").strip().lower())

    explicit_patterns = [
        (
            "responsible_person",
            (
                r"\bкто\s+(?:провод\w*|должен\s+провод\w*|имеет\s+право\s+провод\w*)",
                r"\bкем\s+провод\w*",
                r"\bкакое\s+лицо\s+провод\w*",
                r"\bкто\s+ответствен\w*",
            ),
        ),
        (
            "employer_duty",
            (
                r"\bкто\s+обязан\b",
                r"\bобязан\s+ли\s+(?:наниматель|работодатель)",
                r"\bчто\s+обязан\s+(?:сделать|обеспечить|организовать)",
                r"\bобязанност\w*\s+(?:нанимателя|работодателя)",
            ),
        ),
        (
            "employee_right",
            (
                r"\bимеет\s+ли\s+прав\w*",
                r"\bимеет\s+ли\s+работник\s+прав\w*",
                r"\bвправе\s+ли\b",
                r"\bможет\s+ли\s+работник\b",
            ),
        ),
        (
            "refusal",
            (
                r"\bимеет\s+ли\s+прав\w*.*\bотказ\w*",
                r"\bможет\s+ли\s+отказ\w*",
                r"\bправ\w*.*\bотказ\w*.*\bработ\w*",
            ),
        ),
        (
            "procedure",
            (
                r"\bчто\s+делать\b",
                r"\bчто\s+должен\s+сделать\b",
                r"\bкак\s+(?:должен|следует|правильно)\s+действ\w*",
                r"\bпорядок\s+(?:действий|проведения|оформления)",
                r"\bкаков\s+порядок\b",
            ),
        ),
        (
            "danger",
            (
                r"\bугроз\w*\s+(?:жизни|здоров\w*)",
                r"\bопасн\w*\s+для\s+(?:жизни|здоров\w*)",
            ),
        ),
        (
            "liability",
            (
                r"\bкакая\s+ответственност\w*",
                r"\bкто\s+нес[её]т\s+ответственност\w*",
                r"\bчто\s+грозит\b",
            ),
        ),
    ]

    matched = []
    for intent, patterns in explicit_patterns:
        if intent in intents and any(re.search(p, query, re.IGNORECASE) for p in patterns):
            matched.append(intent)

    if matched:
        # Для вопросов «имеет ли право отказаться» юридически важен
        # именно refusal, если он уже обнаружен классификатором.
        if "refusal" in matched:
            return "refusal"
        return matched[0]

    # Fallback: только если форма вопроса не дала явного сигнала.
    fallback_priority = [
        "responsible_person",
        "employer_duty",
        "employee_right",
        "refusal",
        "procedure",
        "danger",
        "liability",
    ]
    for intent in fallback_priority:
        if intent in intents:
            return intent

    return None


def _is_labor_code_query(user_query: str) -> bool:
    query = str(user_query or "").strip().lower()
    return bool(
        re.search(r"\bтрудов\w*\s+кодекс\w*", query, flags=re.IGNORECASE)
        or re.search(r"\bтк\s*рб\b", query, flags=re.IGNORECASE)
    )


def is_cross_reference_query(intents: List[str]) -> bool:
    return len(set(intents) & {
        "refusal", "danger", "employee_right",
        "employer_duty", "procedure", "liability",
    }) >= 2


def _is_target_briefing_query(user_query: str) -> bool:
    query = re.sub(r"\s+", " ", str(user_query or "").strip().lower())
    if not query:
        return False

    target_patterns = [
        r"\bкакой\s+(?:вид\s+)?инструктаж\w*\b",
        r"\bкакой\s+(?:вид\s+)?инструктаж\w*\s+(?:нужен|необходим|провод\w*|требу\w*)",
        r"\bкакому\s+инструктаж\w*\b",
        r"\bк\s+какому\s+инструктаж\w*\b",
        r"\bвид\s+инструктаж\w*\b.*\bнужен\b",
        r"\bотнос\w*\s+к\s+(?:какому|какой)\s+инструктаж\w*",
    ]

    if any(re.search(pattern, query, flags=re.IGNORECASE) for pattern in target_patterns):
        return True

    target_context_patterns = [
        r"\bразов\w*\s+работ\w*.*\bне\s+связан\w*.*\bпрям\w*\s+обязанност\w*",
        r"\bне\s+связан\w*\s+с\s+прям\w*\s+обязанност\w*",
        r"\bпрям\w*\s+обязанност\w*.*\bразов\w*\s+работ\w*",
        r"\bнаряд\w*[-–—]?\s*допуск\w*",
        r"\bнаряд\w*\s+допуск\w*",
        r"\bцелев\w*\s+инструктаж\w*",
    ]

    return any(re.search(pattern, query, flags=re.IGNORECASE) for pattern in target_context_patterns)


def _is_responsible_briefing_query(user_query: str) -> bool:
    query = re.sub(r"\s+", " ", str(user_query or "").strip().lower())
    responsible_patterns = [
        r"\bкто\s+провод\w*.*\bинструктаж\w*",
        r"\bкем\s+провод\w*.*\bинструктаж\w*",
        r"\bкто\s+должен\s+провод\w*.*\bинструктаж\w*",
        r"\bкто\s+имеет\s+право\s+провод\w*.*\bинструктаж\w*",
        r"\bлиц\w*\s+(?:котор\w*|кто)\s+провод\w*.*\bинструктаж\w*",
        r"\bкто\s+ответствен\w*.*\bинструктаж\w*",
        r"\bответствен\w*\s+за\s+проведен\w*\s+инструктаж\w*",
        r"\bпровод\w*\s+инструктаж\w*.*\bкто\b",
    ]
    return any(re.search(pattern, query, flags=re.IGNORECASE) for pattern in responsible_patterns)



# ============================================================
# УНИВЕРСАЛЬНЫЙ ПРОФИЛЬ ЗАПРОСА
# ============================================================

def build_universal_query_profile(user_query: str) -> Dict[str, Any]:
    """Детерминированно извлекает юридическую структуру вопроса."""
    query = re.sub(r"\s+", " ", str(user_query or "").strip().lower())
    profile: Dict[str, Any] = {
        "question_type": "general",
        "subject": None,
        "event": None,
        "action": None,
        "action_state": None,
        "actor": None,
        "recipient": None,
        "object": None,
        "qualifiers": [],
        "legal_phrases": [],
        "constraint": {
            "type": None,
            "value": None,
            "unit": None,
            "subject": None,
            "action": None,
        },
    }
    if not query:
        return profile

    qtypes = [
        ("what_to_do", [r"\bчто\s+делать\b", r"\bкак\s+(?:должен|следует)\s+действ", r"\bпорядок\s+действ"]),
        ("frequency", [r"\bкак\s+часто\b", r"\bс\s+какой\s+периодичност", r"\bпериодичност\w*\b"]),
        ("limit", [r"\bсколько\b.*\bкг\b", r"\bсколько\s+разрешено\b", r"\bпредельн\w*\s+норм", r"\bнорм\w*\s+(?:подъема|перемещения)"]),
        ("who", [r"^кто\b", r"\bкто\s+(?:провод|должен|обязан|назнач|ответствен)", r"\bкем\b", r"\bкакое\s+лицо\b"]),
        ("kind", [r"\bкакой\s+(?:вид\s+)?инструктаж", r"\bкакому\s+инструктаж", r"\bвид\w*\s+инструктаж"]),
        ("whether", [r"\bможно\s+ли\b", r"\bразрешено\s+ли\b", r"\bдопускается\s+ли\b", r"\bимеет\s+ли\s+прав"]),
        ("responsibility", [r"\bкто\s+нес[её]т\s+ответствен", r"\bкто\s+ответствен", r"\bкакая\s+ответствен"]),
        ("term", [r"\bкакой\s+срок\b", r"\bсрок\w*\b", r"\bв\s+течение\b"]),
        ("document", [r"\bкаким\s+документ", r"\bкакой\s+(?:нпа|документ|акт)\b"]),
    ]
    for qtype, patterns in qtypes:
        if any(re.search(p, query, re.IGNORECASE) for p in patterns):
            profile["question_type"] = qtype
            break

    if re.search(r"\bне\s+сообщил\w*\b|\bне\s+сообщила\w*\b|\bне\s+выдан\w*\b|\bне\s+обеспечен\w*\b|\bне\s+прошел\w*\b|\bне\s+приступ\w*\b", query, re.IGNORECASE):
        profile["action_state"] = "not_done"

    if re.search(r"\bнесчастн\w*\s+случа\w*|\bтравм\w*\s+на\s+производств", query, re.IGNORECASE):
        profile.update({"subject": "employee", "event": "work_accident", "action": "report", "object": "work_accident"})
        profile["legal_phrases"].extend([
            "несчастный случай на производстве",
            "сообщить о несчастном случае",
            "немедленно сообщить о несчастном случае",
            "непосредственному руководителю",
            "порядок действий работодателя при несчастном случае",
        ])
        if re.search(r"\bруководител\w*|\bначальник\w*", query, re.IGNORECASE):
            profile["recipient"] = "immediate_supervisor"

    if re.search(r"\bподнима\w*|\bперемещ\w*\s+тяжест\w*|\bтяжест\w*\s+вручн\w*|\bсколько\s+кг\b", query, re.IGNORECASE):
        profile.update({"subject": "manual_handling", "event": "lifting_and_moving_loads", "action": "lift_move", "object": "load"})
        profile["legal_phrases"].extend([
            "предельно допустимые нормы подъема и перемещения тяжестей вручную",
            "подъем и перемещение тяжестей вручную",
            "предельно допустимая масса тяжести",
            "нормы подъема тяжестей вручную",
        ])
        is_male = bool(re.search(r"\bмужчин\w*", query, re.IGNORECASE))
        is_female = bool(re.search(r"\bженщин\w*", query, re.IGNORECASE))
        if is_male:
            profile["qualifiers"].append("men")
            profile["constraint"].update({"type": "maximum", "unit": "kg", "subject": "adult_male", "action": "lifting"})
        elif is_female:
            profile["qualifiers"].append("women")
            profile["constraint"].update({"type": "maximum", "unit": "kg", "subject": "adult_female", "action": "lifting"})
        else:
            profile["constraint"].update({"type": "maximum", "unit": "kg", "action": "lifting"})

    if re.search(r"\bаттестаци\w*\s+рабоч\w*\s+мест", query, re.IGNORECASE):
        profile.update({"subject": "workplace", "event": "workplace_attestation", "action": "attest", "object": "working_conditions"})
        profile["legal_phrases"].extend([
            "аттестация рабочих мест по условиям труда",
            "срок действия результатов аттестации",
            "результаты аттестации рабочих мест",
        ])

    if re.search(r"\bсиз\b|\bсредств\w*\s+индивидуальн\w*\s+защит", query, re.IGNORECASE):
        profile.update({"subject": "employee", "event": "ppe", "object": "personal_protective_equipment"})
        profile["legal_phrases"].extend([
            "средства индивидуальной защиты",
            "обеспечение средствами индивидуальной защиты",
            "невыдача средств индивидуальной защиты",
        ])

    if re.search(r"\bмедицинск\w*\s+осмотр\w*|\bмедосмотр\w*", query, re.IGNORECASE):
        profile.update({"subject": "employee", "event": "medical_exam", "action": "medical_examination", "object": "medical_exam"})
        profile["legal_phrases"].extend([
            "обязательный медицинский осмотр",
            "предварительный медицинский осмотр",
            "периодический медицинский осмотр",
        ])

    if re.search(r"\bинструктаж\w*", query, re.IGNORECASE):
        profile["event"] = "occupational_briefing"
        profile["object"] = "occupational_briefing"
        if profile["question_type"] == "kind":
            profile["legal_phrases"].extend([
                "виды инструктажей по охране труда",
                "вводный инструктаж",
                "первичный инструктаж",
                "повторный инструктаж",
                "внеплановый инструктаж",
                "целевой инструктаж",
            ])

    if re.search(r"\bстажиров\w*", query, re.IGNORECASE):
        profile["event"] = "occupational_training"
        profile["object"] = "internship"
        profile["legal_phrases"].extend([
            "стажировка по охране труда",
            "допуск к самостоятельной работе",
            "продолжительность стажировки",
        ])

    profile["legal_phrases"] = list(dict.fromkeys(profile["legal_phrases"]))
    profile["qualifiers"] = list(dict.fromkeys(profile["qualifiers"]))
    return profile


def build_universal_search_queries(profile: Dict[str, Any], original: str) -> List[str]:
    """Генерирует нормативные формулировки независимо от конкретной темы."""
    queries = list(profile.get("legal_phrases") or [])
    event = profile.get("event")
    qtype = profile.get("question_type")
    state = profile.get("action_state")

    if event == "lifting_and_moving_loads":
        constraint = profile.get("constraint") or {}
        if constraint.get("subject") == "adult_male":
            queries.extend([
                "предельно допустимая норма разового подъема тяжестей вручную работающим мужчиной 50 кг",
                "пункт 86 постановления 12 26.01.2018 погрузочно-разгрузочные работы 50 кг мужчина",
                "ручные погрузочно-разгрузочные работы разовый подъем тяжестей мужчиной не более 50 кг",
            ])
        elif constraint.get("subject") == "adult_female":
            queries.extend([
                "предельные нормы подъема и перемещения тяжестей вручную женщинами",
                "нормы подъема тяжестей вручную женщины Республика Беларусь",
            ])
        else:
            queries.extend([
                "предельно допустимые нормы подъема и перемещения тяжестей вручную",
                "предельные нормы подъема и перемещения тяжестей вручную кг",
                "нормы подъема тяжестей вручную Республика Беларусь",
            ])
    elif event == "work_accident":
        queries.extend([
            "обязанность немедленно сообщить о несчастном случае непосредственному руководителю",
            "работник не сообщил о несчастном случае порядок действий",
            "порядок действий работодателя при получении сообщения о несчастном случае",
        ])
    elif event == "workplace_attestation" and qtype == "frequency":
        queries.extend([
            "срок действия результатов аттестации составляет пять лет",
            "пункт 19 аттестация рабочих мест срок действия результатов",
        ])
    elif event == "occupational_briefing" and qtype == "kind":
        queries.extend([
            "виды инструктажей по охране труда",
            "какой инструктаж проводится при разовых работах не связанных с прямыми обязанностями",
        ])

    if qtype == "what_to_do":
        queries.append(f"порядок действий {event or ''} {original}".strip())
    elif qtype == "frequency":
        queries.append(f"периодичность {event or ''} {original}".strip())
    elif qtype == "limit":
        queries.append(f"предельно допустимая норма {event or ''} {original}".strip())
    elif qtype == "who":
        queries.append(f"кто обязан кто проводит кто отвечает {event or ''} {original}".strip())
    elif qtype == "whether":
        queries.append(f"разрешено ли допускается имеет право {event or ''} {original}".strip())

    if state == "not_done":
        queries.append(f"нарушение обязанности не выполнено {event or ''} что делать {original}".strip())

    return list(dict.fromkeys(q for q in queries if q))[:6]


def build_search_queries(
    user_query: str,
    topic: str,
    legal_domain: str,
    intents: List[str],
) -> List[str]:
    original = str(user_query or "").strip()
    universal_profile = build_universal_query_profile(original)
    queries: List[str] = [original]
    queries.extend(build_universal_search_queries(universal_profile, original))

    target_briefing_query = (
        topic == "occupational_briefing"
        and _is_target_briefing_query(original)
    )

    responsible_briefing_query = (
        topic == "occupational_briefing"
        and not target_briefing_query
        and (
            _is_responsible_briefing_query(original)
            or "responsible_person" in intents
        )
    )

    special_category = detect_special_category(original)
    minor_issue = _minor_special_issue(original) if special_category == "minor" else None

    if special_category == "minor":
        minor_queries = [
            "несовершеннолетние работники глава 20 Трудового кодекса Республики Беларусь",
            "работники моложе восемнадцати лет Трудовой кодекс Республики Беларусь",
            "статья 273 Трудовой кодекс Республики Беларусь несовершеннолетние",
            "статья 274 Трудовой кодекс Республики Беларусь лица моложе восемнадцати лет",
            "статья 275 Трудовой кодекс Республики Беларусь медицинские осмотры лиц моложе восемнадцати лет",
            "статья 276 Трудовой кодекс Республики Беларусь работники моложе восемнадцати лет",
        ]

        if minor_issue == "holiday_weekend":
            minor_queries[0:0] = [
                "статья 276 Трудовой кодекс Республики Беларусь работники моложе восемнадцати лет государственные праздники праздничные выходные дни",
                "несовершеннолетний работник праздничный день можно ли привлекать статья 276",
                "работник 17 лет праздничный нерабочий день статья 276",
            ]
        elif minor_issue == "night_overtime":
            minor_queries[0:0] = [
                "статья 276 Трудовой кодекс Республики Беларусь работники моложе восемнадцати лет ночные сверхурочные работы",
            ]
        elif minor_issue == "prohibited_work":
            minor_queries[0:0] = [
                "статья 274 Трудовой кодекс Республики Беларусь лица моложе восемнадцати лет тяжелые вредные опасные работы",
            ]
        elif minor_issue == "medical":
            minor_queries[0:0] = [
                "статья 275 Трудовой кодекс Республики Беларусь медицинские осмотры лиц моложе восемнадцати лет",
            ]
        elif minor_issue == "leave":
            minor_queries[0:0] = [
                "статья 277 Трудовой кодекс Республики Беларусь трудовые отпуска работникам моложе восемнадцати лет",
            ]

        queries.extend(minor_queries)

    topic_queries = {
        "ppe_nonprovision": [
            f"СИЗ не выданы повреждены неисправны работник безопасность труда {original}",
            "право работника отказаться от выполнения работы при угрозе жизни и здоровью",
            "действия работника при возникновении опасности для жизни и здоровья",
            "обязанности нанимателя по обеспечению безопасных условий труда и средствами индивидуальной защиты",
            "неприступление к работе или отказ от выполнения опасной работы трудовое законодательство",
        ],
        "occupational_briefing": [],
        "accident_investigation": [
            f"несчастный случай расследование {original}",
            "порядок расследования несчастного случая на производстве",
            "сообщение о несчастном случае работником непосредственному руководителю",
            "обязанность работника сообщить о несчастном случае нанимателю",
        ],
        "workplace_attestation": [
            f"аттестация рабочих мест условия труда {original}",
            "основания проведения аттестации рабочих мест по условиям труда",
            "результаты аттестации рабочие места вредные условия труда дополнительные отпуска",
        ],
        "medical_examinations": [
            f"обязательные медицинские осмотры работников {original}",
            "кто обязан организовать и оплачивать медицинские осмотры работников",
            "предварительные периодические медицинские осмотры порядок направления работников",
        ],
        "occupational_training": [
            f"стажировка допуск к самостоятельной работе {original}",
            "минимальная продолжительность стажировки по охране труда",
            "стажировка рабочие дни рабочие смены повышенная опасность проверка знаний",
        ],
    }

    if topic == "accident_investigation":
        accident_mode = _accident_query_mode(original)

        constraint = (query_profile or {}).get("constraint") or {}

    if constraint.get("type") == "maximum" and constraint.get("unit") == "kg":
        scored_constraint = []
        for chunk in ranked_chunks:
            cscore = _universal_query_relevance_score(chunk, query_profile or {})
            text_lower = (
                f"{_get_document_name(chunk)} "
                f"{_get_point_number(chunk)} "
                f"{str(chunk.get('content') or '')}"
            ).lower()

            if constraint.get("subject") == "adult_male":
                if "50 кг" in text_lower or "50 килограмм" in text_lower:
                    cscore += 0.50
                if "погрузочно-разгрузоч" in text_lower:
                    cscore += 0.25
                if "пункт 86" in text_lower or "п. 86" in text_lower:
                    cscore += 0.25
                if any(x in text_lower for x in (
                    "женщин", "женщина", "несовершеннолетн",
                    "моложе восемнадцати лет",
                )) and not any(x in text_lower for x in (
                    "мужчин", "мужчина", "работающим мужчиной",
                )):
                    cscore -= 0.45

            scored_constraint.append((
                cscore,
                _safe_float(chunk.get("_combined_score")),
                chunk,
            ))

        scored_constraint.sort(
            key=lambda item: (item[0], item[1]),
            reverse=True,
        )

        for cscore, _, chunk in scored_constraint:
            if cscore < 0.35:
                continue
            if _add(chunk, max_per_document=4, max_per_point=1):
                if len(selected) >= limit:
                    return selected

        if selected:
            return selected

    if accident_mode == "worker_did_not_report":
            queries.extend([
                f"работник не сообщил о несчастном случае руководителю {original}",
                "если работник не сообщил о несчастном случае непосредственному руководителю",
                "потерпевший не сообщил о несчастном случае что делать",
                "обязанность работника немедленно сообщить о несчастном случае",
                "сообщить о несчастном случае непосредственному руководителю порядок действий",
            ])

    if topic == "workplace_attestation":
        attestation_mode = _attestation_query_mode(original)

        if attestation_mode == "periodicity":
            queries.extend([
                f"периодичность аттестации рабочих мест по условиям труда {original}",
                "пункт 19 Положения о порядке проведения аттестации рабочих мест по условиям труда",
                "срок действия результатов аттестации составляет пять лет",
                "аттестация рабочих мест проводится один раз в пять лет",
            ])
        elif attestation_mode == "extraordinary":
            queries.extend([
                f"внеочередная аттестация рабочих мест по условиям труда {original}",
                "пункт 17 Положения о порядке проведения аттестации рабочих мест по условиям труда",
                "внеочередная аттестация переаттестация в течение шести месяцев",
            ])
        elif attestation_mode == "results":
            queries.extend([
                f"результаты аттестации рабочих мест по условиям труда {original}",
                "пункт 12 Положения о порядке проведения аттестации рабочих мест по условиям труда",
            ])
        elif attestation_mode == "commission":
            queries.extend([
                f"комиссия по аттестации рабочих мест по условиям труда {original}",
                "состав комиссии по проведению аттестации рабочих мест",
            ])

    if topic == "occupational_briefing":
        if target_briefing_query:
            queries.extend([
                f"целевой инструктаж разовые работы не связанные с прямыми обязанностями {original}",
                "целевой инструктаж разовые работы не связанные с прямыми обязанностями",
                "разовые работы не связанные с прямыми обязанностями какой инструктаж",
                "целевой инструктаж наряд-допуск работы с повышенной опасностью",
                "Инструкция №175 пункт 29 целевой инструктаж разовые работы",
            ])
        elif responsible_briefing_query:
            queries.extend([
                f"вводный инструктаж по охране труда кто проводит {original}",
                "кто проводит вводный инструктаж по охране труда специалист по охране труда уполномоченное должностное лицо нанимателя",
                "вводный инструктаж проводит специалист по охране труда уполномоченное должностное лицо нанимателя",
                "вводный инструктаж руководитель организации специалист по охране труда",
            ])
        else:
            queries.extend([
                f"инструктаж по охране труда {original}",
                "виды инструктажей по охране труда порядок проведения",
                "Инструкция №175 инструктаж по охране труда",
            ])
    else:
        queries.extend(topic_queries.get(topic, []))

    if (
        "responsible_person" in intents
        and not target_briefing_query
        and responsible_briefing_query
    ):
        pass
    elif "responsible_person" in intents and topic != "occupational_briefing":
        queries.extend([
            "вводный инструктаж по охране труда кто проводит",
            "вводный инструктаж проводит специалист по охране труда",
            "вводный инструктаж проводит уполномоченное должностное лицо нанимателя",
            "вводный инструктаж руководитель организации специалист по охране труда",
        ])

    if "employer_duty" in intents:
        queries.extend([
            "Трудовой кодекс Республики Беларусь обязанности нанимателя охрана труда",
            "Трудовой кодекс Республики Беларусь обязанности нанимателя безопасные условия труда",
            "обязанности нанимателя по охране труда Трудовой кодекс Республики Беларусь",
            "наниматель обязан обеспечить безопасные условия труда Трудовой кодекс Республики Беларусь",
        ])

    if "employee_right" in intents:
        queries.append(f"право работника охрана труда безопасные условия труда {original}")

    if "danger" in intents:
        queries.append(f"угроза жизни и здоровью работника опасность труд {original}")

    if "procedure" in intents:
        queries.append(f"порядок действий работника при нарушении требований охраны труда {original}")

    if _is_labor_code_query(original):
        queries.extend([
            "Трудовой кодекс Республики Беларусь охрана труда работник наниматель",
            "Трудовой кодекс Республики Беларусь безопасные условия труда",
        ])

    result: List[str] = []
    seen = set()

    accident_worker_not_report_query = (
        topic == "accident_investigation"
        and _accident_query_mode(original) == "worker_did_not_report"
    )

    target_forbidden = (
        "кто проводит",
        "кто должен проводить",
        "кто имеет право проводить",
        "какое лицо проводит",
        "какой специалист проводит",
        "специалист по охране труда",
    )
    responsible_forbidden = (
        "какой инструктаж",
        "какому инструктаж",
        "целевой инструктаж разовые работы",
        "разовые работы не связанные с прямыми обязанностями",
        "наряд-допуск",
    )

    for q in queries:
        q_clean = re.sub(r"\s+", " ", str(q or "")).strip()
        qn = q_clean.lower()

        if not qn:
            continue

        if target_briefing_query and any(marker in qn for marker in target_forbidden):
            continue

        if responsible_briefing_query and any(marker in qn for marker in responsible_forbidden):
            continue

        if accident_worker_not_report_query and any(
            marker in qn
            for marker in (
                "групповой несчастный случай",
                "прокуратур",
                "государственной инспекции труда",
                "акт н-1",
                "вручение потерпевшему",
                "вручение родственникам",
                "после окончания расследования",
            )
        ):
            continue

        if qn not in seen:
            seen.add(qn)
            result.append(q_clean)

    final_queries = result[:8]

    if target_briefing_query:
        final_queries = [
            q for q in final_queries
            if not any(marker in q.lower() for marker in target_forbidden)
        ]
        if not final_queries:
            final_queries = [
                original,
                "целевой инструктаж разовые работы не связанные с прямыми обязанностями",
                "Инструкция №175 пункт 29 целевой инструктаж",
            ]

    return final_queries[:8]


def _merge_search_results(
    groups: List[List[Dict[str, Any]]],
    query_roles: List[str],
) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}

    for group_index, group in enumerate(groups):
        role = query_roles[group_index] if group_index < len(query_roles) else "semantic"

        for chunk in group:
            document = _get_document_name(chunk)
            point = _get_point_number(chunk)
            content = str(chunk.get("content") or "").strip()
            key = (
                document.lower(),
                point.lower(),
                content[:300].lower(),
            )
            score = _semantic_score(chunk)

            if key not in merged:
                item = dict(chunk)
                item["_best_similarity"] = score
                item["_search_hits"] = 1
                item["_query_roles"] = [role]
                merged[key] = item
                continue

            item = merged[key]
            if score > _safe_float(item.get("_best_similarity")):
                item["_best_similarity"] = score
                for field in ("similarity", "score", "distance"):
                    if field in chunk:
                        item[field] = chunk[field]
                        break

            item["_search_hits"] = int(item.get("_search_hits", 0)) + 1

            if chunk.get("_accident_worker_not_report_targeted"):
                item["_accident_worker_not_report_targeted"] = True
            roles = item.setdefault("_query_roles", [])
            if role not in roles:
                roles.append(role)

    return list(merged.values())


def _intent_relevance_score(
    chunk: Dict[str, Any],
    intents: List[str],
) -> float:
    if not intents:
        return 0.0

    text = " ".join([
        str(chunk.get("doc_name") or chunk.get("document") or ""),
        str(chunk.get("content") or chunk.get("text") or ""),
    ]).lower()

    markers = {
        "refusal": ["отказ", "не приступ", "приостанов", "не выполнять"],
        "danger": ["угроз", "опасност", "жизни", "здоров", "риск"],
        "employee_right": ["имеет право", "право работника", "вправе"],
        "employer_duty": ["обязан", "обязанность нанимателя", "наниматель обязан", "работодатель обязан"],
        "procedure": ["порядок", "действия работника", "немедленно сообщ", "уведом"],
        "responsible_person": [
            "проводит вводный инструктаж",
            "проводит инструктаж",
            "специалист по охране труда",
            "уполномоченное должностное лицо",
            "руководитель организации",
        ],
        "liability": ["ответственност", "штраф", "взыскан", "наказан"],
    }

    score = 0.0
    for intent in intents:
        matches = sum(1 for marker in markers.get(intent, []) if marker in text)
        score += min(matches * 0.18, 0.35)

    return min(score, 1.0)


def _primary_intent_relevance_score(
    chunk: Dict[str, Any],
    primary_intent: Optional[str],
    topic: str,
) -> float:
    if not primary_intent:
        return 0.0

    role = _legal_chunk_role(
        chunk,
        topic,
        [primary_intent],
        primary_intent=primary_intent,
    )

    if role == primary_intent:
        return 1.0

    return 0.0


def _exact_match_score(
    chunk: Dict[str, Any],
    user_query: str,
    topic: str,
) -> float:
    """Strong lexical/exact-match signal for legal RAG."""
    document_name = str(chunk.get("doc_name") or chunk.get("document") or "").lower()
    content = str(chunk.get("content") or chunk.get("text") or "").lower()
    text = re.sub(r"\s+", " ", f"{document_name} {content}").strip()
    query = re.sub(r"\s+", " ", str(user_query or "").strip().lower())

    if not text or not query:
        return 0.0

    score = 0.90 if len(query) >= 18 and query in text else 0.0

    phrase_weights = {
        "ppe_nonprovision": (
            ("средства индивидуальной защиты", 0.40),
            ("не выданы", 0.40),
            ("не обеспечен средствами индивидуальной защиты", 0.40),
            ("не приступать к работе", 0.35),
            ("приостановить работу", 0.35),
            ("отказаться от выполнения работы", 0.30),
        ),
        "medical_examinations": (
            ("медицинский осмотр", 0.45),
            ("обязательный медицинский осмотр", 0.40),
            ("предварительный медицинский осмотр", 0.35),
            ("периодический медицинский осмотр", 0.35),
        ),
        "workplace_attestation": (
            ("аттестация рабочих мест", 0.50),
            ("условия труда", 0.30),
            ("вредные условия труда", 0.35),
            ("полный рабочий день", 0.25),
            ("периодичность аттестации", 0.70),
            ("один раз в пять лет", 0.90),
            ("срок действия результатов аттестации", 0.75),
            ("очередной аттестации", 0.55),
        ),
        "accident_investigation": (
            ("акт формы н-1", 0.45),
            ("форма н-1", 0.45),
            ("несчастный случай на производстве", 0.45),
            ("окончания расследования", 0.30),
            ("пострадавшему", 0.25),
        ),
        "occupational_training": (
            ("стажировка", 0.40),
            ("самостоятельной работе", 0.35),
            ("не менее двух рабочих дней", 0.45),
            ("повышенной опасностью", 0.30),
            ("проверки знаний", 0.25),
        ),
    }

    if topic == "occupational_briefing":
        mode = (
            "target"
            if _is_target_briefing_query(user_query)
            else (
                "responsible"
                if _is_responsible_briefing_query(user_query)
                else "generic"
            )
        )

        target_phrases = (
            ("целевой инструктаж", 0.75),
            ("разовых работ", 0.60),
            ("не связанных с прямыми обязанностями", 0.90),
            ("не связаны с прямыми обязанностями", 0.90),
            ("наряд-допуск", 0.55),
            ("наряду-допуску", 0.65),
            ("наряд допуск", 0.45),
        )
        responsible_phrases = (
            ("вводный инструктаж", 0.70),
            ("проводит инструктаж", 0.60),
            ("специалист по охране труда", 0.60),
            ("уполномоченное должностное лицо", 0.60),
            ("руководитель организации", 0.25),
            ("руководитель структурного подразделения", 0.25),
        )

        if mode == "target":
            for phrase, weight in target_phrases:
                if phrase in text:
                    score += weight
            responsible_matches = sum(
                1 for phrase, _ in responsible_phrases if phrase in text
            )
            score -= min(responsible_matches * 0.12, 0.36)
        elif mode == "responsible":
            for phrase, weight in responsible_phrases:
                if phrase in text:
                    score += weight
            target_matches = sum(
                1 for phrase, _ in target_phrases if phrase in text
            )
            score -= min(target_matches * 0.10, 0.30)
        else:
            for phrase, weight in target_phrases + responsible_phrases:
                if phrase in text:
                    score += weight * 0.55
    else:
        for phrase, weight in phrase_weights.get(topic, ()):
            if phrase in text:
                score += weight

    if topic in ("occupational_briefing", "occupational_training") and "175" in document_name:
        score += 0.08

    return max(0.0, min(score, 1.00))


def _minor_special_relevance_score(
    chunk: Dict[str, Any],
    special_category: Optional[str],
    issue: Optional[str],
) -> float:
    """Точечный score для специальных гарантий несовершеннолетних."""
    if special_category != "minor":
        return 0.0

    document = _get_document_name(chunk).lower()
    point = _get_point_number(chunk).lower()
    content = str(chunk.get("content") or chunk.get("text") or "").lower()
    text = f"{document} {point} {content}"
    score = 0.0

    if issue == "holiday_weekend":
        if re.search(r"(?<!\d)276(?!\d)", point) or "статья 276" in text:
            score += 0.50
        if "работников моложе восемнадцати лет" in text:
            score += 0.28
        if "государственные праздники" in text or "праздничные дни" in text:
            score += 0.22
        if "выходные дни" in text:
            score += 0.12
    elif issue == "night_overtime":
        if re.search(r"(?<!\d)276(?!\d)", point) or "статья 276" in text:
            score += 0.50
        if "работников моложе восемнадцати лет" in text:
            score += 0.25
        if "ночным" in text or "сверхурочным" in text:
            score += 0.20
    elif issue == "prohibited_work":
        if re.search(r"(?<!\d)274(?!\d)", point) or "статья 274" in text:
            score += 0.50
        if "лиц моложе восемнадцати лет" in text:
            score += 0.25
        if "тяжелых работах" in text or "вредными" in text or "опасными условиями" in text:
            score += 0.20
    elif issue == "medical":
        if re.search(r"(?<!\d)275(?!\d)", point) or "статья 275" in text:
            score += 0.50
        if "лиц моложе восемнадцати лет" in text:
            score += 0.25
        if "медицинских осмотров" in text:
            score += 0.20
    elif issue == "leave":
        if re.search(r"(?<!\d)277(?!\d)", point) or "статья 277" in text:
            score += 0.50
        if "работникам моложе восемнадцати лет" in text:
            score += 0.30
        if "трудовые отпуска" in text:
            score += 0.20
    elif issue == "work_time":
        if re.search(r"(?<!\d)278(?!\d)|(?<!\d)279(?!\d)", point):
            score += 0.50
        if "работников моложе восемнадцати лет" in text:
            score += 0.20
        if "сокращенной продолжительности" in text:
            score += 0.20

    if "моложе восемнадцати лет" in text or "несовершеннолетн" in text:
        score += 0.10

    return min(score, 1.00)



def _universal_query_relevance_score(
    chunk: Dict[str, Any],
    profile: Dict[str, Any],
) -> float:
    """Универсальный reranker: насколько chunk отвечает именно вопросу."""
    document = _get_document_name(chunk).lower()
    content = str(chunk.get("content") or chunk.get("text") or "").lower()
    point = _get_point_number(chunk).lower()
    text = f"{document} {point} {content}"
    score = 0.0

    qtype = profile.get("question_type") or "general"
    event = profile.get("event")
    state = profile.get("action_state")
    qualifiers = set(profile.get("qualifiers") or [])
    phrases = profile.get("legal_phrases") or []

    def hits(markers):
        return sum(1 for marker in markers if marker in text)

    score += min(sum(1 for phrase in phrases if phrase.lower() in text) * 0.16, 0.48)

    if event == "work_accident":
        score += min(hits(("несчастный случай", "несчастном случае")) * 0.10, 0.20)
        score += min(hits(("сообщить", "сообщает", "сообщают", "уведомить")) * 0.07, 0.21)
        if profile.get("recipient") == "immediate_supervisor":
            score += min(hits(("непосредственному руководителю", "должностному лицу страхователя", "руководителю")) * 0.10, 0.30)
        if state == "not_done":
            score += min(hits(("не сообщил", "не сообщила", "несвоевременно")) * 0.14, 0.28)

    if event == "lifting_and_moving_loads":
        score += min(hits(("подъем", "подъема", "перемещение", "перемещения", "тяжест")) * 0.07, 0.28)
        score += min(hits(("вручную", "ручн")) * 0.10, 0.20)
        score += min(hits(("предельно допустим", "предельные нормы", "нормы подъема")) * 0.18, 0.36)
        constraint = profile.get("constraint") or {}
        constraint_subject = constraint.get("subject")
        if "men" in qualifiers or constraint_subject == "adult_male":
            score += min(hits(("мужчин", "мужчина", "мужского пола", "работающим мужчиной")) * 0.16, 0.32)
            wrong_subject_hits = hits(("женщин", "женщина", "лиц моложе восемнадцати лет", "несовершеннолетн"))
            score -= min(wrong_subject_hits * 0.18, 0.36)
            if hits(("50 кг", "50 килограмм", "не более 50 кг", "не более 50 килограмм")):
                score += 0.42
            if hits(("погрузочно-разгрузочн", "погрузочно разгрузочн")):
                score += 0.22
            if hits(("разовый подъем", "разового подъема", "разовом подъеме")):
                score += 0.18
            if hits(("пункт 86", "п. 86", "26.01.2018", "постановления 12")):
                score += 0.20
        elif "women" in qualifiers or constraint_subject == "adult_female":
            score += min(hits(("женщин", "женщина", "женского пола")) * 0.16, 0.32)
            wrong_subject_hits = hits(("мужчин", "мужчина", "лиц моложе восемнадцати лет", "несовершеннолетн"))
            score -= min(wrong_subject_hits * 0.18, 0.36)
        if qtype == "limit":
            score += min(hits(("кг", "килограмм", "масса", "вес")) * 0.08, 0.16)

    if event == "workplace_attestation":
        score += min(hits(("аттестаци", "рабочих мест", "условия труда")) * 0.08, 0.24)
        if qtype == "frequency":
            score += min(hits(("периодичност", "срок действия результатов", "пять лет", "очередн")) * 0.15, 0.45)

    if event == "occupational_briefing":
        score += min(hits(("инструктаж", "охране труда")) * 0.08, 0.16)
        if qtype == "kind":
            score += min(hits(("виды инструктажей", "вводный инструктаж", "первичный инструктаж", "повторный инструктаж", "внеплановый инструктаж", "целевой инструктаж")) * 0.13, 0.39)
        elif qtype == "who":
            score += min(hits(("проводит", "специалист по охране труда", "уполномоченное должностное лицо", "руководитель")) * 0.12, 0.36)

    if event == "ppe":
        score += min(hits(("средств индивидуальной защиты", "сиз", "обеспеч", "выдач")) * 0.08, 0.24)
        if state == "not_done":
            score += min(hits(("не выдан", "не обеспечен", "неисправн", "поврежден")) * 0.14, 0.28)

    if event == "medical_exam":
        score += min(hits(("медицинск", "осмотр", "обязательн", "предварительн", "периодическ")) * 0.08, 0.32)

    if event == "occupational_training":
        score += min(hits(("стажиров", "самостоятельной работе", "рабочих дней", "рабочих смен")) * 0.10, 0.35)

    focus_markers = {
        "what_to_do": ("порядок", "действия", "обязан", "должен", "немедленно", "следует"),
        "who": ("проводит", "обязан", "должностному лицу", "ответствен", "назнач"),
        "frequency": ("периодичност", "раз в", "срок действия", "пять лет"),
        "limit": ("предельн", "норм", "максимальн", "кг", "килограмм", "допустим"),
        "whether": ("вправе", "имеет право", "может", "допускается", "разрешается"),
        "responsibility": ("ответствен", "взыскан", "штраф", "нарушен"),
        "term": ("срок", "в течение", "не позднее", "дней", "месяц", "лет"),
        "document": ("положение", "правила", "инструкция", "кодекс", "постановлен", "приказ"),
        "kind": ("виды инструктаж", "вводный", "первичный", "повторный", "внеплановый", "целевой"),
    }
    if qtype in focus_markers:
        score += min(hits(focus_markers[qtype]) * 0.08, 0.24)

    if state == "not_done":
        negative_hits = hits(("не сообщил", "не выдан", "не обеспечен", "не прошел", "не проведен", "не допущен", "не приступ"))
        if negative_hits:
            score += min(negative_hits * 0.10, 0.20)

    return max(0.0, min(score, 1.0))


def _legal_relevance_score(
    chunk: Dict[str, Any],
    query_terms: List[str],
    topic: str,
    intents: List[str],
    cross_reference: bool,
    primary_intent: Optional[str] = None,
    labor_code_query: bool = False,
    user_query: str = "",
    special_category: Optional[str] = None,
    special_issue: Optional[str] = None,
    query_profile: Optional[Dict[str, Any]] = None,
) -> float:
    semantic = _safe_float(chunk.get("_best_similarity", _semantic_score(chunk)))
    hybrid_score = _safe_float(chunk.get("_hybrid_final_score"))
    keyword = _keyword_score(chunk, query_terms)
    exact_score = _exact_match_score(chunk, user_query, topic)

    briefing_mode_bonus = 0.0
    if topic == "occupational_briefing":
        mode = (
            "target"
            if _is_target_briefing_query(user_query)
            else (
                "responsible"
                if _is_responsible_briefing_query(user_query)
                else "generic"
            )
        )
        content_lower = str(chunk.get("content") or "").lower()
        document_lower = _get_document_name(chunk).lower()

        target_matches = sum(
            marker in content_lower
            for marker in (
                "целевой инструктаж",
                "разовых работ",
                "не связанных с прямыми обязанностями",
                "прямыми обязанностями",
                "наряд-допуск",
                "наряду-допуску",
            )
        )
        responsible_matches = sum(
            marker in content_lower
            for marker in (
                "вводный инструктаж",
                "проводит инструктаж",
                "специалист по охране труда",
                "уполномоченное должностное лицо",
            )
        )

        if mode == "target":
            briefing_mode_bonus += min(target_matches * 0.08, 0.24)
            briefing_mode_bonus -= min(responsible_matches * 0.04, 0.12)
            if "175" in document_lower:
                briefing_mode_bonus += 0.04
        elif mode == "responsible":
            briefing_mode_bonus += min(responsible_matches * 0.08, 0.24)
            briefing_mode_bonus -= min(target_matches * 0.04, 0.12)

    chunk["_briefing_mode_bonus"] = briefing_mode_bonus
    topic_score = min(_topic_relevance_score(chunk, topic), 1.0)
    intent_score = _intent_relevance_score(chunk, intents)
    primary_score = _primary_intent_relevance_score(chunk, primary_intent, topic)
    hits = int(chunk.get("_search_hits", 1))
    repeated_bonus = min(max(hits - 1, 0) * 0.025, 0.10)
    topic_weight = 0.06 if cross_reference else 0.08
    keyword_weight = 0.10 if cross_reference else 0.12

    labor_code_bonus = 0.0
    if labor_code_query:
        document_name = _get_document_name(chunk).lower()
        if (
            "трудовой кодекс" in document_name
            or "трудовои кодекс" in document_name
            or "трудовой_кодекс" in document_name
        ):
            labor_code_bonus = 0.10

    special_category_bonus = _minor_special_relevance_score(
        chunk,
        special_category,
        special_issue,
    )
    chunk["_special_category_bonus"] = special_category_bonus

    universal_score = _universal_query_relevance_score(
        chunk,
        query_profile or build_universal_query_profile(user_query),
    )
    chunk["_universal_score"] = universal_score

    return (
        semantic * 0.28
        + hybrid_score * 0.12
        + exact_score * 0.16
        + universal_score * 0.22
        + keyword * keyword_weight
        + topic_score * topic_weight
        + intent_score * 0.05
        + primary_score * 0.10
        + briefing_mode_bonus
        + labor_code_bonus
        + special_category_bonus
        + repeated_bonus
    )


def _legal_chunk_role(
    chunk: Dict[str, Any],
    topic: str,
    intents: List[str],
    primary_intent: Optional[str] = None,
) -> str:
    document = str(chunk.get("doc_name") or chunk.get("document") or "").lower()
    content = str(chunk.get("content") or chunk.get("text") or "").lower()
    text = f"{document} {content}"

    if topic == "ppe_nonprovision" and (
        "209" in document
        or "средств индивидуальной защиты" in document
        or "сиз" in document
    ) and any(
        marker in content
        for marker in (
            "не выдан",
            "невыдач",
            "поврежден",
            "неисправн",
            "отказ",
        )
    ):
        return "ppe_specific"

    employer_markers = (
        "обязанность нанимателя",
        "обязанности нанимателя",
        "наниматель обязан",
        "наниматель обеспечивает",
        "наниматель должен",
        "работодатель обязан",
        "обязан обеспечить",
    )

    employee_right_markers = (
        "имеет право",
        "право работника",
        "работник вправе",
        "вправе",
    )

    procedure_markers = (
        "действия работника",
        "немедленно сообщ",
        "уведом",
        "порядок действий",
        "не приступать",
        "приостановить работу",
    )

    danger_markers = (
        "угроз",
        "опасност",
        "жизни и здоров",
        "жизни или здоров",
    )

    liability_markers = (
        "ответственност",
        "штраф",
        "взыскан",
        "дисциплинарн",
    )

    responsible_person_markers = (
        "проводит вводный инструктаж",
        "проводит инструктаж",
        "специалист по охране труда",
        "уполномоченное должностное лицо",
        "руководитель организации",
        "руководитель структурного подразделения",
    )

    if primary_intent == "responsible_person":
        ordered_checks = [
            ("responsible_person", responsible_person_markers),
            ("employer_duty", employer_markers),
            ("procedure", procedure_markers),
            ("employee_right", employee_right_markers),
            ("danger", danger_markers),
            ("liability", liability_markers),
        ]
    elif primary_intent == "employer_duty":
        ordered_checks = [
            ("employer_duty", employer_markers),
            ("employee_right", employee_right_markers),
            ("procedure", procedure_markers),
            ("danger", danger_markers),
            ("liability", liability_markers),
        ]
    elif primary_intent == "employee_right":
        ordered_checks = [
            ("employee_right", employee_right_markers),
            ("employer_duty", employer_markers),
            ("procedure", procedure_markers),
            ("danger", danger_markers),
            ("liability", liability_markers),
        ]
    elif primary_intent in ("refusal", "danger", "procedure"):
        ordered_checks = [
            ("procedure", procedure_markers),
            ("danger", danger_markers),
            ("employee_right", employee_right_markers),
            ("employer_duty", employer_markers),
            ("liability", liability_markers),
        ]
    else:
        ordered_checks = [
            ("employee_right", employee_right_markers),
            ("employer_duty", employer_markers),
            ("procedure", procedure_markers),
            ("danger", danger_markers),
            ("liability", liability_markers),
        ]

    for role, markers in ordered_checks:
        if any(marker in text for marker in markers):
            return role

    return "general"



def _accident_worker_not_report_relevance_score(
    chunk: Dict[str, Any],
) -> float:
    """
    Узкий score для вопроса: работник/потерпевший не сообщил о НС
    непосредственному руководителю/нанимателю.

    Наличие слова «сообщить» само по себе недостаточно: нормы о сообщении
    в прокуратуру, ГИТ, страховщику, оформлении Н-1 и вручении акта относятся
    к другим этапам процедуры расследования.
    """
    document = _get_document_name(chunk).lower()
    point = _get_point_number(chunk).lower()
    content = str(chunk.get("content") or chunk.get("text") or "").lower()
    text = f"{document} {point} {content}"

    score = 0.0

    direct_phrases = (
        ("не сообщил", 0.45),
        ("не сообщила", 0.45),
        ("не сообщив", 0.40),
        ("сообщить о несчастном случае", 0.45),
        ("сообщить о происшедшем несчастном случае", 0.50),
        ("сообщить непосредственному руководителю", 0.60),
        ("непосредственному руководителю", 0.45),
        ("сообщить руководителю", 0.40),
        ("немедленно сообщить", 0.35),
        ("обязан сообщить", 0.35),
    )

    for phrase, weight in direct_phrases:
        if phrase in text:
            score += weight

    if "30" in document:
        score += 0.12
    if "81 144" in document or "81-144" in document:
        score += 0.10

    unrelated_markers = (
        "группов",
        "акт н-1",
        "форма н-1",
        "вручение потерпевшему",
        "вручение родственникам",
        "после окончания расследования",
        "прокуратур",
        "государственной инспекции труда",
        "страховщик",
    )
    unrelated_matches = sum(marker in text for marker in unrelated_markers)
    score -= min(unrelated_matches * 0.12, 0.48)

    return max(0.0, min(score, 1.50))


def _select_legal_diverse_chunks(
    ranked_chunks: List[Dict[str, Any]],
    limit: int,
    topic: str,
    intents: List[str],
    cross_reference: bool,
    primary_intent: Optional[str] = None,
    labor_code_query: bool = False,
    special_category: Optional[str] = None,
    special_issue: Optional[str] = None,
    accident_mode: Optional[str] = None,
    query_profile: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    if not ranked_chunks or limit <= 0:
        return []

    selected: List[Dict[str, Any]] = []
    selected_keys = set()
    roles_seen = set()
    documents_seen: Dict[str, int] = {}
    points_seen = set()

    def _key(chunk: Dict[str, Any]):
        return (
            _get_document_key(chunk),
            _get_point_number(chunk).lower(),
            str(chunk.get("content") or "")[:200].lower(),
        )

    def _point_key(chunk: Dict[str, Any]):
        point = _get_point_number(chunk).strip().lower()
        if not point:
            return None
        return (
            _get_document_key(chunk),
            point,
        )

    def _add(
        chunk: Dict[str, Any],
        max_per_document: int = 4,
        max_per_point: int = 1,
    ) -> bool:
        key = _key(chunk)
        document_key = _get_document_key(chunk)
        point_key = _point_key(chunk)

        if key in selected_keys:
            return False

        # Один и тот же пункт/статья не должен занимать несколько мест
        # финального юридического контекста. Разные пункты одного документа
        # по-прежнему могут попадать в ответ.
        if point_key is not None and point_key in points_seen:
            return False

        if documents_seen.get(document_key, 0) >= max_per_document:
            return False

        selected.append(chunk)
        selected_keys.add(key)
        documents_seen[document_key] = documents_seen.get(document_key, 0) + 1
        if point_key is not None:
            points_seen.add(point_key)
        return True

    if accident_mode == "worker_did_not_report":
        # Для этого intent финальный отбор идёт из узкого targeted-пула.
        # Общий vector/hybrid-поиск не должен вытеснять нужные нормы
        # правилами про Н-1, групповые НС, прокуратуру или вручение акта.
        targeted_pool = [
            chunk for chunk in ranked_chunks
            if chunk.get("_accident_worker_not_report_targeted")
        ]

        targeted_pool.sort(
            key=lambda chunk: (
                _accident_worker_not_report_relevance_score(chunk),
                _safe_float(chunk.get("_combined_score")),
            ),
            reverse=True,
        )

        for chunk in targeted_pool:
            relevance = _accident_worker_not_report_relevance_score(chunk)
            if relevance < 0.45:
                continue
            if _add(chunk, max_per_document=4, max_per_point=1):
                if len(selected) >= limit:
                    return selected

        # Добор разрешён только из основных НПА №30 и №81-144
        # и только при наличии прямой связи с сообщением о НС.
        fallback_pool = [
            chunk for chunk in ranked_chunks
            if _accident_worker_not_report_relevance_score(chunk) >= 0.35
            and (
                "30" in _get_document_name(chunk).lower()
                or "81 144" in _get_document_name(chunk).lower()
                or "81-144" in _get_document_name(chunk).lower()
            )
        ]
        fallback_pool.sort(
            key=lambda chunk: (
                _accident_worker_not_report_relevance_score(chunk),
                _safe_float(chunk.get("_combined_score")),
            ),
            reverse=True,
        )

        for chunk in fallback_pool:
            _add(chunk, max_per_document=4, max_per_point=1)
            if len(selected) >= limit:
                return selected

    if special_category == "minor":
        # Для конкретного ограничения несовершеннолетнего сначала выбираем
        # именно норму, регулирующую этот вопрос. Для holiday_weekend это
        # должна быть ст. 276, а не случайные статьи главы 20.
        if special_issue:
            issue_threshold = 0.70

            for chunk in ranked_chunks:
                special_score = _minor_special_relevance_score(
                    chunk,
                    special_category,
                    special_issue,
                )
                if special_score < issue_threshold:
                    continue

                # max_per_point=1 не позволяет двум chunks одной статьи
                # занимать два из пяти мест финального контекста.
                if _add(
                    chunk,
                    max_per_document=3,
                    max_per_point=1,
                ):
                    break

            # Если основной нормы недостаточно для заполнения контекста,
            # добираем только действительно релевантные нормы той же
            # специальной категории. Слабые совпадения (например, общие
            # статьи ТК о труде) сюда не должны попадать на раннем этапе.
            category_threshold = 0.50

            for chunk in ranked_chunks:
                if len(selected) >= min(3, limit):
                    break

                if _minor_special_relevance_score(
                    chunk,
                    special_category,
                    special_issue,
                ) < category_threshold:
                    continue

                _add(
                    chunk,
                    max_per_document=3,
                    max_per_point=1,
                )

    if primary_intent:
        for chunk in ranked_chunks:
            role = _legal_chunk_role(
                chunk,
                topic,
                intents,
                primary_intent=primary_intent,
            )

            if role != primary_intent:
                continue

            if labor_code_query and primary_intent == "employer_duty":
                document_name = _get_document_name(chunk).lower()
                is_labor_code = (
                    "трудовой кодекс" in document_name
                    or "трудовои кодекс" in document_name
                    or "трудовой_кодекс" in document_name
                )
                if not is_labor_code:
                    continue

            _add(chunk, max_per_document=3)

            if len(selected) >= min(2, limit):
                break

    if labor_code_query and len(selected) < limit:
        for chunk in ranked_chunks:
            document_name = _get_document_name(chunk).lower()
            is_labor_code = (
                "трудовой кодекс" in document_name
                or "трудовои кодекс" in document_name
                or "трудовой_кодекс" in document_name
            )

            if not is_labor_code:
                continue

            if _add(chunk, max_per_document=3):
                if len(selected) >= min(limit, 3):
                    break

    if cross_reference:
        max_per_document = 2
        for chunk in ranked_chunks:
            role = _legal_chunk_role(
                chunk,
                topic,
                intents,
                primary_intent=primary_intent,
            )

            if role in roles_seen:
                continue

            if _add(chunk, max_per_document=max_per_document):
                roles_seen.add(role)

            if len(selected) >= limit:
                return selected

    max_per_document = 3 if primary_intent else (2 if cross_reference else 4)

    for chunk in ranked_chunks:
        _add(chunk, max_per_document=max_per_document)
        if len(selected) >= limit:
            return selected

    for chunk in ranked_chunks:
        key = _key(chunk)
        if key in selected_keys:
            continue
        selected.append(chunk)
        selected_keys.add(key)
        if len(selected) >= limit:
            break

    return selected


# ============================================================
# ОСНОВНОЙ RAG
# ============================================================

async def retrieve_context(
    user_query: str,
    supabase,
) -> Dict[str, Any]:
    logger.info("RAG | query=%s", user_query)

    legal_domain = detect_legal_domain(user_query)
    topic = detect_topic(user_query)
    query_terms = _extract_query_terms(user_query)
    intents = detect_query_intents(user_query)
    cross_reference = is_cross_reference_query(intents)
    primary_intent = detect_primary_intent(intents, user_query)
    labor_code_query = _is_labor_code_query(user_query)
    special_category = detect_special_category(user_query)
    special_issue = _minor_special_issue(user_query) if special_category == "minor" else None
    query_profile = build_universal_query_profile(user_query)

    search_queries = build_search_queries(
        user_query,
        topic,
        legal_domain,
        intents,
    )

    logger.info(
        "RAG | query_input | %r",
        user_query,
    )
    logger.info(
        "RAG | classify | domain=%s | topic=%s | intents=%s | primary=%s | cross_reference=%s | labor_code=%s | special_category=%s | special_issue=%s | question_type=%s | subject=%s | event=%s | action=%s | state=%s",
        legal_domain,
        topic,
        intents,
        primary_intent,
        cross_reference,
        labor_code_query,
        special_category,
        special_issue,
        query_profile.get("question_type"),
        query_profile.get("subject"),
        query_profile.get("event"),
        query_profile.get("action"),
        query_profile.get("action_state"),
    )
    logger.info(
        "RAG | search_queries | count=%s | queries=%s",
        len(search_queries),
        search_queries,
    )

    query_vectors = await asyncio.to_thread(
        get_query_embeddings,
        search_queries,
    )

    valid_pairs = [
        (search_query, vector)
        for search_query, vector in zip(search_queries, query_vectors)
        if vector
    ]

    valid_queries = [search_query for search_query, _ in valid_pairs]
    query_vectors = [vector for _, vector in valid_pairs]

    for vector in query_vectors:
        if len(vector) != 384:
            raise ValueError(f"Unexpected embedding dimension: {len(vector)}. Expected 384.")

    if not query_vectors:
        logger.warning("RAG | all embeddings are empty")
        return {
            "chunks": [],
            "retrieved_text": "",
            "found": False,
            "candidate_count": 0,
            "final_count": 0,
            "source_references": [],
            "legal_domain": legal_domain,
            "topic": topic,
            "intents": intents,
            "cross_reference": cross_reference,
            "domain_specific_count": 0,
            "topic_specific_count": 0,
        }

    async def _run_search(
        vector: List[float],
        search_query: str,
        remove_domain_filter: bool = False,
    ) -> List[Dict[str, Any]]:
        return await asyncio.to_thread(
            _search_chunks,
            supabase,
            vector,
            search_query,
            None if remove_domain_filter else legal_domain,
            None,
        )

    search_tasks = [
        _run_search(
            vector,
            search_query,
            remove_domain_filter=(cross_reference and index == 0),
        )
        for index, (vector, search_query) in enumerate(zip(query_vectors, valid_queries))
    ]

    search_groups = await asyncio.gather(*search_tasks, return_exceptions=True)
    clean_groups: List[List[Dict[str, Any]]] = []

    for index, result in enumerate(search_groups):
        if isinstance(result, Exception):
            logger.warning(
                "RAG | search failed | query=%s | error=%s",
                valid_queries[index],
                result,
            )
            clean_groups.append([])
            continue

        clean_groups.append(result or [])

    query_roles = [
        "main" if index == 0 else "expanded"
        for index in range(len(clean_groups))
    ]

    candidate_chunks = _merge_search_results(
        clean_groups,
        query_roles,
    )

    targeted_chunks = await _get_targeted_chunks(
        supabase,
        topic,
        user_query,
    )

    if targeted_chunks:
        accident_worker_not_report_mode = (
            topic == "accident_investigation"
            and _accident_query_mode(user_query) == "worker_did_not_report"
        )

        if accident_worker_not_report_mode:
            for chunk in targeted_chunks:
                chunk["_accident_worker_not_report_targeted"] = True

        targeted_merged = _merge_search_results([targeted_chunks], ["targeted"])
        candidate_chunks = _merge_search_results(
            [candidate_chunks, targeted_merged],
            ["merged", "targeted"],
        )

    candidate_count = len(candidate_chunks)

    if not candidate_chunks:
        return {
            "chunks": [],
            "retrieved_text": "",
            "found": False,
            "candidate_count": 0,
            "final_count": 0,
            "source_references": [],
            "legal_domain": legal_domain,
            "topic": topic,
            "intents": intents,
            "cross_reference": cross_reference,
            "domain_specific_count": 0,
            "topic_specific_count": 0,
        }

    domain_specific_count = sum(
        1 for chunk in candidate_chunks if chunk.get("legal_domain") == legal_domain
    )
    topic_specific_count = sum(
        1 for chunk in candidate_chunks if chunk.get("topic") == topic
    )

    for chunk in candidate_chunks:
        # Передаём исходный вопрос в topic-score, чтобы различать
        # подтипы внутри workplace_attestation.
        chunk["_user_query_for_scoring"] = user_query

        chunk["_combined_score"] = _legal_relevance_score(
            chunk,
            query_terms,
            topic,
            intents,
            cross_reference,
            primary_intent=primary_intent,
            labor_code_query=labor_code_query,
            user_query=user_query,
            special_category=special_category,
            special_issue=special_issue,
            query_profile=query_profile,
        )

    ranked_chunks = sorted(
        candidate_chunks,
        key=lambda chunk: chunk.get("_combined_score", 0.0),
        reverse=True,
    )

    for rank, chunk in enumerate(ranked_chunks[:10], start=1):
        logger.info(
            "RAG | candidate | rank=%s | score=%.4f | sim=%.4f | exact=%.4f | universal=%.4f | topic=%.4f | intent=%.4f | primary=%.4f | briefing=%.4f | special=%.4f | doc=%s | point=%s",
            rank,
            _safe_float(chunk.get("_combined_score")),
            _safe_float(chunk.get("_best_similarity", _semantic_score(chunk))),
            _safe_float(_exact_match_score(chunk, user_query, topic)),
            _safe_float(chunk.get("_universal_score")),
            _safe_float(_topic_relevance_score(chunk, topic)),
            _safe_float(_intent_relevance_score(chunk, intents)),
            _safe_float(_primary_intent_relevance_score(chunk, primary_intent, topic)),
            _safe_float(chunk.get("_briefing_mode_bonus")),
            _safe_float(chunk.get("_special_category_bonus")),
            _get_document_name(chunk),
            _get_point_number(chunk),
        )

    final_limit = (
        max(RAG_FINAL_COUNT, 7)
        if cross_reference
        else RAG_FINAL_COUNT
    )

    accident_mode = (
        _accident_query_mode(user_query)
        if topic == "accident_investigation"
        else None
    )

    final_chunks = _select_legal_diverse_chunks(
        ranked_chunks,
        final_limit,
        topic,
        intents,
        cross_reference,
        primary_intent=primary_intent,
        labor_code_query=labor_code_query,
        special_category=special_category,
        special_issue=special_issue,
        accident_mode=accident_mode,
        query_profile=query_profile,
    )

    logger.info(
        "RAG | final | count=%s | accident_mode=%s | special_category=%s | special_issue=%s | sources=%s",
        len(final_chunks),
        accident_mode,
        special_category,
        special_issue,
        [
            f"{_get_document_name(chunk)}#{_get_point_number(chunk)}"
            for chunk in final_chunks
        ],
    )

    source_references = []

    for index, chunk in enumerate(final_chunks, start=1):
        source_id = build_source_id(chunk, index)
        chunk["_source_id"] = source_id

        document_name = _get_document_name(chunk)
        point = _get_point_number(chunk)

        if point:
            reference = f"{document_name} — пункт/статья {point}"
        else:
            reference = document_name

        source_references.append(
            {
                "source_id": source_id,
                "reference": reference,
            }
        )

    retrieved_text = _build_retrieved_text(final_chunks)

    return {
        "chunks": final_chunks,
        "retrieved_text": retrieved_text,
        "found": bool(final_chunks),
        "candidate_count": candidate_count,
        "final_count": len(final_chunks),
        "source_references": source_references,
        "legal_domain": legal_domain,
        "topic": topic,
        "intents": intents,
        "query_profile": query_profile,
        "cross_reference": cross_reference,
        "domain_specific_count": domain_specific_count,
        "topic_specific_count": topic_specific_count,
    }


# ============================================================
# ИСТОЧНИКИ
# ============================================================

def get_source_references(
    chunks: List[Dict[str, Any]],
) -> List[str]:
    references = []
    seen = set()

    for chunk in chunks:
        document_name = _get_document_name(chunk)
        point = _get_point_number(chunk)

        if point:
            reference = f"{document_name} — пункт/статья {point}"
        else:
            reference = document_name

        if reference not in seen:
            seen.add(reference)
            references.append(reference)

    return references


def get_source_names(
    chunks: List[Dict[str, Any]],
) -> List[str]:
    names = []
    seen = set()

    for chunk in chunks:
        document_name = _get_document_name(chunk)
        if document_name not in seen:
            seen.add(document_name)
            names.append(document_name)

    return names