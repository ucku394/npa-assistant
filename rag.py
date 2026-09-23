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

# Для юридического RAG берём больше кандидатов,
# чтобы нужная норма не потерялась на этапе первичного поиска.
RAG_CANDIDATE_COUNT = max(
    RAG_FINAL_COUNT * 10,
    50,
)

RAG_MIN_CANDIDATES = 10

# Сколько точечных нормативных кандидатов дополнительно
# получать через обычный текстовый поиск Supabase.
TARGETED_SEARCH_LIMIT = 50


# ============================================================
# ОПРЕДЕЛЕНИЕ ОБЛАСТИ ЗАПРОСА
# ============================================================

def detect_legal_domain(user_query: str) -> str:
    """
    Определяет основную правовую область запроса.

    Приоритет:
    1. промышленная безопасность
    2. пожарная безопасность
    3. электробезопасность
    4. охрана труда
    5. санитарные требования
    6. general
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

    if any(
        re.search(pattern, query, flags=re.IGNORECASE)
        for pattern in industrial_patterns
    ):
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

    if any(
        re.search(pattern, query, flags=re.IGNORECASE)
        for pattern in fire_patterns
    ):
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

    if any(
        re.search(pattern, query, flags=re.IGNORECASE)
        for pattern in electrical_patterns
    ):
        return "electrical_safety"

    occupational_patterns = [
        # Несчастные случаи / расследование / акт Н-1.
        # Эти маркеры нужны, даже если пользователь не произносит
        # фразу «несчастный случай» напрямую.
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

    if any(
        re.search(pattern, query, flags=re.IGNORECASE)
        for pattern in occupational_patterns
    ):
        return "occupational_safety"

    sanitary_patterns = [
        r"\bсанитар\w*",
        r"\bсанитарно-эпидемиологическ\w*",
        r"\bгигиен\w*",
        r"\bмикроклимат\w*",
        r"\bсанитарн\w*\s+норм\w*",
    ]

    if any(
        re.search(pattern, query, flags=re.IGNORECASE)
        for pattern in sanitary_patterns
    ):
        return "sanitary"

    return "general"


# ============================================================
# ОПРЕДЕЛЕНИЕ ТЕМЫ
# ============================================================

def detect_topic(user_query: str) -> str:
    """
    Определяет специализированную тему запроса.

    Внутренние темы RAG:

    medical_examinations
    workplace_attestation
    accident_investigation
    occupational_training
    ppe_nonprovision

    Для medical_examinations и ppe_nonprovision
    topic_filter БД отключён, поскольку эти темы могут
    отсутствовать в существующем поле topic.
    """

    query = str(user_query or "").strip().lower()

    if not query:
        return "general"

    # --------------------------------------------------------
    # СИЗ — НЕВЫДАЧА / ПОВРЕЖДЕНИЕ / ОТКАЗ ОТ РАБОТЫ
    # --------------------------------------------------------

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

    if any(
        re.search(pattern, query, flags=re.IGNORECASE)
        for pattern in ppe_nonprovision_patterns
    ):
        return "ppe_nonprovision"

    # --------------------------------------------------------
    # МЕДИЦИНСКИЕ ОСМОТРЫ
    # --------------------------------------------------------

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

    if any(
        re.search(pattern, query, flags=re.IGNORECASE)
        for pattern in medical_exam_patterns
    ):
        return "medical_examinations"

    # --------------------------------------------------------
    # АТТЕСТАЦИЯ РАБОЧИХ МЕСТ
    # --------------------------------------------------------

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

    if any(
        re.search(pattern, query, flags=re.IGNORECASE)
        for pattern in workplace_attestation_patterns
    ):
        return "workplace_attestation"

    # --------------------------------------------------------
    # ВВОДНЫЙ ИНСТРУКТАЖ / ЛИЦО, ПРОВОДЯЩЕЕ ИНСТРУКТАЖ
    # --------------------------------------------------------

    occupational_briefing_patterns = [
        r"\bвводн\w*\s+инструктаж\w*",
        r"\bкто\s+провод\w*.*\bинструктаж\w*",
        r"\bпровод\w*.*\bвводн\w*\s+инструктаж\w*",
        r"\bлиц\w*.*\bпровод\w*.*\bвводн\w*\s+инструктаж\w*",
        r"\bответственн\w*.*\bвводн\w*\s+инструктаж\w*",
        r"\bспециалист\w*\s+по\s+охран\w*\s+труд\w*.*\bинструктаж\w*",

        # Целевой инструктаж: разовые работы, не связанные
        # с прямыми обязанностями работника, включая работы
        # по наряду-допуску.
        r"\bразов\w*\s+работ\w*.*\bне\s+связан\w*.*\bпрям\w*\s+обязанност\w*",
        r"\bне\s+связан\w*\s+с\s+прям\w*\s+обязанност\w*",
        r"\bпрям\w*\s+обязанност\w*.*\bразов\w*\s+работ\w*",
        r"\bнаряд\w*[-–—]?\s*допуск\w*",
        r"\bнаряд\w*\s+допуск\w*",
        r"\bцелев\w*\s+инструктаж\w*",
    ]

    if any(
        re.search(pattern, query, flags=re.IGNORECASE)
        for pattern in occupational_briefing_patterns
    ):
        return "occupational_briefing"

    # --------------------------------------------------------
    # СТАЖИРОВКА / ДОПУСК К САМОСТОЯТЕЛЬНОЙ РАБОТЕ
    # --------------------------------------------------------
    # Требования к стажировке могут быть разделены между
    # несколькими пунктами одного НПА: отдельный пункт
    # устанавливает необходимость стажировки, а другой —
    # её продолжительность. Поэтому выделяем запросы
    # о стажировке в отдельную тему и запускаем
    # специальный текстовый поиск.

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

    if any(
        re.search(pattern, query, flags=re.IGNORECASE)
        for pattern in occupational_training_patterns
    ):
        return "occupational_training"

    # --------------------------------------------------------
    # НЕСЧАСТНЫЕ СЛУЧАИ
    # --------------------------------------------------------

    accident_patterns = [
        # Прямые признаки несчастного случая.
        r"\bнесчастн\w*\s+случа\w*",
        r"\bрасследован\w*\s+несчастн\w*",
        r"\bучет\w*\s+несчастн\w*",
        r"\bпотерпевш\w*",
        r"\bтравм\w*\s+на\s+производств\w*",
        r"\bтравм\w*\s+работник\w*",
        r"\bпроисшеств\w*\s+на\s+производств\w*",

        # Сильные юридические маркеры, по которым тему
        # можно определить без фразы «несчастный случай».
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

    if any(
        re.search(pattern, query, flags=re.IGNORECASE)
        for pattern in accident_patterns
    ):
        return "accident_investigation"

    return "general"


def _get_topic_filter(topic: str) -> Optional[str]:
    """
    Для general фильтр по topic отключаем.

    Для medical_examinations и ppe_nonprovision также
    отключаем фильтр, поскольку существующие записи БД
    могут ещё иметь topic=None/general/другое значение.

    Для остальных специализированных тем используем фильтр.
    """

    if not topic or topic in (
        "general",
        "medical_examinations",
        "ppe_nonprovision",
        # Для стажировки связанные пункты одного НПА могут
        # иметь разные/пустые значения topic.
        "occupational_training",
        # Не ограничиваем RPC topic-фильтром для несчастных
        # случаев: старые записи базы могут иметь topic=None/general.
        "accident_investigation",
    ):
        return None

    return topic


# ============================================================
# КЛЮЧЕВЫЕ СЛОВА
# ============================================================

def _extract_query_terms(user_query: str) -> List[str]:
    """
    Выделяет важные юридические термины.
    """

    query = str(user_query or "").strip().lower()

    terms: List[str] = []

    keyword_patterns = [

        # ----------------------------------------------------
        # СИЗ — НЕВЫДАЧА / ПОВРЕЖДЕНИЕ / ОТКАЗ
        # ----------------------------------------------------

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
        r"\bотказ\w*.*\bработ\w*",
        r"\bотказ\w*.*\bвыполнен\w*",

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

        # ----------------------------------------------------
        # МЕДИЦИНСКИЕ ОСМОТРЫ
        # ----------------------------------------------------

        r"\bмедицинск\w*\s+осмотр\w*",
        r"\bмедицинск\w*",
        r"\bмедосмотр\w*",
        r"\bпредварительн\w*",
        r"\bпериодическ\w*",
        r"\bвнеочередн\w*",
        r"\bобязательн\w*",
        r"\bработник\w*",
        r"\bработающ\w*",

        # Финансирование / оплата.
        r"\bза\s+чей\s+счет\b",
        r"\bза\s+чей\s+сч[её]т\b",
        r"\bза\s+сч[её]т\b",
        r"\bсчет\b",
        r"\bоплат\w*",
        r"\bфинанс\w*",
        r"\bрасход\w*",
        r"\bзатрат\w*",
        r"\bсредств\w*",

        # ----------------------------------------------------
        # СИЗ — ОБЩИЕ ВОПРОСЫ
        # ----------------------------------------------------

        r"\bиспользован\w*",
        r"\bиспользовани\w*",
        r"\bбывш\w*\s+в\s+употреблен\w*",
        r"\bпериод\w*\s+использован\w*",
        r"\bсрок\w*\s+носк\w*",

        # ----------------------------------------------------
        # АТТЕСТАЦИЯ
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # НЕСЧАСТНЫЕ СЛУЧАИ / АКТ Н-1 / РАССЛЕДОВАНИЕ
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # СТАЖИРОВКА / ДОПУСК / МИНИМАЛЬНАЯ ПРОДОЛЖИТЕЛЬНОСТЬ
        # ----------------------------------------------------

        r"\bстажиров\w*",
        r"\bпродолжительност\w*\s+стажиров\w*",
        r"\bсрок\w*\s+стажиров\w*",
        r"\bне\s+менее\s+двух\b",
        r"\bрабоч\w*\s+дн\w*",
        r"\bрабоч\w*\s+смен\w*",
        r"\bповышенн\w*\s+опасност\w*",
        r"\bсамостоятельн\w*\s+работ\w*",
        r"\bдопуск\w*",
        r"\bпровер\w*\s+знан\w*",
        r"\b№\s*175\b",

        # ----------------------------------------------------
        # ПРОМЫШЛЕННАЯ БЕЗОПАСНОСТЬ
        # ----------------------------------------------------

        r"\bпромышленн\w*\s+безопасност\w*",
        r"\bопасн\w*\s+производственн\w*\s+объект\w*",
        r"\bопо\b",
        r"\bпоо\b",
        r"\bпроизводственн\w*\s+контрол\w*",
        r"\bтехническ\w*\s+устройств\w*",
        r"\bавари\w*",
        r"\bинцидент\w*",

        # ----------------------------------------------------
        # ПОЖАРНАЯ БЕЗОПАСНОСТЬ
        # ----------------------------------------------------

        r"\bпожарн\w*\s+безопасност\w*",
        r"\bпожар\w*",
        r"\bогнетушител\w*",
        r"\bэвакуац\w*",
    ]

    for pattern in keyword_patterns:

        matches = re.findall(
            pattern,
            query,
            flags=re.IGNORECASE,
        )

        for match in matches:

            value = str(match).strip().lower()

            if value and value not in terms:
                terms.append(value)

    return terms


# ============================================================
# KEYWORD SCORE
# ============================================================

def _keyword_score(
    chunk: Dict[str, Any],
    query_terms: List[str],
) -> float:

    if not query_terms:
        return 0.0

    document_name = str(
        chunk.get("doc_name")
        or chunk.get("document")
        or ""
    ).lower()

    content = str(
        chunk.get("content")
        or chunk.get("text")
        or ""
    ).lower()

    point = str(
        chunk.get("point_num")
        or ""
    ).lower()

    text = f"{document_name} {content} {point}"

    matched = 0

    for term in query_terms:

        if term in text:
            matched += 1

    return matched / len(query_terms)


# ============================================================
# СПЕЦИАЛЬНЫЙ SCORE ДЛЯ ЮРИДИЧЕСКОЙ ТЕМЫ
# ============================================================

def _topic_relevance_score(
    chunk: Dict[str, Any],
    topic: str,
) -> float:
    """
    Дополнительный score.

    Для специализированных юридических вопросов
    профильный нормативный документ получает преимущество
    перед просто семантически похожим документом.
    """

    if not topic or topic == "general":
        return 0.0

    document_name = str(
        chunk.get("doc_name")
        or chunk.get("document")
        or ""
    ).lower()

    content = str(
        chunk.get("content")
        or chunk.get("text")
        or ""
    ).lower()

    db_topic = str(
        chunk.get("topic")
        or ""
    ).lower()

    score = 0.0

    # --------------------------------------------------------
    # ВВОДНЫЙ ИНСТРУКТАЖ
    # --------------------------------------------------------

    if topic == "occupational_briefing":
        if db_topic == "occupational_briefing":
            score += 1.0

        # Для вопросов о разовых работах по наряду-допуску
        # приоритет имеет именно норма о целевом инструктаже.
        target_instruction_markers = [
            "целевой инструктаж",
            "разовых работ",
            "не связанных с прямыми обязанностями",
            "прямыми обязанностями",
            "наряд-допуск",
            "наряду-допуску",
        ]

        target_matches = sum(
            1
            for marker in target_instruction_markers
            if marker in content
        )

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

        matches = sum(
            1
            for marker in briefing_markers
            if marker in content or marker in document_name
        )

        if matches:
            score += min(matches * 0.30, 1.20)

        if "175" in document_name:
            score += 0.80

    # --------------------------------------------------------
    # СИЗ — НЕВЫДАЧА / ПОВРЕЖДЕНИЕ / ОТКАЗ
    # --------------------------------------------------------

    if topic == "ppe_nonprovision":

        if db_topic == "ppe_nonprovision":
            score += 1.0

        # Основной профильный документ:
        # Правила по обеспечению СИЗ №209.
        is_npa_209 = bool(
            re.search(
                r"№\s*209\b",
                document_name,
                flags=re.IGNORECASE,
            )
            or re.search(
                r"\b209\b",
                document_name,
                flags=re.IGNORECASE,
            )
        )

        if is_npa_209:
            score += 1.20

        # Если название документа явно связано с СИЗ.
        if (
            "сиз" in document_name
            or "средств" in document_name
            and "индивидуальн" in document_name
            and "защит" in document_name
        ):
            score += 0.70

        # Общая тематическая релевантность содержимого.
        if "сиз" in content:
            score += 0.35

        if (
            "средств" in content
            and "индивидуальн" in content
            and "защит" in content
        ):
            score += 0.35

        # Ключевые признаки именно проблемной ситуации.
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

        problem_matches = sum(
            1
            for marker in problem_markers
            if marker in content
        )

        if problem_matches:
            score += min(
                problem_matches * 0.25,
                0.75,
            )

    # --------------------------------------------------------
    # МЕДИЦИНСКИЕ ОСМОТРЫ
    # --------------------------------------------------------

    elif topic == "medical_examinations":

        if db_topic == "medical_examinations":
            score += 1.0

        # Основной профильный документ:
        # Постановление №74
        if (
            re.search(
                r"№\s*74\b",
                document_name,
                flags=re.IGNORECASE,
            )
            or re.search(
                r"\b74\b",
                document_name,
                flags=re.IGNORECASE,
            )
        ):
            score += 1.20

        # Название документа прямо говорит
        # об обязательных и внеочередных медосмотрах.
        if (
            "медицинск" in document_name
            and "осмотр" in document_name
        ):
            score += 0.80

        if "медицинск" in content:
            score += 0.30

        if "осмотр" in content:
            score += 0.30

        # Особый дополнительный вес для вопросов
        # о финансировании/оплате.
        financing_markers = [
            "за счет",
            "за счёт",
            "оплата",
            "расход",
            "средств",
            "финанс",
            "затрат",
        ]

        if any(
            marker in content
            for marker in financing_markers
        ):
            score += 0.60

    # --------------------------------------------------------
    # АТТЕСТАЦИЯ
    # --------------------------------------------------------

    elif topic == "workplace_attestation":

        if db_topic == "workplace_attestation":
            score += 1.0

        if "аттестаци" in document_name:
            score += 0.80

        if "аттестаци" in content:
            score += 0.35

        if "рабоч" in content and "мест" in content:
            score += 0.15

        # Постановление №253 — специальный boost.
        if (
            "253" in document_name
            and (
                "аттестаци" in document_name
                or "аттестаци" in content
                or "рабоч" in content
            )
        ):
            score += 1.50

    # --------------------------------------------------------
    # НЕСЧАСТНЫЕ СЛУЧАИ
    # --------------------------------------------------------

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

        # Профильные признаки формы Н-1 имеют больший вес,
        # чем общая семантическая близость к ТК/КоАП.
        if re.search(r"\bн[-–—]?\s*1\b", content, re.IGNORECASE):
            score += 0.90

        if "акт" in content and re.search(
            r"\bн[-–—]?\s*1\b",
            content,
            re.IGNORECASE,
        ):
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

        marker_matches = sum(
            1
            for marker in accident_markers
            if marker in content
        )

        if marker_matches:
            score += min(
                marker_matches * 0.15,
                0.75,
            )

    return min(score, 2.5)


# ============================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

def _safe_float(
    value: Any,
    default: float = 0.0,
) -> float:

    try:
        return float(value)

    except (TypeError, ValueError):
        return default


def _normalize_identifier(
    value: Any,
) -> str:

    if value is None:
        return ""

    text = str(value).strip()

    text = re.sub(
        r"\s+",
        "_",
        text,
    )

    text = re.sub(
        r"[^A-Za-zА-Яа-яЁё0-9_./-]+",
        "",
        text,
    )

    text = re.sub(
        r"_+",
        "_",
        text,
    )

    return text.strip("_")


def _normalize_point_identifier(
    value: Any,
) -> str:

    if value is None:
        return ""

    text = str(value).strip()

    text = re.sub(
        r"[.,;:]+$",
        "",
        text,
    )

    text = re.sub(
        r"\s+",
        "_",
        text,
    )

    text = re.sub(
        r"[^A-Za-zА-Яа-яЁё0-9_.-]+",
        "",
        text,
    )

    text = re.sub(
        r"_+",
        "_",
        text,
    )

    text = re.sub(
        r"[.,;:]+$",
        "",
        text,
    )

    return text.strip("_")


def _normalize_source_id(
    value: Any,
) -> str:

    if value is None:
        return ""

    text = str(value).strip()

    text = re.sub(
        r"[.,;:]+$",
        "",
        text,
    )

    return _normalize_identifier(text)


def _get_document_name(
    chunk: Dict[str, Any],
) -> str:

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


def _get_point_number(
    chunk: Dict[str, Any],
) -> str:

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

    text = (
        chunk.get("content")
        or chunk.get("text")
        or chunk.get("chunk_text")
        or ""
    )

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

        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        if match:
            return match.group(1)

    return ""


def _extract_npa_number(
    document_name: str,
) -> str:

    if not document_name:
        return ""

    patterns = [
        r"№\s*([0-9]+(?:[-/][A-Za-zА-Яа-я0-9]+)*)",
        r"N\s*([0-9]+(?:[-/][A-Za-zА-Яа-я0-9]+)*)",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            document_name,
            flags=re.IGNORECASE,
        )

        if match:
            return match.group(1)

    return ""


def build_source_id(
    chunk: Dict[str, Any],
    index: int = 0,
) -> str:

    existing_source_id = (
        chunk.get("source_id")
        or chunk.get("_source_id")
    )

    if existing_source_id:
        return _normalize_source_id(
            existing_source_id
        )

    document_name = _get_document_name(chunk)
    point = _get_point_number(chunk)

    npa_number = _extract_npa_number(
        document_name
    )

    if npa_number:

        base = f"NPA_{npa_number}"

    else:

        normalized_document = _normalize_identifier(
            document_name
        )

        base = (
            f"NPA_{normalized_document[:80]}"
            if normalized_document
            else "NPA_UNKNOWN"
        )

    if point:

        normalized_point = _normalize_point_identifier(
            point
        )

        if normalized_point:
            return f"{base}_P{normalized_point}"

    text = (
        chunk.get("content")
        or chunk.get("text")
        or chunk.get("chunk_text")
        or ""
    )

    stable_string = (
        f"{document_name}|{str(text).strip()}"
    )

    digest = hashlib.sha1(
        stable_string.encode("utf-8")
    ).hexdigest()[:12]

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

    text = (
        chunk.get("content")
        or chunk.get("text")
        or chunk.get("chunk_text")
        or ""
    )

    source_id = build_source_id(
        chunk,
        index,
    )

    chunk["_source_id"] = source_id

    legal_domain = (
        chunk.get("legal_domain")
        or ""
    )

    topic = chunk.get("topic")

    semantic = _semantic_score(chunk)
    combined = chunk.get("_combined_score")

    lines = [
        f"SOURCE_ID: {source_id}",
        f"DOCUMENT: {document_name}",
    ]

    if legal_domain:
        lines.append(
            f"LEGAL_DOMAIN: {legal_domain}"
        )

    if topic:
        lines.append(
            f"TOPIC: {topic}"
        )

    if point:
        lines.append(
            f"POINT_OR_ARTICLE: {point}"
        )

    lines.append(
        f"SEMANTIC_SCORE: {semantic:.4f}"
    )

    if combined is not None:
        lines.append(
            f"RAG_SCORE: {float(combined):.4f}"
        )

    lines.append("TEXT:")
    lines.append(str(text).strip())

    return "\n".join(lines)


def _build_retrieved_text(
    chunks: List[Dict[str, Any]],
) -> str:

    blocks = []

    for index, chunk in enumerate(
        chunks,
        start=1,
    ):

        source_id = build_source_id(
            chunk,
            index,
        )

        chunk["_source_id"] = source_id

        block = [
            f"===== RAG SOURCE {index} =====",
            _format_chunk(
                chunk,
                index,
            ),
            f"===== END RAG SOURCE {index} =====",
        ]

        blocks.append(
            "\n".join(block)
        )

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
    """
    Hybrid retrieval:
    semantic HNSW + lexical PGroonga + RRF.

    Fallback использует актуальный semantic_search_npa_chunks RPC.
    Старый match_npa_chunks_v3 намеренно не используется.
    """

    try:
        response = (
            supabase
            .rpc(
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
            )
            .execute()
        )

        data = response.data or []

        # Сохраняем привычное поле similarity для всего
        # существующего legal rerank-кода.
        for chunk in data:
            chunk["similarity"] = _safe_float(
                chunk.get("semantic_score"),
                _safe_float(chunk.get("similarity")),
            )
            chunk["_hybrid_rrf_score"] = _safe_float(
                chunk.get("rrf_score")
            )
            chunk["_hybrid_final_score"] = _safe_float(
                chunk.get("final_score")
            )

        return data

    except Exception as exc:
        logger.warning(
            "RAG | hybrid search failed; fallback to semantic_search_npa_chunks | error=%s",
            exc,
        )

        # Fallback использует актуальный RPC. Старый v3 RPC
        # намеренно не вызываем: он не является частью текущего
        # hybrid-search контура и может давать 404 из-за API cache.
        response = (
            supabase
            .rpc(
                "semantic_search_npa_chunks",
                {
                    "query_embedding": query_vector,
                    "match_count": RAG_CANDIDATE_COUNT,
                    "domain_filter": legal_domain,
                    "topic_filter": topic_filter,
                },
            )
            .execute()
        )

        data = response.data or []

        for chunk in data:
            chunk["similarity"] = _safe_float(
                chunk.get("similarity")
            )
            chunk["_hybrid_rrf_score"] = 0.0
            chunk["_hybrid_final_score"] = 0.0

        return data


# ============================================================
# TARGETED SEARCH — АТТЕСТАЦИЯ
# ============================================================

def _targeted_attestation_search(
    supabase,
) -> List[Dict[str, Any]]:

    results: List[Dict[str, Any]] = []

    queries = [
        "doc_name.ilike.%253%",
        "doc_name.ilike.%аттестаци%",
        "content.ilike.%аттестаци%",
    ]

    for query in queries:

        try:

            response = (
                supabase
                .table("npa_chunks")
                .select(
                    "doc_name,"
                    "doc_type,"
                    "point_num,"
                    "content,"
                    "legal_domain,"
                    "topic,"
                    "source_url"
                )
                .eq(
                    "legal_domain",
                    "occupational_safety",
                )
                .or_(query)
                .limit(
                    TARGETED_SEARCH_LIMIT
                )
                .execute()
            )

            data = response.data or []

            results.extend(data)

        except Exception as exc:

            logger.warning(
                "RAG | targeted attestation search failed: %s",
                exc,
            )

    return _deduplicate_chunks(results)


# ============================================================
# TARGETED SEARCH — МЕДИЦИНСКИЕ ОСМОТРЫ
# ============================================================

def _targeted_medical_exam_search(
    supabase,
) -> List[Dict[str, Any]]:
    """
    Точечный поиск нормативных фрагментов
    по обязательным медицинским осмотрам.

    Главная цель — гарантированно подтянуть
    Постановление №74, даже если embedding
    поставил другой документ выше.
    """

    results: List[Dict[str, Any]] = []

    queries = [
        "doc_name.ilike.%74%",
        "doc_name.ilike.%медицинск%",
        "doc_name.ilike.%осмотр%",
        "content.ilike.%медицинск%",
        "content.ilike.%медосмотр%",
    ]

    for query in queries:

        try:

            response = (
                supabase
                .table("npa_chunks")
                .select(
                    "doc_name,"
                    "doc_type,"
                    "point_num,"
                    "content,"
                    "legal_domain,"
                    "topic,"
                    "source_url"
                )
                .eq(
                    "legal_domain",
                    "occupational_safety",
                )
                .or_(query)
                .limit(
                    TARGETED_SEARCH_LIMIT
                )
                .execute()
            )

            data = response.data or []

            results.extend(data)

        except Exception as exc:

            logger.warning(
                "RAG | targeted medical exam search failed: %s",
                exc,
            )

    return _deduplicate_chunks(results)


# ============================================================
# TARGETED SEARCH — СИЗ / НЕВЫДАЧА / ОТКАЗ
# ============================================================

def _targeted_ppe_nonprovision_search(
    supabase,
) -> List[Dict[str, Any]]:
    """
    Точечный поиск по ситуациям, когда:

    - СИЗ не выданы;
    - СИЗ повреждены;
    - СИЗ неисправны;
    - работник не обеспечен СИЗ;
    - работник отказывается от выполнения работы;
    - работник не приступает к работе;
    - работа приостанавливается.

    Главная цель — подтянуть нужные пункты
    Правил по обеспечению СИЗ №209 даже тогда,
    когда обычный embedding ставит общие пункты
    №209 выше релевантной нормы.
    """

    results: List[Dict[str, Any]] = []

    queries = [
        # Профильный НПА.
        "doc_name.ilike.%209%",

        # Общие упоминания СИЗ.
        "doc_name.ilike.%СИЗ%",
        "doc_name.ilike.%средств%индивидуальн%защит%",

        # Конкретная проблемная ситуация.
        "content.ilike.%не выдан%",
        "content.ilike.%невыдач%",
        "content.ilike.%поврежден%",
        "content.ilike.%поврежд%",
        "content.ilike.%неисправн%",
        "content.ilike.%не обеспечен%",

        # Действия / право работника.
        "content.ilike.%отказ%",
        "content.ilike.%не приступ%",
        "content.ilike.%приостанов%",
        "content.ilike.%работник%",
    ]

    for query in queries:

        try:

            response = (
                supabase
                .table("npa_chunks")
                .select(
                    "doc_name,"
                    "doc_type,"
                    "point_num,"
                    "content,"
                    "legal_domain,"
                    "topic,"
                    "source_url"
                )
                .eq(
                    "legal_domain",
                    "occupational_safety",
                )
                .or_(query)
                .limit(
                    TARGETED_SEARCH_LIMIT
                )
                .execute()
            )

            data = response.data or []

            results.extend(data)

        except Exception as exc:

            logger.warning(
                "RAG | targeted PPE nonprovision search failed: %s",
                exc,
            )

    return _deduplicate_chunks(results)


# ============================================================
# TARGETED SEARCH — НЕСЧАСТНЫЕ СЛУЧАИ
# ============================================================

def _targeted_occupational_training_search(
    supabase,
) -> List[Dict[str, Any]]:
    """
    Точечный поиск по стажировке.

    Вопрос о минимальной продолжительности стажировки
    требует связать положения одного НПА, которые могут
    находиться в разных пунктах. Поэтому ищем одновременно
    номер Инструкции №175, стажировку, количественную норму
    и условия допуска к самостоятельной работе.
    """

    results: List[Dict[str, Any]] = []

    queries = [
        # Профильный НПА.
        "doc_name.ilike.%175%",
        "doc_name.ilike.%Инструкци%",

        # Стажировка и продолжительность.
        "content.ilike.%стажиров%",
        "content.ilike.%продолжительн%стажиров%",
        "content.ilike.%срок%стажиров%",

        # Количественная норма.
        "content.ilike.%не менее двух%",
        "content.ilike.%рабочих дней%",
        "content.ilike.%рабочих смен%",

        # Связь с повышенной опасностью и допуском.
        "content.ilike.%повышенной опасностью%",
        "content.ilike.%допуск к самостоятельной работе%",
        "content.ilike.%самостоятельной работе%",
        "content.ilike.%проверка знаний%",
    ]

    for query in queries:

        try:

            response = (
                supabase
                .table("npa_chunks")
                .select(
                    "doc_name,"
                    "doc_type,"
                    "point_num,"
                    "content,"
                    "legal_domain,"
                    "topic,"
                    "source_url"
                )
                .eq(
                    "legal_domain",
                    "occupational_safety",
                )
                .or_(query)
                .limit(
                    TARGETED_SEARCH_LIMIT
                )
                .execute()
            )

            results.extend(
                response.data or []
            )

        except Exception as exc:

            logger.warning(
                "RAG | targeted occupational training search failed: %s",
                exc,
            )

    return _deduplicate_chunks(results)


def _targeted_occupational_briefing_search(
    supabase,
    user_query: str = "",
) -> List[Dict[str, Any]]:
    """
    Точечный поиск по инструктажам.

    КРИТИЧЕСКОЕ ПРАВИЛО:
    «какой/какому инструктаж» и «кто проводит инструктаж» —
    разные поисковые задачи. Никогда не смешиваем их в одном
    targeted search.

    TARGET:
        ищем вид инструктажа, разовые работы, прямые обязанности,
        наряд-допуск и п. 29 Инструкции №175.

    RESPONSIBLE:
        ищем только лицо, проводящее инструктаж.
    """

    results: List[Dict[str, Any]] = []

    target_mode = _is_target_briefing_query(user_query)
    responsible_mode = (
        not target_mode
        and _is_responsible_briefing_query(user_query)
    )

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
        mode = "target"
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
        mode = "responsible"
    else:
        # Для неоднозначного запроса не подмешиваем ни вводный,
        # ни целевой инструктаж принудительно.
        queries = [
            "doc_name.ilike.%175%",
            "doc_name.ilike.%Инструкци%",
            "content.ilike.%инструктаж%",
        ]
        mode = "generic"

    logger.info(
        "RAG | targeted briefing search | mode=%s | query=%s",
        mode,
        user_query,
    )

    for query in queries:
        try:
            response = (
                supabase
                .table("npa_chunks")
                .select(
                    "doc_name,"
                    "doc_type,"
                    "point_num,"
                    "content,"
                    "legal_domain,"
                    "topic,"
                    "source_url"
                )
                .eq(
                    "legal_domain",
                    "occupational_safety",
                )
                .or_(query)
                .limit(
                    TARGETED_SEARCH_LIMIT
                )
                .execute()
            )

            results.extend(response.data or [])

        except Exception as exc:
            logger.warning(
                "RAG | targeted occupational briefing search failed | mode=%s | error=%s",
                mode,
                exc,
            )

    return _deduplicate_chunks(results)


def _targeted_accident_search(
    supabase,
) -> List[Dict[str, Any]]:

    results: List[Dict[str, Any]] = []

    queries = [
        # Название профильного НПА.
        "doc_name.ilike.%несчаст%",
        "doc_name.ilike.%расслед%",

        # Форма и акт Н-1 — самые сильные маркеры.
        "content.ilike.%Н-1%",
        "content.ilike.%н-1%",
        "content.ilike.%форма Н-1%",
        "content.ilike.%форма н-1%",
        "content.ilike.%акт Н-1%",
        "content.ilike.%акт н-1%",

        # Участники и действие, о котором спрашивают.
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

    for query in queries:

        try:

            response = (
                supabase
                .table("npa_chunks")
                .select(
                    "doc_name,"
                    "doc_type,"
                    "point_num,"
                    "content,"
                    "legal_domain,"
                    "topic,"
                    "source_url"
                )
                .eq(
                    "legal_domain",
                    "occupational_safety",
                )
                .or_(query)
                .limit(
                    TARGETED_SEARCH_LIMIT
                )
                .execute()
            )

            results.extend(
                response.data or []
            )

        except Exception as exc:

            logger.warning(
                "RAG | targeted accident search failed: %s",
                exc,
            )

    return _deduplicate_chunks(results)


def _deduplicate_chunks(
    chunks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:

    unique: List[Dict[str, Any]] = []
    seen = set()

    for chunk in chunks:

        document = _get_document_name(
            chunk
        )

        point = _get_point_number(
            chunk
        )

        content = str(
            chunk.get("content")
            or ""
        ).strip()

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

    if topic == "ppe_nonprovision":

        return await asyncio.to_thread(
            _targeted_ppe_nonprovision_search,
            supabase,
        )

    if topic == "medical_examinations":

        return await asyncio.to_thread(
            _targeted_medical_exam_search,
            supabase,
        )

    if topic == "workplace_attestation":

        return await asyncio.to_thread(
            _targeted_attestation_search,
            supabase,
        )

    if topic == "occupational_briefing":

        return await asyncio.to_thread(
            _targeted_occupational_briefing_search,
            supabase,
            user_query,
        )

    if topic == "accident_investigation":

        return await asyncio.to_thread(
            _targeted_accident_search,
            supabase,
        )

    if topic == "occupational_training":

        return await asyncio.to_thread(
            _targeted_occupational_training_search,
            supabase,
        )

    return []


# ============================================================
# SEMANTIC SCORE
# ============================================================

def _semantic_score(
    chunk: Dict[str, Any],
) -> float:

    if "similarity" in chunk:

        return _safe_float(
            chunk["similarity"]
        )

    if "score" in chunk:

        return _safe_float(
            chunk["score"]
        )

    if "distance" in chunk:

        distance = _safe_float(
            chunk["distance"]
        )

        return 1.0 - distance

    return 0.0


# ============================================================
# ИТОГОВОЕ РАНЖИРОВАНИЕ
# ============================================================

def _combined_score(
    chunk: Dict[str, Any],
    query_terms: List[str],
    topic: str,
) -> float:

    semantic = _semantic_score(
        chunk
    )

    keyword = _keyword_score(
        chunk,
        query_terms,
    )

    topic_score = _topic_relevance_score(
        chunk,
        topic,
    )

    # --------------------------------------------------------
    # БАЗОВАЯ ФОРМУЛА
    # --------------------------------------------------------
    #
    # semantic = 70%
    # keyword  = 15%
    # topic    = 15%
    #
    # Архитектура не меняется.
    # --------------------------------------------------------

    score = (
        semantic * 0.70
        + keyword * 0.15
        + min(topic_score, 1.0) * 0.15
    )

    # --------------------------------------------------------
    # СИЛЬНЫЙ BOOST ДЛЯ ПОСТАНОВЛЕНИЯ №253
    # --------------------------------------------------------

    if topic == "workplace_attestation":

        document_name = _get_document_name(
            chunk
        ).lower()

        content = str(
            chunk.get("content")
            or ""
        ).lower()

        if (
            "253" in document_name
            and (
                "аттестаци" in document_name
                or "аттестаци" in content
            )
        ):
            score += 0.20

    # --------------------------------------------------------
    # СИЛЬНЫЙ BOOST ДЛЯ НПА №74
    # --------------------------------------------------------

    if topic == "medical_examinations":

        document_name = _get_document_name(
            chunk
        ).lower()

        content = str(
            chunk.get("content")
            or ""
        ).lower()

        is_npa_74 = bool(
            re.search(
                r"№\s*74\b",
                document_name,
                flags=re.IGNORECASE,
            )
            or re.search(
                r"\b74\b",
                document_name,
                flags=re.IGNORECASE,
            )
        )

        is_medical_document = (
            "медицинск" in document_name
            and "осмотр" in document_name
        )

        if is_npa_74:
            score += 0.35

        if is_medical_document:
            score += 0.20

        financing_markers = [
            "за счет",
            "за счёт",
            "оплата",
            "расход",
            "средств",
            "финанс",
            "затрат",
        ]

        if any(
            marker in content
            for marker in financing_markers
        ):
            score += 0.15

    # --------------------------------------------------------
    # СИЛЬНЫЙ BOOST ДЛЯ НПА №209
    # --------------------------------------------------------

    if topic == "ppe_nonprovision":

        document_name = _get_document_name(
            chunk
        ).lower()

        content = str(
            chunk.get("content")
            or ""
        ).lower()

        is_npa_209 = bool(
            re.search(
                r"№\s*209\b",
                document_name,
                flags=re.IGNORECASE,
            )
            or re.search(
                r"\b209\b",
                document_name,
                flags=re.IGNORECASE,
            )
        )

        is_ppe_document = (
            "сиз" in document_name
            or (
                "средств" in document_name
                and "индивидуальн" in document_name
                and "защит" in document_name
            )
        )

        if is_npa_209:
            score += 0.35

        if is_ppe_document:
            score += 0.20

        # Самый важный бонус:
        # конкретные слова проблемной ситуации
        # должны поднимать фрагмент выше общих
        # положений о выдаче СИЗ.
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

        problem_matches = sum(
            1
            for marker in problem_markers
            if marker in content
        )

        if problem_matches:
            score += min(
                problem_matches * 0.10,
                0.30,
            )

    # --------------------------------------------------------
    # СИЛЬНЫЙ BOOST ДЛЯ СТАЖИРОВКИ / ИНСТРУКЦИИ №175
    # --------------------------------------------------------

    if topic == "occupational_training":

        document_name = _get_document_name(
            chunk
        ).lower()

        content = str(
            chunk.get("content")
            or ""
        ).lower()

        is_npa_175 = bool(
            re.search(
                r"№\s*175\b",
                document_name,
                flags=re.IGNORECASE,
            )
            or re.search(
                r"\b175\b",
                document_name,
                flags=re.IGNORECASE,
            )
        )

        has_internship = (
            "стажиров" in content
        )

        has_minimum_two = (
            "не менее двух" in content
        )

        has_working_days = (
            "рабочих дней" in content
            or "рабочих смен" in content
        )

        has_duration = (
            "продолжительн" in content
            or "срок" in content
        )

        has_hazardous_work = (
            "повышенной опасност" in content
            or "повышенную опасност" in content
        )

        has_independent_admission = (
            "самостоятельной работе" in content
            or "допуск к самостоятельной" in content
        )

        has_knowledge_test = (
            "проверка знаний" in content
            or "проверку знаний" in content
        )

        if is_npa_175:
            score += 0.40

        if has_internship:
            score += 0.20

        if has_minimum_two:
            score += 0.35

        if has_working_days:
            score += 0.25

        if has_duration:
            score += 0.20

        if has_hazardous_work:
            score += 0.15

        if has_independent_admission:
            score += 0.15

        if has_knowledge_test:
            score += 0.10

        # Точная количественная норма имеет максимальный
        # приоритет для вопроса о минимальной продолжительности.
        if (
            has_internship
            and has_minimum_two
            and has_working_days
        ):
            score += 0.25

    return score


def _sort_chunks(
    chunks: List[Dict[str, Any]],
    query_terms: List[str],
    topic: str,
) -> List[Dict[str, Any]]:

    for chunk in chunks:

        chunk["_combined_score"] = (
            _combined_score(
                chunk,
                query_terms,
                topic,
            )
        )

    return sorted(
        chunks,
        key=lambda chunk: chunk.get(
            "_combined_score",
            0.0,
        ),
        reverse=True,
    )


# ============================================================
# DIVERSIFICATION
# ============================================================

def _get_document_key(
    chunk: Dict[str, Any],
) -> str:

    return _get_document_name(
        chunk
    ).strip().lower()


def _diversify_chunks(
    chunks: List[Dict[str, Any]],
    limit: int,
    max_per_document: int = 4,
) -> List[Dict[str, Any]]:

    selected: List[Dict[str, Any]] = []

    per_document: Dict[str, int] = {}

    for chunk in chunks:

        document_key = _get_document_key(
            chunk
        )

        count = per_document.get(
            document_key,
            0,
        )

        if count >= max_per_document:
            continue

        selected.append(chunk)

        per_document[document_key] = (
            count + 1
        )

        if len(selected) >= limit:
            break

    if len(selected) < limit:

        selected_ids = {
            id(chunk)
            for chunk in selected
        }

        for chunk in chunks:

            if id(chunk) in selected_ids:
                continue

            selected.append(chunk)

            if len(selected) >= limit:
                break

    return selected


# ============================================================
# ПРОВЕРКА НУЖНОГО НПА
# ============================================================

def _is_occupational_training_document(
    chunk: Dict[str, Any],
) -> bool:
    """
    Определяет профильные фрагменты по стажировке.

    Не требует topic metadata: в базе связанные пункты
    одного НПА могут иметь разные значения topic.
    """

    document_name = _get_document_name(chunk).lower()
    content = str(chunk.get("content") or "").lower()

    is_npa_175 = bool(
        re.search(
            r"№\s*175\b",
            document_name,
            flags=re.IGNORECASE,
        )
        or (
            "инструкци" in document_name
            and re.search(
                r"\b175\b",
                document_name,
                flags=re.IGNORECASE,
            )
        )
    )

    has_internship = (
        "стажиров" in content
    )

    has_duration = (
        "продолжительн" in content
        or "срок" in content
    )

    has_minimum_two = (
        "не менее двух" in content
    )

    has_working_period = (
        "рабочих дней" in content
        or "рабочих смен" in content
    )

    return (
        is_npa_175
        or (
            has_internship
            and (
                has_minimum_two
                or has_working_period
                or has_duration
            )
        )
    )


def _is_accident_investigation_document(
    chunk: Dict[str, Any],
) -> bool:
    """
    Определяет, относится ли фрагмент к расследованию
    и учету несчастных случаев / оформлению акта Н-1.

    Намеренно не привязываемся к одному номеру НПА:
    база может содержать действующую редакцию под другим
    названием, а нужная норма может находиться в профильном
    постановлении или в его отдельных пунктах.
    """

    document_name = _get_document_name(chunk).lower()
    content = str(chunk.get("content") or "").lower()

    has_n1 = bool(
        re.search(
            r"\bн[-–—]?\s*1\b",
            document_name + " " + content,
            re.IGNORECASE,
        )
    )

    has_accident = (
        "несчаст" in document_name
        or "несчаст" in content
    )

    has_investigation = (
        "расследован" in document_name
        or "расследован" in content
    )

    has_victim = (
        "пострадавш" in content
        or "потерпевш" in content
        or "родственник" in content
    )

    return (
        has_n1
        or (has_accident and has_investigation)
        or (has_accident and has_victim)
    )


def _accident_marker_score(
    chunk: Dict[str, Any],
) -> float:
    """
    Сильный lexical boost для точечных вопросов по Н-1.
    """

    document_name = _get_document_name(chunk).lower()
    content = str(chunk.get("content") or "").lower()
    text = f"{document_name} {content}"

    score = 0.0

    if re.search(r"\bн[-–—]?\s*1\b", text, re.IGNORECASE):
        score += 0.80

    if "акт" in text and re.search(
        r"\bн[-–—]?\s*1\b",
        text,
        re.IGNORECASE,
    ):
        score += 0.40

    markers = [
        "пострадавш",
        "потерпевш",
        "родственник",
        "вруч",
        "утвержден",
        "утверждён",
        "рабочих дней",
        "рабочие дни",
        "окончани",
        "расследован",
    ]

    matches = sum(1 for marker in markers if marker in text)

    score += min(matches * 0.12, 0.72)

    return min(score, 1.80)


def _is_attestation_document(
    chunk: Dict[str, Any],
) -> bool:

    document_name = _get_document_name(
        chunk
    ).lower()

    content = str(
        chunk.get("content")
        or ""
    ).lower()

    return (
        "253" in document_name
        and (
            "аттестаци" in document_name
            or "аттестаци" in content
        )
    )


def _is_medical_exam_document(
    chunk: Dict[str, Any],
) -> bool:
    """
    Определяет, относится ли фрагмент
    к основному НПА по медицинским осмотрам.
    """

    document_name = _get_document_name(
        chunk
    ).lower()

    content = str(
        chunk.get("content")
        or ""
    ).lower()

    is_npa_74 = bool(
        re.search(
            r"№\s*74\b",
            document_name,
            flags=re.IGNORECASE,
        )
        or re.search(
            r"\b74\b",
            document_name,
            flags=re.IGNORECASE,
        )
    )

    is_medical_document = (
        "медицинск" in document_name
        and "осмотр" in document_name
    )

    return (
        is_npa_74
        or is_medical_document
    )


def _is_ppe_nonprovision_document(
    chunk: Dict[str, Any],
) -> bool:
    """
    Определяет, относится ли фрагмент
    к профильному документу по обеспечению СИЗ.

    Основной критерий — НПА №209.
    """

    document_name = _get_document_name(
        chunk
    ).lower()

    content = str(
        chunk.get("content")
        or ""
    ).lower()

    is_npa_209 = bool(
        re.search(
            r"№\s*209\b",
            document_name,
            flags=re.IGNORECASE,
        )
        or re.search(
            r"\b209\b",
            document_name,
            flags=re.IGNORECASE,
        )
    )

    is_ppe_document = (
        "сиз" in document_name
        or (
            "средств" in document_name
            and "индивидуальн" in document_name
            and "защит" in document_name
        )
    )

    # Для фрагментов с явными признаками ситуации
    # также допускаем профильный документ по СИЗ.
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

    return (
        is_npa_209
        or (
            is_ppe_document
            and has_problem_context
        )
    )


# ============================================================
# ОСНОВНОЙ RAG
# ====================================================
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
        if any(
            re.search(p, query, flags=re.IGNORECASE)
            for p in pats
        )
    ]


def detect_primary_intent(
    intents: List[str],
    user_query: str = "",
) -> Optional[str]:
    """
    Выделяет основной юридический интент.

    Для точечных вопросов основной интент получает
    повышенный вес при rerank и определяет приоритет
    финальной юридической диверсификации.
    """
    priority = [
        "responsible_person",
        "employer_duty",
        "employee_right",
        "refusal",
        "danger",
        "procedure",
        "liability",
    ]

    for intent in priority:
        if intent in intents:
            return intent

    return None


def _is_labor_code_query(user_query: str) -> bool:
    query = str(user_query or "").strip().lower()

    return bool(
        re.search(
            r"\bтрудов\w*\s+кодекс\w*",
            query,
            flags=re.IGNORECASE,
        )
        or re.search(
            r"\bтк\s*рб\b",
            query,
            flags=re.IGNORECASE,
        )
    )


def is_cross_reference_query(intents: List[str]) -> bool:
    return len(set(intents) & {
        "refusal", "danger", "employee_right",
        "employer_duty", "procedure", "liability",
    }) >= 2


def _is_target_briefing_query(user_query: str) -> bool:
    """
    True для вопросов, где пользователь спрашивает, КАКОЙ вид
    инструктажа требуется/проводится.

    Критически важно: такие вопросы нельзя превращать в запросы
    вида «кто проводит вводный инструктаж».
    """
    query = re.sub(
        r"\s+",
        " ",
        str(user_query or "").strip().lower(),
    )

    if not query:
        return False

    # Прямые формулировки: «какой инструктаж», «какой вид инструктажа»,
    # «какой инструктаж проводится» и т.п.
    target_patterns = [
        r"\bкакой\s+(?:вид\s+)?инструктаж\w*\b",
        r"\bкакой\s+(?:вид\s+)?инструктаж\w*\s+(?:нужен|необходим|провод\w*|требу\w*)",
        r"\bкакому\s+инструктаж\w*\b",
        r"\bк\s+какому\s+инструктаж\w*\b",
        r"\bвид\s+инструктаж\w*\b.*\bнужен\b",
        r"\bотнос\w*\s+к\s+(?:какому|какой)\s+инструктаж\w*",
    ]

    if any(
        re.search(pattern, query, flags=re.IGNORECASE)
        for pattern in target_patterns
    ):
        return True

    # Юридические признаки целевого инструктажа. Даже если пользователь
    # не написал «какой инструктаж», наличие этой связки означает,
    # что нам нужен поиск нормы о ВИДЕ инструктажа.
    target_context_patterns = [
        r"\bразов\w*\s+работ\w*.*\bне\s+связан\w*.*\bпрям\w*\s+обязанност\w*",
        r"\bне\s+связан\w*\s+с\s+прям\w*\s+обязанност\w*",
        r"\bпрям\w*\s+обязанност\w*.*\bразов\w*\s+работ\w*",
        r"\bнаряд\w*[-–—]?\s*допуск\w*",
        r"\bнаряд\w*\s+допуск\w*",
        r"\bцелев\w*\s+инструктаж\w*",
    ]

    return any(
        re.search(pattern, query, flags=re.IGNORECASE)
        for pattern in target_context_patterns
    )


def _is_responsible_briefing_query(user_query: str) -> bool:
    """
    True только для вопросов о лице/должностном лице,
    которое проводит инструктаж.

    Не срабатывает на «какой инструктаж».
    """
    query = re.sub(
        r"\s+",
        " ",
        str(user_query or "").strip().lower(),
    )

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

    return any(
        re.search(pattern, query, flags=re.IGNORECASE)
        for pattern in responsible_patterns
    )


def build_search_queries(
    user_query: str,
    topic: str,
    legal_domain: str,
    intents: List[str],
) -> List[str]:

    original = str(user_query or "").strip()
    queries: List[str] = [original]

    # --------------------------------------------------------
    # КРИТИЧЕСКОЕ РАЗДЕЛЕНИЕ «КАКОЙ ИНСТРУКТАЖ» / «КТО ПРОВОДИТ»
    # --------------------------------------------------------
    #
    # Эти два типа вопросов относятся к одной теме, но требуют
    # принципиально разных поисковых запросов.
    #
    # Раньше topic=occupational_briefing автоматически добавлял:
    #   «кто проводит вводный инструктаж...»
    # даже если пользователь спрашивал:
    #   «какой инструктаж...»
    #
    # Теперь:
    #   target  -> ищем вид/основание инструктажа;
    #   person  -> ищем лицо, проводящее инструктаж;
    #   ambiguous -> нейтральные запросы без навязывания ответа.
    # --------------------------------------------------------

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

    topic_queries = {
        "ppe_nonprovision": [
            f"СИЗ не выданы повреждены неисправны работник безопасность труда {original}",
            "право работника отказаться от выполнения работы при угрозе жизни и здоровью",
            "действия работника при возникновении опасности для жизни и здоровья",
            "обязанности нанимателя по обеспечению безопасных условий труда и средствами индивидуальной защиты",
            "неприступление к работе или отказ от выполнения опасной работы трудовое законодательство",
        ],

        # Для occupational_briefing формируем запросы НИЖЕ отдельно.
        # Здесь специально НЕТ запроса «кто проводит».
        "occupational_briefing": [],

        "accident_investigation": [
            f"несчастный случай расследование {original}",
            "порядок расследования несчастного случая на производстве права потерпевшего",
            "акт Н-1 утверждение вручение потерпевшему родственникам",
            "обязанности нанимателя после окончания расследования несчастного случая",
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

    if topic == "occupational_briefing":
        if target_briefing_query:
            # ТОЛЬКО поиск нормы о виде инструктажа.
            queries.extend([
                f"целевой инструктаж разовые работы не связанные с прямыми обязанностями {original}",
                "целевой инструктаж разовые работы не связанные с прямыми обязанностями",
                "разовые работы не связанные с прямыми обязанностями какой инструктаж",
                "целевой инструктаж наряд-допуск работы с повышенной опасностью",
                "Инструкция №175 пункт 29 целевой инструктаж разовые работы",
            ])
        elif responsible_briefing_query:
            # ТОЛЬКО поиск лица, проводящего инструктаж.
            queries.extend([
                f"вводный инструктаж по охране труда кто проводит {original}",
                "кто проводит вводный инструктаж по охране труда специалист по охране труда уполномоченное должностное лицо нанимателя",
                "вводный инструктаж проводит специалист по охране труда уполномоченное должностное лицо нанимателя",
                "вводный инструктаж руководитель организации специалист по охране труда",
            ])
        else:
            # Неопределённый запрос: не подставляем ни «кто»,
            # ни «какой». Это предотвращает ложное направление поиска.
            queries.extend([
                f"инструктаж по охране труда {original}",
                "виды инструктажей по охране труда порядок проведения",
                "Инструкция №175 инструктаж по охране труда",
            ])
    else:
        queries.extend(topic_queries.get(topic, []))

    primary_intent = detect_primary_intent(
        intents,
        original,
    )

    # Для occupational_briefing intent responsible_person имеет право
    # добавлять person-запросы ТОЛЬКО если это действительно вопрос
    # о лице, проводящем инструктаж.
    if (
        "responsible_person" in intents
        and not target_briefing_query
        and responsible_briefing_query
    ):
        # Уже добавлены профильные person-запросы выше.
        # Ничего дополнительно не добавляем, чтобы не плодить дубли.
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
        queries.append(
            f"право работника охрана труда безопасные условия труда {original}"
        )

    if "danger" in intents:
        queries.append(
            f"угроза жизни и здоровью работника опасность труд {original}"
        )

    if "procedure" in intents:
        queries.append(
            f"порядок действий работника при нарушении требований охраны труда {original}"
        )

    if _is_labor_code_query(original):
        queries.extend([
            "Трудовой кодекс Республики Беларусь охрана труда работник наниматель",
            "Трудовой кодекс Республики Беларусь безопасные условия труда",
        ])

    result: List[str] = []
    seen = set()

    for q in queries:
        qn = re.sub(
            r"\s+",
            " ",
            q,
        ).strip().lower()

        if qn and qn not in seen:
            seen.add(qn)
            result.append(q.strip())

    return result[:8]

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
    """
    Насколько фрагмент соответствует основному юридическому
    интенту запроса.
    """
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
        "occupational_briefing": (
            ("вводный инструктаж", 0.45),
            ("специалист по охране труда", 0.45),
            ("уполномоченное должностное лицо нанимателя", 0.45),
            ("на которое возложены обязанности специалиста по охране труда", 0.40),
            ("руководителем организации", 0.18),
            ("территориальной удаленности", 0.18),
            ("территориальной удаленностью", 0.18),
        ),
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
        "occupational_briefing": (
            ("целевой инструктаж", 0.60),
            ("разовых работ", 0.50),
            ("не связанных с прямыми обязанностями", 0.75),
            ("не связаны с прямыми обязанностями", 0.75),
            ("наряд-допуск", 0.45),
            ("наряду-допуску", 0.55),
            ("наряд допуск", 0.45),
        ),
    }
    for phrase, weight in phrase_weights.get(topic, ()):
        if phrase in text:
            score += weight
    if topic in ("occupational_briefing", "occupational_training") and "175" in document_name:
        score += 0.08
    return min(score, 1.80)


def _legal_relevance_score(
    chunk: Dict[str, Any],
    query_terms: List[str],
    topic: str,
    intents: List[str],
    cross_reference: bool,
    primary_intent: Optional[str] = None,
    labor_code_query: bool = False,
    user_query: str = "",
) -> float:
    semantic = _safe_float(chunk.get("_best_similarity", _semantic_score(chunk)))
    hybrid_score = _safe_float(chunk.get("_hybrid_final_score"))
    keyword = _keyword_score(chunk, query_terms)
    exact_score = _exact_match_score(chunk, user_query, topic)
    # Сохраняем компонент для диагностики TOP-N и последующего тюнинга.
    chunk["_exact_score"] = exact_score
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
    return (
        semantic * 0.30
        + hybrid_score * 0.12
        + exact_score * 0.28
        + keyword * keyword_weight
        + topic_score * topic_weight
        + intent_score * 0.10
        + primary_score * 0.20
        + labor_code_bonus
        + repeated_bonus
    )

def _legal_chunk_role(
    chunk: Dict[str, Any],
    topic: str,
    intents: List[str],
    primary_intent: Optional[str] = None,
) -> str:

    document = str(
        chunk.get("doc_name")
        or chunk.get("document")
        or ""
    ).lower()

    content = str(
        chunk.get("content")
        or chunk.get("text")
        or ""
    ).lower()

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

    # Для целевого интента проверяем соответствующую роль первой.
    # Это не даёт статье о правах работника быть ошибочно
    # классифицированной как ответ на вопрос об обязанностях нанимателя.
    ordered_checks = []

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


def _select_legal_diverse_chunks(
    ranked_chunks: List[Dict[str, Any]],
    limit: int,
    topic: str,
    intents: List[str],
    cross_reference: bool,
    primary_intent: Optional[str] = None,
    labor_code_query: bool = False,
) -> List[Dict[str, Any]]:

    if not ranked_chunks or limit <= 0:
        return []

    selected: List[Dict[str, Any]] = []
    selected_keys = set()
    roles_seen = set()
    documents_seen: Dict[str, int] = {}

    def _key(chunk: Dict[str, Any]):
        return (
            _get_document_key(chunk),
            _get_point_number(chunk).lower(),
            str(chunk.get("content") or "")[:200].lower(),
        )

    def _add(chunk: Dict[str, Any], max_per_document: int = 4) -> bool:
        key = _key(chunk)
        document_key = _get_document_key(chunk)

        if key in selected_keys:
            return False

        if documents_seen.get(document_key, 0) >= max_per_document:
            return False

        selected.append(chunk)
        selected_keys.add(key)
        documents_seen[document_key] = documents_seen.get(document_key, 0) + 1
        return True

    # Для точечного вопроса сначала берём именно тот тип нормы,
    # который пользователь запросил.
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
                # Вопрос прямо про ТК РБ: приоритет отдаём
                # обязанностям нанимателя из ТК.
                document_name = _get_document_name(chunk).lower()
                is_labor_code = (
                    "трудовой кодекс" in document_name
                    or "трудовои кодекс" in document_name
                    or "трудовой_кодекс" in document_name
                )
                if not is_labor_code:
                    continue

            _add(
                chunk,
                max_per_document=3,
            )

            if len(selected) >= min(2, limit):
                break

    # Для ТК-запроса добавляем ещё один релевантный фрагмент ТК,
    # даже если его роль secondary/general.
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

        # Для сложных вопросов сохраняем юридическое разнообразие,
        # но не вытесняем основной интент.
        for chunk in ranked_chunks:
            role = _legal_chunk_role(
                chunk,
                topic,
                intents,
                primary_intent=primary_intent,
            )
            document_key = _get_document_key(chunk)

            if role in roles_seen:
                continue

            if _add(chunk, max_per_document=max_per_document):
                roles_seen.add(role)

            if len(selected) >= limit:
                return selected

    max_per_document = 3 if primary_intent else (2 if cross_reference else 4)

    for chunk in ranked_chunks:
        _add(
            chunk,
            max_per_document=max_per_document,
        )

        if len(selected) >= limit:
            return selected

    # Финальный fallback без ограничений по документу.
    for chunk in ranked_chunks:
        key = _key(chunk)
        if key in selected_keys:
            continue

        selected.append(chunk)
        selected_keys.add(key)

        if len(selected) >= limit:
            break

    return selected


async def retrieve_context(
    user_query: str,
    supabase,
) -> Dict[str, Any]:

    logger.info(
        "RAG | query=%s",
        user_query,
    )

    legal_domain = detect_legal_domain(
        user_query
    )

    topic = detect_topic(
        user_query
    )

    query_terms = _extract_query_terms(
        user_query
    )

    intents = detect_query_intents(
        user_query
    )

    cross_reference = is_cross_reference_query(
        intents
    )

    primary_intent = detect_primary_intent(
        intents,
        user_query,
    )

    labor_code_query = _is_labor_code_query(
        user_query
    )

    search_queries = build_search_queries(
        user_query,
        topic,
        legal_domain,
        intents,
    )

    logger.info(
        "RAG | legal_domain=%s | topic=%s | intents=%s | cross_reference=%s",
        legal_domain,
        topic,
        intents,
        cross_reference,
    )

    logger.info(
        "RAG | search_queries=%s",
        search_queries,
    )

    # ========================================================
    # EMBEDDINGS
    # ========================================================
    #
    # Embeddings создаём последовательно: локальная модель
    # может быть CPU/RAM-ограниченной. Сам поиск Supabase
    # ниже выполняется параллельно.
    # ========================================================

    query_vectors = await asyncio.to_thread(
        get_query_embeddings,
        search_queries,
    )

    valid_pairs = [
        (search_query, vector)
        for search_query, vector in zip(search_queries, query_vectors)
        if vector
    ]

    valid_queries = [
        search_query
        for search_query, _ in valid_pairs
    ]

    query_vectors = [
        vector
        for _, vector in valid_pairs
    ]

    for vector in query_vectors:
        if len(vector) != 384:
            raise ValueError(
                "Unexpected embedding dimension: "
                f"{len(vector)}. Expected 384."
            )

    if not query_vectors:

        logger.warning(
            "RAG | all embeddings are empty"
        )

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

    # ========================================================
    # MULTI-QUERY VECTOR SEARCH
    # ========================================================
    #
    # ВАЖНО:
    # topic_filter здесь намеренно НЕ используется.
    #
    # Например, вопрос про отсутствие/повреждение СИЗ может
    # одновременно требовать:
    #   - Правила обеспечения СИЗ;
    #   - общую норму о праве работника;
    #   - норму о действиях при опасности;
    #   - обязанность нанимателя.
    #
    # Жёсткий topic filter способен физически удалить
    # юридически применимую норму до rerank.
    # ========================================================

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
            remove_domain_filter=(
                cross_reference and index == 0
            ),
        )
        for index, (vector, search_query)
        in enumerate(
            zip(query_vectors, valid_queries)
        )
    ]

    search_groups = await asyncio.gather(
        *search_tasks,
        return_exceptions=True,
    )

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

        clean_groups.append(
            result or []
        )

        logger.info(
            "RAG | search[%s] candidates=%s | query=%s",
            index,
            len(result or []),
            valid_queries[index],
        )
    query_roles = [
        "main" if index == 0 else "expanded"
        for index in range(len(clean_groups))
    ]

    candidate_chunks = _merge_search_results(
        clean_groups,
        query_roles,
    )

    # ========================================================
    # TARGETED LEGAL SEARCH
    # ========================================================
    #
    # Сохраняем существующий точечный поиск для №209, №253,
    # №74, №175 и Н-1, но теперь он лишь добавляет кандидатов,
    # а не определяет весь ответ.
    # ========================================================

    targeted_chunks = await _get_targeted_chunks(
        supabase,
        topic,
        user_query,
    )

    if targeted_chunks:

        logger.info(
            "RAG | targeted search | topic=%s | found=%s",
            topic,
            len(targeted_chunks),
        )

        targeted_merged = _merge_search_results(
            [targeted_chunks],
            ["targeted"],
        )

        candidate_chunks = _merge_search_results(
            [candidate_chunks, targeted_merged],
            ["merged", "targeted"],
        )

    else:

        logger.info(
            "RAG | targeted search | topic=%s | found=0",
            topic,
        )

    candidate_count = len(
        candidate_chunks
    )

    logger.info(
        "RAG | total candidates after multi-query merge=%s",
        candidate_count,
    )

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

    # ========================================================
    # СТАТИСТИКА
    # ========================================================

    domain_specific_count = sum(
        1
        for chunk in candidate_chunks
        if chunk.get("legal_domain")
        == legal_domain
    )

    topic_specific_count = sum(
        1
        for chunk in candidate_chunks
        if chunk.get("topic")
        == topic
    )

    logger.info(
        "RAG | domain=%s | domain_specific=%s | topic=%s | topic_specific=%s",
        legal_domain,
        domain_specific_count,
        topic,
        topic_specific_count,
    )

    # ========================================================
    # ДИАГНОСТИКА СПЕЦИАЛИЗИРОВАННЫХ ТЕМ
    # ========================================================

    if topic == "workplace_attestation":

        attestation_chunks = [
            chunk
            for chunk in candidate_chunks
            if _is_attestation_document(chunk)
        ]

        logger.info(
            "RAG | workplace_attestation | NPA_253_candidates=%s",
            len(attestation_chunks),
        )

    if topic == "medical_examinations":

        medical_chunks = [
            chunk
            for chunk in candidate_chunks
            if _is_medical_exam_document(chunk)
        ]

        logger.info(
            "RAG | medical_examinations | medical_candidates=%s",
            len(medical_chunks),
        )

    if topic == "ppe_nonprovision":

        ppe_chunks = [
            chunk
            for chunk in candidate_chunks
            if _is_ppe_nonprovision_document(chunk)
        ]

        logger.info(
            "RAG | ppe_nonprovision | NPA_209_candidates=%s",
            len(ppe_chunks),
        )

    if topic == "accident_investigation":

        accident_chunks = [
            chunk
            for chunk in candidate_chunks
            if _is_accident_investigation_document(chunk)
        ]

        logger.info(
            "RAG | accident_investigation | candidates=%s",
            len(accident_chunks),
        )

    if topic == "occupational_training":

        training_chunks = [
            chunk
            for chunk in candidate_chunks
            if _is_occupational_training_document(chunk)
        ]

        logger.info(
            "RAG | occupational_training | candidates=%s",
            len(training_chunks),
        )

    # ========================================================
    # ЮРИДИЧЕСКОЕ РАНЖИРОВАНИЕ
    # ========================================================

    for chunk in candidate_chunks:

        chunk["_combined_score"] = _legal_relevance_score(
            chunk,
            query_terms,
            topic,
            intents,
            cross_reference,
            primary_intent=primary_intent,
            labor_code_query=labor_code_query,
            user_query=user_query,
        )

    ranked_chunks = sorted(
        candidate_chunks,
        key=lambda chunk: chunk.get(
            "_combined_score",
            0.0,
        ),
        reverse=True,
    )

    # ========================================================
    # ФИНАЛЬНЫЙ КОНТЕКСТ
    # ========================================================
    #
    # Для сложных вопросов увеличиваем окно с 5 до 7 фрагментов.
    # Это даёт модели шанс увидеть и специальную норму, и общую
    # норму о праве/обязанности/опасности.
    # ========================================================

    final_limit = (
        max(RAG_FINAL_COUNT, 7)
        if cross_reference
        else RAG_FINAL_COUNT
    )

    final_chunks = _select_legal_diverse_chunks(
        ranked_chunks,
        final_limit,
        topic,
        intents,
        cross_reference,
        primary_intent=primary_intent,
        labor_code_query=labor_code_query,
    )

    # ========================================================
    # SOURCES
    # ========================================================

    source_references = []

    for index, chunk in enumerate(
        final_chunks,
        start=1,
    ):

        source_id = build_source_id(
            chunk,
            index,
        )

        chunk["_source_id"] = source_id

        document_name = _get_document_name(
            chunk
        )

        point = _get_point_number(
            chunk
        )

        if point:
            reference = (
                f"{document_name} — "
                f"пункт/статья {point}"
            )
        else:
            reference = document_name

        source_references.append(
            {
                "source_id": source_id,
                "reference": reference,
            }
        )

        logger.info(
            "RAG | TOP %s | role=%s | "
            "domain=%s | topic=%s | source=%s | "
            "semantic=%.4f | best_similarity=%.4f | "
            "hits=%s | combined=%.4f",
            index,
            _legal_chunk_role(
                chunk,
                topic,
                intents,
                primary_intent=primary_intent,
            ),
            chunk.get("legal_domain"),
            chunk.get("topic"),
            source_id,
            _semantic_score(chunk),
            _safe_float(
                chunk.get(
                    "_best_similarity",
                    _semantic_score(chunk),
                )
            ),
            chunk.get("_search_hits", 1),
            chunk.get(
                "_combined_score",
                0.0,
            ),
        )

        logger.info(
            "RAG | TOP %s | doc=%s | point=%s",
            index,
            document_name,
            point,
        )

    # ========================================================
    # CONTEXT
    # ========================================================

    retrieved_text = _build_retrieved_text(
        final_chunks
    )

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

        document_name = _get_document_name(
            chunk
        )

        point = _get_point_number(
            chunk
        )

        if point:

            reference = (
                f"{document_name} — "
                f"пункт/статья {point}"
            )

        else:

            reference = document_name

        if reference not in seen:

            seen.add(reference)

            references.append(
                reference
            )

    return references


def get_source_names(
    chunks: List[Dict[str, Any]],
) -> List[str]:

    names = []
    seen = set()

    for chunk in chunks:

        document_name = _get_document_name(
            chunk
        )

        if document_name not in seen:

            seen.add(document_name)

            names.append(
                document_name
            )

    return names
