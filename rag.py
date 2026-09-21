import asyncio
import hashlib
import logging
import os
import re
from typing import Any, Dict, List, Optional

from embedding import get_query_embedding

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
    legal_domain: str,
    topic_filter: Optional[str],
) -> List[Dict[str, Any]]:

    response = (
        supabase
        .rpc(
            "match_npa_chunks_v3",
            {
                "match_count": RAG_CANDIDATE_COUNT,
                "match_threshold": 0.0,
                "query_embedding": query_vector,
                "legal_domain_filter": legal_domain,
                "topic_filter": topic_filter,
            },
        )
        .execute()
    )

    return response.data or []


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

    if topic == "accident_investigation":

        return await asyncio.to_thread(
            _targeted_accident_search,
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
# ============================================================

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

    topic_filter = _get_topic_filter(
        topic
    )

    query_terms = _extract_query_terms(
        user_query
    )

    logger.info(
        "RAG | legal_domain=%s | topic=%s | topic_filter=%s | terms=%s",
        legal_domain,
        topic,
        topic_filter,
        query_terms,
    )

    # ========================================================
    # EMBEDDING
    # ========================================================

    query_vector = await asyncio.to_thread(
        get_query_embedding,
        user_query,
    )

    if not query_vector:

        logger.warning(
            "RAG | embedding is empty"
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
            "domain_specific_count": 0,
            "topic_specific_count": 0,
        }

    if len(query_vector) != 384:

        raise ValueError(
            "Unexpected embedding dimension: "
            f"{len(query_vector)}. Expected 384."
        )

    # ========================================================
    # PRIMARY VECTOR SEARCH
    # ========================================================

    candidate_chunks = await asyncio.to_thread(
        _search_chunks,
        supabase,
        query_vector,
        legal_domain,
        topic_filter,
    )

    logger.info(
        "RAG | primary candidates=%s",
        len(candidate_chunks),
    )

    # ========================================================
    # FALLBACK БЕЗ TOPIC
    # ========================================================

    if (
        topic_filter is not None
        and len(candidate_chunks)
        < RAG_MIN_CANDIDATES
    ):

        logger.info(
            "RAG | topic search insufficient, "
            "retrying without topic filter"
        )

        fallback_chunks = await asyncio.to_thread(
            _search_chunks,
            supabase,
            query_vector,
            legal_domain,
            None,
        )

        if len(fallback_chunks) > len(
            candidate_chunks
        ):

            candidate_chunks = fallback_chunks

    # ========================================================
    # TARGETED LEGAL SEARCH
    # ========================================================

    targeted_chunks = await _get_targeted_chunks(
        supabase,
        topic,
    )

    if targeted_chunks:

        logger.info(
            "RAG | targeted search | topic=%s | found=%s",
            topic,
            len(targeted_chunks),
        )

        candidate_chunks = _deduplicate_chunks(
            candidate_chunks
            + targeted_chunks
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
        "RAG | total candidates after merge=%s",
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
        "RAG | domain=%s | specialized=%s | topic=%s | topic_specific=%s",
        legal_domain,
        domain_specific_count,
        topic,
        topic_specific_count,
    )

    # ========================================================
    # СПЕЦИАЛЬНАЯ ПРОВЕРКА АТТЕСТАЦИИ
    # ========================================================

    if topic == "workplace_attestation":

        attestation_chunks = [
            chunk
            for chunk in candidate_chunks
            if _is_attestation_document(
                chunk
            )
        ]

        logger.info(
            "RAG | workplace_attestation | "
            "NPA_253_candidates=%s",
            len(attestation_chunks),
        )

        for chunk in attestation_chunks[:10]:

            logger.info(
                "RAG | NPA_253 | doc=%s | point=%s",
                _get_document_name(chunk),
                _get_point_number(chunk),
            )

    # ========================================================
    # СПЕЦИАЛЬНАЯ ПРОВЕРКА МЕДИЦИНСКИХ ОСМОТРОВ
    # ========================================================

    if topic == "medical_examinations":

        medical_chunks = [
            chunk
            for chunk in candidate_chunks
            if _is_medical_exam_document(
                chunk
            )
        ]

        logger.info(
            "RAG | medical_examinations | "
            "medical_candidates=%s",
            len(medical_chunks),
        )

        for chunk in medical_chunks[:10]:

            logger.info(
                "RAG | MEDICAL | doc=%s | point=%s",
                _get_document_name(chunk),
                _get_point_number(chunk),
            )

    # ========================================================
    # СПЕЦИАЛЬНАЯ ПРОВЕРКА СИЗ
    # ========================================================

    if topic == "ppe_nonprovision":

        ppe_chunks = [
            chunk
            for chunk in candidate_chunks
            if _is_ppe_nonprovision_document(
                chunk
            )
        ]

        logger.info(
            "RAG | ppe_nonprovision | "
            "NPA_209_candidates=%s",
            len(ppe_chunks),
        )

        for chunk in ppe_chunks[:15]:

            content = str(
                chunk.get("content")
                or ""
            ).lower()

            problem_markers = [
                marker
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
                if marker in content
            ]

            logger.info(
                "RAG | PPE | doc=%s | point=%s | problem_markers=%s",
                _get_document_name(chunk),
                _get_point_number(chunk),
                problem_markers,
            )

    # ========================================================
    # СПЕЦИАЛЬНАЯ ПРОВЕРКА НЕСЧАСТНЫХ СЛУЧАЕВ / Н-1
    # ========================================================

    if topic == "accident_investigation":

        accident_chunks = [
            chunk
            for chunk in candidate_chunks
            if _is_accident_investigation_document(chunk)
        ]

        logger.info(
            "RAG | accident_investigation | accident_candidates=%s",
            len(accident_chunks),
        )

        for chunk in accident_chunks[:15]:
            logger.info(
                "RAG | ACCIDENT | doc=%s | point=%s | marker_score=%.3f",
                _get_document_name(chunk),
                _get_point_number(chunk),
                _accident_marker_score(chunk),
            )

    # ========================================================
    # РАНЖИРОВАНИЕ
    # ========================================================

    ranked_chunks = _sort_chunks(
        candidate_chunks,
        query_terms,
        topic,
    )

    # Для вопросов по Н-1 после общего ранжирования
    # дополнительно учитываем точные юридические маркеры.
    # Это не заменяет semantic search, а исправляет ситуацию,
    # когда ТК/КоАП семантически похожи, но не содержат ответа.
    if topic == "accident_investigation":

        for chunk in ranked_chunks:
            marker_score = _accident_marker_score(chunk)
            chunk["_combined_score"] = (
                _safe_float(chunk.get("_combined_score"))
                + marker_score * 0.35
            )

        ranked_chunks = sorted(
            ranked_chunks,
            key=lambda chunk: chunk.get("_combined_score", 0.0),
            reverse=True,
        )

    # ========================================================
    # ДЛЯ СПЕЦИАЛИЗИРОВАННЫХ ТЕМ
    # НЕ ДАЁМ ПРОФИЛЬНОМУ НПА ПОТЕРЯТЬСЯ
    # ========================================================

    # --------------------------------------------------------
    # АТТЕСТАЦИЯ
    # --------------------------------------------------------

    if topic == "workplace_attestation":

        attestation_ranked = [
            chunk
            for chunk in ranked_chunks
            if _is_attestation_document(
                chunk
            )
        ]

        if attestation_ranked:

            required = attestation_ranked[:2]

            remaining = [
                chunk
                for chunk in ranked_chunks
                if chunk not in required
            ]

            ranked_chunks = (
                required
                + remaining
            )

            logger.info(
                "RAG | workplace_attestation | "
                "forced NPA_253 chunks=%s",
                len(required),
            )

    # --------------------------------------------------------
    # МЕДИЦИНСКИЕ ОСМОТРЫ
    # --------------------------------------------------------

    if topic == "medical_examinations":

        medical_ranked = [
            chunk
            for chunk in ranked_chunks
            if _is_medical_exam_document(
                chunk
            )
        ]

        if medical_ranked:

            # Для вопроса о медосмотрах минимум
            # три фрагмента профильного НПА №74
            # получают приоритет.
            required = medical_ranked[:3]

            remaining = [
                chunk
                for chunk in ranked_chunks
                if chunk not in required
            ]

            ranked_chunks = (
                required
                + remaining
            )

            logger.info(
                "RAG | medical_examinations | "
                "forced medical chunks=%s",
                len(required),
            )

    # --------------------------------------------------------
    # СИЗ — НЕВЫДАЧА / ПОВРЕЖДЕНИЕ / ОТКАЗ
    # --------------------------------------------------------

    if topic == "ppe_nonprovision":

        ppe_ranked = [
            chunk
            for chunk in ranked_chunks
            if _is_ppe_nonprovision_document(
                chunk
            )
        ]

        if ppe_ranked:

            # Минимум три фрагмента профильного
            # документа по СИЗ получают приоритет.
            #
            # Это особенно важно для вопросов,
            # где общий embedding поднимает пункты
            # о выдаче СИЗ, но пропускает норму
            # о действиях работника.
            required = ppe_ranked[:3]

            remaining = [
                chunk
                for chunk in ranked_chunks
                if chunk not in required
            ]

            ranked_chunks = (
                required
                + remaining
            )

            logger.info(
                "RAG | ppe_nonprovision | "
                "forced NPA_209 chunks=%s",
                len(required),
            )

    # --------------------------------------------------------
    # НЕСЧАСТНЫЕ СЛУЧАИ / Н-1
    # --------------------------------------------------------

    if topic == "accident_investigation":

        accident_ranked = [
            chunk
            for chunk in ranked_chunks
            if _is_accident_investigation_document(chunk)
        ]

        if accident_ranked:

            # Для точечного вопроса по акту Н-1 сначала даём
            # профильные фрагменты, содержащие сам Н-1/нормы
            # о расследовании и вручении акта.
            required = accident_ranked[:3]

            remaining = [
                chunk
                for chunk in ranked_chunks
                if chunk not in required
            ]

            ranked_chunks = required + remaining

            logger.info(
                "RAG | accident_investigation | forced accident chunks=%s",
                len(required),
            )

    # ========================================================
    # DIVERSIFICATION
    # ========================================================

    final_chunks = _diversify_chunks(
        ranked_chunks,
        RAG_FINAL_COUNT,
        max_per_document=4,
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
            "RAG | TOP %s | "
            "domain=%s | "
            "topic=%s | "
            "source=%s | "
            "semantic=%.4f | "
            "combined=%.4f",
            index,
            chunk.get("legal_domain"),
            chunk.get("topic"),
            source_id,
            _semantic_score(chunk),
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
