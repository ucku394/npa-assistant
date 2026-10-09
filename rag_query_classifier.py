"""Классификация юридических запросов для RAG.

Шаг 1 безопасного рефакторинга: логика классификации вынесена из
монолитного rag.py без изменения алгоритмов.
"""

import re
from typing import List, Optional

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
        r"\bмолок\w*.*\bвредност\w*",
        r"\bравноценн\w*\s+пищев\w*\s+продукт\w*",
        r"\bвредн\w*\s+веществ\w*.*\bмолок\w*",
        r"\bмолок\w*.*\bденежн\w*\s+компенсац\w*",
        r"\bмедицинск\w*\s+осмотр\w*",
        r"\bмедосмотр\w*",
        r"\bработ\w*\s+на\s+высот\w*",
        r"\bпогрузочно-разгрузочн\w*",
        r"\bбывш\w*\s+в\s+употреблен\w*",
        r"\bпериод\w*\s+использован\w*",
        r"\bсрок\w*\s+носк\w*",
        r"\bаттестаци\w*\s+рабоч\w*\s+мест\w*",
        r"\bуслов\w*\s+труд\w*",
        r"\bпереносн\w*\s+лестниц\w*",
        r"\bприставн\w*\s+лестниц\w*",
        r"\bлестниц\w*[-–—]?стремянк\w*",
        r"\bстремянк\w*",
        # Metalworking machines, including lathes, are occupational-safety queries.
        r"\bтокарн\w*",
        r"\bметаллообрабатывающ\w*\s+оборудован\w*",
        r"\bработ\w*\s+на\s+станк\w*",
        # Conveyor and continuous-transport equipment safety belongs to OHS.
        r"\bконвейер\w*",
        r"\bленточн\w*\s+конвейер\w*",
        r"\bтранспортирующ\w*\s+устройств\w*",
        r"\bтранспортн\w*\s+средств\w*\s+непрерывн\w*",
        r"\bаварийн\w*\s+останов\w*",
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

    # Milk/equivalent food products are a distinct Belarus OHS benefit topic.
    milk_provision_patterns = [
        r"\bмолок\w*",
        r"\bравноценн\w*\s+пищев\w*\s+продукт\w*",
        r"\bденежн\w*\s+компенсац\w*.*\bмолок\w*",
        r"\bмолок\w*.*\bденежн\w*\s+компенсац\w*",
        r"\bзамен\w*.*\bмолок\w*",
        r"\bмолок\w*.*\bвредност\w*",
    ]
    if any(re.search(pattern, query, flags=re.IGNORECASE) for pattern in milk_provision_patterns):
        return "milk_provision"

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
        # Общий алгоритм/порядок инструктажа также относится к теме
        # occupational_briefing, даже если в запросе не назван конкретный
        # вид инструктажа.
        r"\bалгоритм\w*.*\bинструктаж\w*",
        r"\bпорядок\w*.*\bинструктаж\w*",
        r"\bпошагов\w*.*\bинструктаж\w*",
        r"\bкак\s+(?:провести|организовать|осуществить)\b.*\bинструктаж\w*",
        r"\bинструктаж\w*.*\bработник\w*",
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

    # Отдельный режим для переносных/приставных лестниц и стремянок.
    # В действующих Правилах № 11 ключевое требование п. 54 — ОСМОТР,
    # а не автоматическое периодическое статическое ИСПЫТАНИЕ.
    portable_ladder_patterns = [
        r"\bпереносн\w*\s+лестниц\w*",
        r"\bприставн\w*\s+лестниц\w*",
        r"\bлестниц\w*[-–—]?стремянк\w*",
        r"\bстремянк\w*",
        r"\bлестниц\w*.*\bиспытан\w*",
        r"\bиспытан\w*.*\bлестниц\w*",
        r"\bлестниц\w*.*\bосмотр\w*",
        r"\bосмотр\w*.*\bлестниц\w*",
    ]
    if any(re.search(pattern, query, flags=re.IGNORECASE) for pattern in portable_ladder_patterns):
        return "portable_ladder"

    # Специальный режим для работ на высоте: вопрос о 1 группе должен
    # приоритетно искать нормы Правил № 11, а не только общую Инструкцию № 175.
    height_work_training_patterns = [
        r"\bработ\w*\s+на\s+высот\w*",
        r"\bработающ\w*\s+1\s+групп\w*",
        r"\b1[-–—]?й\s+групп\w*.*\bвысот\w*",
        r"\bпервая\s+групп\w*.*\bвысот\w*",
        r"\bпостановлен\w*\s+№\s*11\b.*\bвысот\w*",
        r"\bобучен\w*.*\bработ\w*\s+на\s+высот\w*",
    ]
    if any(re.search(pattern, query, flags=re.IGNORECASE) for pattern in height_work_training_patterns):
        return "height_work_training"

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

    # Dedicated retrieval mode for lathe and cold metalworking safety.
    lathe_work_patterns = [
        r"\bтокарн\w*",
        r"\bработ\w*\s+на\s+металлообрабатывающ\w*\s+станк\w*",
        r"\bметаллообрабатывающ\w*\s+оборудован\w*",
        r"\bхолодн\w*\s+обработк\w*\s+металл\w*",
    ]
    if any(re.search(pattern, query, flags=re.IGNORECASE) for pattern in lathe_work_patterns):
        return "lathe_work"

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

    scope_patterns = [
        r"\bна\s+кого\s+(?:распространяется|распространяются|действует|действуют|применяется|применяются)\b",
        r"\bк\s+кому\s+(?:применяется|относится)\b",
        r"\bсфера\s+действия\b",
        r"\bобласть\s+действия\b",
        r"\bв\s+отношении\s+кого\b",
    ]
    if any(re.search(pattern, query, flags=re.IGNORECASE) for pattern in scope_patterns):
        return "law_scope"

    return "general"


def detect_scope_target(user_query: str) -> Optional[dict]:
    """Извлекает явно названный НПА для вопросов о сфере действия."""
    query = re.sub(r"\s+", " ", str(user_query or "").strip().lower())
    if not query:
        return None

    if re.search(
        r"\b356[-–—]з\b|\b356\s*[-–—]\s*з\b|закона?\s+.{0,80}об\s+охране\s+труда",
        query,
        re.IGNORECASE,
    ):
        return {
            "document_key": "356-з",
            "document_name": "Закон Республики Беларусь «Об охране труда»",
            "article": "3",
        }

    if re.search(r"\bтрудов(?:ого|ым)\s+кодекс\w*\b|\bтрудовой\s+кодекс\b", query, re.IGNORECASE):
        return {
            "document_key": "трудовой кодекс",
            "document_name": "Трудовой кодекс Республики Беларусь",
            "article": None,
        }

    # Общие «Правила по охране труда» — постановление Минтруда № 53.
    # Не сопоставляем сюда специальные правила вида
    # «Правила по охране труда при выполнении...».
    rules_match = re.search(r"\bправила\s+по\s+охране\s+труда\b", query, re.IGNORECASE)
    if rules_match:
        tail = query[rules_match.end():]
        if not re.match(r"\s+при\b", tail, re.IGNORECASE):
            return {
                "document_key": "правила по охране труда 53",
                "document_name": "Правила по охране труда, постановление Минтруда № 53",
                "article": "2",
            }

    return None


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
