"""Иерархический Topic Planner 2.1 для юридического RAG.

Уровни:
    domain -> topic -> subtopic -> legal_object

Правила детерминированные и объяснимые. Модуль не заменяет legal relevance:
он определяет смысловой маршрут поиска, а не юридическую достаточность доказательства.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple


_RULES: Dict[str, Dict[str, List[Tuple[str, int]]]] = {
    "industrial_safety": {
        "general": [
            (r"\bопасн\w*\s+производственн\w*\s+объект\w*", 8),
            (r"\bпоо\b|\bопо\b", 8),
            (r"\bпромышленн\w*\s+безопасност\w*", 7),
        ],
        "production_control": [
            (r"\bпроизводственн\w*\s+контрол\w*", 10),
        ],
        "expertise": [
            (r"\bэкспертиз\w*\s+промышленн\w*\s+безопасност\w*", 10),
        ],
        "registration": [
            (r"\bрегистрац\w*\s+опасн\w*\s+производственн\w*\s+объект\w*", 10),
            (r"\bзарегистрир\w*\s+.*\bопо\b", 9),
        ],
        "accident": [
            (r"\bавари\w*\s+на\s+опасн\w*\s+производственн\w*\s+объект\w*", 10),
        ],
    },
    "fire_safety": {
        "general": [
            (r"\bпожарн\w*\s+безопасност\w*", 8),
            (r"\bпожар\w*", 6),
            (r"\bпротивопожарн\w*", 8),
        ],
        "fire_extinguishers": [
            (r"\bогнетушител\w*", 10),
        ],
        "evacuation": [
            (r"\bэвакуац\w*", 10),
            (r"\bэвакуационн\w*\s+выход\w*", 10),
        ],
        "fire_alarm": [
            (r"\bпожарн\w*\s+сигнализац\w*", 10),
            (r"\bоповещен\w*\s+о\s+пожар\w*", 10),
        ],
        "fire_briefing": [
            (r"\bпротивопожарн\w*\s+инструктаж\w*", 10),
        ],
        "fire_actions": [
            (r"\bчто\s+делать\b.*\bпожар\w*", 9),
            (r"\bобнаруж\w*\s+пожар\w*", 10),
        ],
    },
    "electrical_safety": {
        "general": [
            (r"\bэлектробезопасност\w*", 8),
            (r"\bэлектроустановк\w*", 8),
        ],
        "qualification_group": [
            (r"\bгрупп\w*\s+по\s+электробезопасност\w*", 10),
        ],
        "electrical_personnel": [
            (r"\bэлектротехническ\w*\s+персонал\w*", 10),
            (r"\bэлектротехнологическ\w*\s+персонал\w*", 10),
        ],
        "knowledge_testing": [
            (r"\bпровер\w*\s+знани\w*.*\bэлектротехническ\w*", 10),
            (r"\bпровер\w*\s+знани\w*.*\bэлектроустановк\w*", 10),
        ],
        "admission": [
            (r"\bдопуск\w*\s+к\s+работ\w*.*\bэлектроустановк\w*", 10),
        ],
        "responsible_person": [
            (r"\bответственн\w*\s+за\s+электрохозяйств\w*", 10),
        ],
        "protective_equipment": [
            (r"\bсредств\w*\s+защит\w*.*\bэлектроустановк\w*", 10),
        ],
    },
    "sanitary": {
        "microclimate": [
            (r"\bмикроклимат\w*", 10),
            (r"\bтемператур\w*\b.*\bрабоч\w*\s+мест\w*", 9),
            (r"\bтемператур\w*\b.*\bпроизводственн\w*", 9),
        ],
        "ventilation": [
            (r"\bвентиляц\w*", 10),
        ],
        "production_premises": [
            (r"\bсанитарн\w*\s+требован\w*.*\bпроизводственн\w*\s+помещен\w*", 10),
            (r"\bпроизводственн\w*\s+помещен\w*.*\bсанитар\w*", 9),
        ],
        "general": [
            (r"\bсанитар\w*", 7),
            (r"\bгигиен\w*", 7),
            (r"\bсанитарно-эпидемиологическ\w*", 8),
        ],
    },
    "occupational_safety": {
        "employer_duties": [
            (r"\bобязанност\w*\s+работодател\w*.*\bохран\w*\s+труд\w*", 10),
        ],
        "employee_rights": [
            (r"\bправ\w*\s+работник\w*.*\bохран\w*\s+труд\w*", 10),
        ],
        "management_system": [
            (r"\bсистем\w*\s+управлен\w*.*\bохран\w*\s+труд\w*", 10),
        ],
        "instructions": [
            (r"\bинструкци\w*\s+по\s+охран\w*\s+труд\w*", 10),
        ],
        "general": [
            (r"\bохран\w*\s+труд\w*", 6),
        ],
    },
}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def build_topic_hierarchy(user_query: str, base_topic: str = "general") -> Dict[str, Any]:
    """Возвращает иерархический смысловой маршрут запроса."""
    query = _normalize(user_query)
    if not query:
        return {
            "domain": "general",
            "topic": "general",
            "subtopic": "general",
            "legal_object": None,
            "confidence": 0.0,
            "signals": [],
        }

    domain_rules = _RULES.get(base_topic, {})
    candidates: List[Tuple[int, str, List[str]]] = []

    for subtopic, rules in domain_rules.items():
        score = 0
        signals: List[str] = []
        for pattern, weight in rules:
            if re.search(pattern, query, re.IGNORECASE):
                score += weight
                signals.append(pattern)
        if score:
            candidates.append((score, subtopic, signals))

    if not candidates:
        return {
            "domain": base_topic,
            "topic": base_topic,
            "subtopic": "general",
            "legal_object": None,
            "confidence": 0.55 if base_topic != "general" else 0.0,
            "signals": [],
        }

    candidates.sort(key=lambda x: (-x[0], x[1] == "general"))
    score, subtopic, signals = candidates[0]

    object_map = {
        ("industrial_safety", "production_control"): "production_control",
        ("industrial_safety", "expertise"): "industrial_safety_expertise",
        ("industrial_safety", "registration"): "hazardous_production_facility",
        ("industrial_safety", "accident"): "accident_at_hazardous_facility",
        ("fire_safety", "fire_extinguishers"): "fire_extinguisher",
        ("fire_safety", "evacuation"): "evacuation",
        ("fire_safety", "fire_alarm"): "fire_detection_and_alarm",
        ("fire_safety", "fire_briefing"): "fire_briefing",
        ("electrical_safety", "qualification_group"): "electrical_safety_group",
        ("electrical_safety", "electrical_personnel"): "electrical_personnel",
        ("electrical_safety", "knowledge_testing"): "electrical_knowledge_testing",
        ("electrical_safety", "admission"): "admission_to_electrical_work",
        ("electrical_safety", "responsible_person"): "electrical_responsible_person",
        ("electrical_safety", "protective_equipment"): "electrical_protective_equipment",
        ("sanitary", "microclimate"): "microclimate",
        ("sanitary", "ventilation"): "production_ventilation",
        ("sanitary", "production_premises"): "production_premises_sanitary_requirements",
        ("occupational_safety", "employer_duties"): "employer_ohs_duties",
        ("occupational_safety", "employee_rights"): "employee_ohs_rights",
        ("occupational_safety", "management_system"): "ohs_management_system",
        ("occupational_safety", "instructions"): "ohs_instructions",
    }

    return {
        "domain": base_topic,
        "topic": base_topic,
        "subtopic": subtopic,
        "legal_object": object_map.get((base_topic, subtopic)),
        "confidence": min(1.0, 0.45 + score / 20.0),
        "signals": signals,
    }


def hierarchy_preferred_terms(hierarchy: Dict[str, Any]) -> List[str]:
    """Русскоязычные термины для усиления hybrid retrieval."""
    terms = {
        "production_control": "производственный контроль",
        "expertise": "экспертиза промышленной безопасности",
        "registration": "регистрация опасного производственного объекта",
        "accident": "авария на опасном производственном объекте",
        "fire_extinguishers": "огнетушители",
        "evacuation": "эвакуационные выходы",
        "fire_alarm": "пожарная сигнализация",
        "fire_briefing": "противопожарный инструктаж",
        "fire_actions": "действия при пожаре",
        "qualification_group": "группа по электробезопасности",
        "electrical_personnel": "электротехнический персонал",
        "knowledge_testing": "проверка знаний электротехнического персонала",
        "admission": "допуск к работе в электроустановках",
        "responsible_person": "ответственный за электрохозяйство",
        "protective_equipment": "средства защиты в электроустановках",
        "microclimate": "микроклимат на рабочем месте",
        "ventilation": "вентиляция производственных помещений",
        "production_premises": "санитарные требования к производственным помещениям",
        "employer_duties": "обязанности работодателя в области охраны труда",
        "employee_rights": "права работника в области охраны труда",
        "management_system": "система управления охраной труда",
        "instructions": "инструкции по охране труда",
    }
    value = terms.get(hierarchy.get("subtopic"))
    return [value] if value else []
