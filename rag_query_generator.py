import re
from typing import Any, Dict, List

from rag_query_classifier import detect_special_category, _minor_special_issue, _is_labor_code_query, _is_target_briefing_query, _is_responsible_briefing_query
from rag_query_profile import build_universal_query_profile
from rag_query_modes import _attestation_query_mode, _accident_query_mode

def build_universal_search_queries(profile: Dict[str, Any], original: str) -> List[str]:
    """Генерирует нормативные формулировки независимо от конкретной темы."""
    event = profile.get("event")
    qtype = profile.get("question_type")
    state = profile.get("action_state")

    # Для количественного ограничения сначала идут точные нормативные
    # формулировки. Это важно: build_search_queries() ограничивает
    # универсальное расширение, поэтому специфические запросы не должны
    # теряться после общих legal_phrases.
    queries: List[str] = []

    if event == "lifting_and_moving_loads":
        constraint = profile.get("constraint") or {}
        if constraint.get("subject") == "adult_male":
            queries.extend([
                "пункт 86 постановления 12 26.01.2018 погрузочно-разгрузочные работы 50 кг мужчина",
                "предельно допустимая норма разового подъема тяжестей вручную работающим мужчиной 50 кг",
                "ручные погрузочно-разгрузочные работы разовый подъем тяжестей мужчиной не более 50 кг",
                "погрузочно-разгрузочные работы мужчина разовый подъем 50 кг",
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

        queries.extend(profile.get("legal_phrases") or [])
    else:
        if (
            event == "occupational_briefing"
            and qtype == "kind"
            and "work_break_over_six_months" in (profile.get("qualifiers") or [])
        ):
            queries.extend([
                "внеплановый инструктаж при перерыве в работе по профессии более шести месяцев",
                "перерыв в работе по профессии более шести месяцев внеплановый инструктаж",
                "какой инструктаж проводится при перерыве в работе более шести месяцев",
                "Инструкция № 175 пункт 27 внеплановый инструктаж перерыв более шести месяцев",
            ])
        queries.extend(profile.get("legal_phrases") or [])

    if event == "workplace_attestation" and qtype == "frequency":
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

    # Для вопросов «какой/какому инструктаж» и «кто проводит инструктаж»
    # должны существовать взаимоисключающие поисковые режимы.
    # Иначе responsible_person может добавить запросы про вводный
    # инструктаж даже тогда, когда пользователь спрашивает только вид.
    briefing_kind_query = (
        topic == "occupational_briefing"
        and universal_profile.get("question_type") == "kind"
    )

    target_briefing_query = (
        topic == "occupational_briefing"
        and _is_target_briefing_query(original)
    )

    responsible_briefing_query = (
        topic == "occupational_briefing"
        and not briefing_kind_query
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
        and not briefing_kind_query
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

    responsible_markers = (
        "кто проводит",
        "кто должен проводить",
        "кто обязан проводить",
        "кто имеет право проводить",
        "кто отвечает за проведение",
        "какое лицо проводит",
        "какой специалист проводит",
        "специалист по охране труда",
        "уполномоченное должностное лицо",
    )

    target_forbidden = responsible_markers
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

        # Режим «какой инструктаж» никогда не должен смешиваться
        # с режимом «кто проводит инструктаж».
        if briefing_kind_query and any(marker in qn for marker in responsible_markers):
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

    if briefing_kind_query:
        final_queries = [
            q for q in final_queries
            if not any(marker in q.lower() for marker in responsible_markers)
        ]
        if not final_queries:
            final_queries = [
                original,
                "виды инструктажей по охране труда",
                "вводный первичный повторный внеплановый целевой инструктаж по охране труда",
            ]

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


