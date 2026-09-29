"""Структурированный профиль пользовательского запроса для RAG."""

import re
from typing import Any, Dict

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
        "target_document": None,
        "target_article": None,
        "qualifiers": [],
        "legal_phrases": [],
        "constraint": {
            "type": None,
            "value": None,
            "unit": None,
            "subject": None,
            "action": None,
            "scope": None,
            "scope_signals": [],
        },
    }
    if not query:
        return profile

    scope_pattern = (
        r"\bна\s+кого\s+(?:распространяется|распространяются|действует|действуют|применяется|применяются)\b|"
        r"\bсфера\s+действия\b|"
        r"\bв\s+отношении\s+кого\b|"
        r"\bк\s+кому\s+(?:применяется|относится)\b"
    )
    if re.search(scope_pattern, query, re.IGNORECASE):
        profile["question_type"] = "scope"
        profile["event"] = "law_scope"
        profile["object"] = "law_scope"

        if re.search(
            r"\b356[-–—]з\b|\b356\s*[-–—]\s*з\b|"
            r"закона?\s+.{0,80}об\s+охране\s+труда",
            query,
            re.IGNORECASE,
        ):
            profile["target_document"] = "356-з"
            profile["target_article"] = "3"
            profile["legal_phrases"].extend([
                "сфера действия настоящего Закона",
                "настоящий Закон применяется в отношении всех работодателей",
                "работающих граждан Республики Беларусь",
                "иностранных граждан и лиц без гражданства",
            ])
        elif re.search(
            r"\bтрудов(?:ого|ым)\s+кодекс\w*\b|\bтрудовой\s+кодекс\b",
            query,
            re.IGNORECASE,
        ):
            profile["target_document"] = "трудовой кодекс"
            profile["legal_phrases"].extend([
                "сфера действия Трудового кодекса",
                "трудовые отношения",
            ])
        elif re.search(r"\bправила\s+по\s+охране\s+труда\b", query, re.IGNORECASE):
            # Общие Правила по охране труда — постановление Минтруда № 53.
            # Специальные правила «при ...» не переводим в этот документ.
            rules_match = re.search(r"\bправила\s+по\s+охране\s+труда\b", query, re.IGNORECASE)
            tail = query[rules_match.end():] if rules_match else ""
            if not re.match(r"\s+при\b", tail, re.IGNORECASE):
                profile["target_document"] = "правила по охране труда 53"
                profile["target_article"] = "2"
                profile["legal_phrases"].extend([
                    "Правила по охране труда",
                    "требования по охране труда распространяются на работодателей",
                    "независимо от их организационно-правовых форм и форм собственности",
                    "различные виды экономической деятельности",
                ])

    qtypes = [
        ("scope", [scope_pattern]),
        ("what_to_do", [r"\bчто\s+делать\b", r"\bкак\s+(?:должен|следует)\s+действ", r"\bпорядок\s+действ"]),
        ("frequency", [r"\bкак\s+часто\b", r"\bс\s+какой\s+периодичност", r"\bпериодичност\w*\b"]),
        ("limit", [r"\bсколько\b.*\bкг\b", r"\bсколько\s+разрешено\b", r"\bпредельн\w*\s+норм", r"\bнорм\w*\s+(?:подъема|перемещения)"]),
        ("who", [r"^кто\b", r"\bкто\s+(?:провод|должен|обязан|назнач|ответствен)", r"\bкем\b", r"\bкакое\s+лицо\b"]),
        ("kind", [r"\bкакой\s+(?:вид\s+)?инструктаж", r"\bкакому\s+инструктаж", r"\bвид\w*\s+инструктаж"]),
        ("whether", [r"\bможно\s+ли\b", r"\bразрешено\s+ли\b", r"\bдопускается\s+ли\b", r"\bимеет\s+ли\s+прав", r"\bобязан\s+ли\b", r"\bобязана\s+ли\b"]),
        ("responsibility", [r"\bкто\s+нес[её]т\s+ответствен", r"\bкто\s+ответствен", r"\bкакая\s+ответствен"]),
        ("term", [r"\bкакой\s+срок\b", r"\bсрок\w*\b", r"\bв\s+течение\b"]),
        ("document", [r"\bкаким\s+документ", r"\bкакой\s+(?:нпа|документ|акт)\b"]),
    ]
    for qtype, patterns in qtypes:
        if any(re.search(p, query, re.IGNORECASE) for p in patterns):
            profile["question_type"] = qtype
            break

    if (\n        re.search(r"\bотстран\w*", query, re.IGNORECASE)\n        and re.search(r"\bне\s+прош\w*\s+инструктаж\w*", query, re.IGNORECASE)\n        and re.search(r"\bпровер\w*\s+знан\w*", query, re.IGNORECASE)\n    ):\n        profile["qualifiers"].append("suspension_for_unpassed_osh_training")\n        profile["action"] = "suspend_work"\n        profile["legal_phrases"].extend([\n            "Трудовой кодекс Республики Беларусь статья 49 отстранение от работы",\n            "наниматель обязан не допускать к работе не прошедшего инструктаж стажировку и проверку знаний",\n            "не прошедший инструктаж стажировку и проверку знаний по вопросам охраны труда",\n            "отстранить от работы в соответствующий день смену",\n        ])\n\n    if re.search(r"\bне\s+сообщил\w*\b|\bне\s+сообщила\w*\b|\bне\s+выдан\w*\b|\bне\s+обеспечен\w*\b|\bне\s+прошел\w*\b|\bне\s+приступ\w*\b", query, re.IGNORECASE):
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

        # Отдельно определяем область действия нормы. Это предотвращает
        # ошибочное превращение конкретной нормы «50 кг» в универсальный
        # предел для любого ручного труда.
        lifting_scope_signals = []
        if re.search(
            r"\bпогрузочн\w*[-–— ]+разгрузочн\w*|\bпогрузк\w*\b|\bразгрузк\w*\b",
            query,
            re.IGNORECASE,
        ):
            lifting_scope_signals.extend([
                "manual_loading_unloading",
                "погрузочно-разгрузочные работы",
            ])
        elif re.search(
            r"\bвручн\w*|\bподнима\w*\b|\bперемещ\w*\s+тяжест\w*|\bперенос\w*\b",
            query,
            re.IGNORECASE,
        ):
            lifting_scope_signals.append("manual_handling")

        scope = (
            "manual_loading_unloading"
            if "manual_loading_unloading" in lifting_scope_signals
            else "manual_handling"
            if "manual_handling" in lifting_scope_signals
            else None
        )

        constraint_data = {
            "type": "maximum",
            "unit": "kg",
            "action": "lifting",
            "scope": scope,
            "scope_signals": lifting_scope_signals,
        }
        if is_male:
            profile["qualifiers"].append("men")
            constraint_data["subject"] = "adult_male"
        elif is_female:
            profile["qualifiers"].append("women")
            constraint_data["subject"] = "adult_female"

        profile["constraint"].update(constraint_data)

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

        # Отдельный квалификатор для юридического вопроса о праве
        # работника отказаться от работы при необеспечении СИЗ.
        # Такой вопрос нельзя оставлять только в общем PPE-поиске:
        # нужны нормы о правах работника в области охраны труда.
        refusal_due_to_no_ppe_pattern = (
            r"(?:имеет\s+ли\s+прав\w*|может\s+ли|можно\s+ли|"
            r"разрешено\s+ли|допускается\s+ли)"
            r".{0,100}?"
            r"(?:отказ\w*\s+от\s+(?:выполнения\s+)?работ\w*|"
            r"отказ\w*\s+работ\w*|не\s+приступ\w*\s+к\s+работ\w*)"
            r".{0,100}?"
            r"(?:сиз|средств\w*\s+индивидуальн\w*\s+защит)"
        )
        refusal_due_to_no_ppe_reverse_pattern = (
            r"(?:сиз|средств\w*\s+индивидуальн\w*\s+защит)"
            r".{0,100}?"
            r"(?:отказ\w*\s+от\s+(?:выполнения\s+)?работ\w*|"
            r"отказ\w*\s+работ\w*|не\s+приступ\w*\s+к\s+работ\w*)"
        )

        if (
            re.search(refusal_due_to_no_ppe_pattern, query, re.IGNORECASE)
            or re.search(
                refusal_due_to_no_ppe_reverse_pattern,
                query,
                re.IGNORECASE,
            )
        ):
            profile["qualifiers"].append("refusal_due_to_no_ppe")
            profile["action"] = "refuse_work"
            profile["legal_phrases"].extend([
                "право работника отказаться от выполнения работы при необеспечении средствами индивидуальной защиты",
                "отказ работника от выполнения работы при отсутствии средств индивидуальной защиты",
                "право на отказ от работы при невыдаче средств индивидуальной защиты",
                "права работника в области охраны труда средства индивидуальной защиты отказ от работы",
            ])

        # Отдельный сценарий: пользователь спрашивает не о самом праве
        # на отказ, а о том, что работник должен сделать при невыдаче СИЗ.
        ppe_nonprovision_action_pattern = (
            r"(?:что\s+(?:должен|следует)\s+(?:сделать|делать)|"
            r"как\s+(?:должен|следует)\s+поступить|"
            r"каков\w*\s+порядок\s+действий)"
            r".{0,100}?"
            r"(?:не\s+(?:выдан\w*|выда\w*|предостав\w*|обеспеч\w*))"
            r".{0,100}?"
            r"(?:сиз|средств\w*\s+индивидуальн\w*\s+защит)"
        )
        ppe_nonprovision_action_reverse_pattern = (
            r"(?:сиз|средств\w*\s+индивидуальн\w*\s+защит)"
            r".{0,100}?"
            r"(?:не\s+(?:выдан\w*|выда\w*|предостав\w*|обеспеч\w*))"
            r".{0,100}?"
            r"(?:что\s+(?:должен|следует)\s+(?:сделать|делать)|"
            r"как\s+(?:должен|следует)\s+поступить|"
            r"порядок\s+действий)"
        )

        if (
            re.search(ppe_nonprovision_action_pattern, query, re.IGNORECASE)
            or re.search(
                ppe_nonprovision_action_reverse_pattern,
                query,
                re.IGNORECASE,
            )
        ):
            profile["qualifiers"].append("ppe_nonprovision_action")
            profile["action"] = "refuse_work"
            profile["legal_phrases"].extend([
                "что должен сделать работник при непредоставлении средств индивидуальной защиты",
                "действия работника при невыдаче средств индивидуальной защиты",
                "обязанность работника письменно сообщить о причинах отказа от работы без СИЗ",
                "статья 11 Закон 356-З средства индивидуальной защиты отказ от работы",
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

            # Для вопросов вида «какой инструктаж при перерыве ...»
            # важнее искать не общий перечень инструктажей, а норму,
            # связывающую условие перерыва с конкретным видом инструктажа.
            work_break_pattern = (
                r"\bперерыв\w*\s+(?:в\s+работе\s+)?"
                r"(?:по\s+профессии|в\s+должности|в\s+выполнении\s+работ)?"
                r".{0,60}?"
                r"(?:более\s+(?:полугод|шест(?:и|ь)\s+месяц|6\s*месяц|"
                r"одн(?:ого|ому)\s+года|год|12\s*месяц)|"
                r"свыше\s+(?:шест(?:и|ь)\s+месяц|6\s*месяц|"
                r"одн(?:ого|ому)\s+года|год|12\s*месяц))"
            )

            if re.search(work_break_pattern, query, re.IGNORECASE):
                profile["qualifiers"].append("work_break_over_six_months")
                profile["legal_phrases"].extend([
                    "внеплановый инструктаж при перерыве в работе по профессии более шести месяцев",
                    "перерыв в работе по профессии более шести месяцев внеплановый инструктаж",
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


