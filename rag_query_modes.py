import re


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
