"""Focused legal-query construction for photo inspection findings.

The vision model describes observable risks; this module routes each finding
to the relevant Belarus legal domain before RAG retrieval.
"""


def build_finding_query(finding: dict) -> str:
    raw_query = str(finding.get("description") or "").strip()
    visual_evidence = str(finding.get("visual_evidence") or "").strip()
    text = f"{raw_query} {visual_evidence}".lower()

    # An explicit evacuation-route finding must be searched against fire-safety
    # rules. Keep this branch separate from scaffolding/work-at-height findings.
    if any(marker in text for marker in (
        "эвакуац", "путь эвакуации", "пути эвакуации",
        "эвакуационн", "загроможд", "свободн.*проход",
    )):
        focus = (
            "пожарная безопасность: действующие правила пожарной безопасности "
            "Республики Беларусь, содержание путей эвакуации свободными, "
            "лестничные клетки и марши, загромождение проходов, "
            "минимальная эвакуационная ширина и высота"
        )
    elif any(marker in text for marker in (
        "настил", "доск", "брус", "деревянн", "вышк", "тур",
        "строительн.*лес", "лесах", "подмост", "подмащив",
        "опор", "работ.*высот", "огражден", "перил", "бортов",
        "неинвентар", "площадк.*лестниц",
    )):
        focus = (
            "охрана труда: действующие правила Республики Беларусь при "
            "выполнении работ на высоте и строительных работ; строительные "
            "леса, вышки-туры, подмости и средства подмащивания, настилы, "
            "опоры и устойчивость, перильные и бортовые ограждения, "
            "проект производства работ, приёмка и осмотр средств подмащивания"
        )
    elif any(marker in text for marker in (
        "сварочн", "welding", "щиток", "helmet", "очки", "glasses",
        "перчат", "glove", "каск", "сиз", "ppe", "кабел", "cable",
        "электроинструмент", "grinder", "болгар", "ушм", "дрел", "drill",
        "пил", "tool", "инструмент", "рукоят", "кожух", "guard",
    )):
        focus = "охрана труда: действующие требования Республики Беларусь к безопасному выполнению работ и применению оборудования"
    elif any(marker in text for marker in (
        "огнетуш", "extinguisher", "искр", "sparks", "горюч",
        "combust", "пожар", "fire", "средства пожаротушения",
    )):
        focus = "пожарная безопасность: действующие требования Республики Беларусь"
    else:
        focus = "охрана труда и пожарная безопасность: применимые действующие требования Республики Беларусь"

    query = f"Нормативные требования Республики Беларусь: {focus}. Наблюдение: {raw_query}"
    if visual_evidence:
        query += f". Визуальный признак: {visual_evidence}"
    return query
