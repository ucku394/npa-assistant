"""Generate a human-reviewable DOCX draft prescription."""

from datetime import date, timedelta
from io import BytesIO
from typing import Any, Dict, List, Optional

from docx import Document
from docx.shared import Inches, Pt

from config import PRESCRIPTION_DEFAULT_DEADLINE_DAYS


def build_draft_prescription(
    findings: List[Dict[str, Any]],
    photo_bytes: Optional[bytes] = None,
    enterprise: str = "____________________________",
    subdivision: str = "____________________________",
    workplace: str = "____________________________",
    responsible: str = "____________________________",
    deadline: Optional[str] = None,
) -> bytes:
    doc = Document()
    section = doc.sections[0]
    section.top_margin = Inches(0.6)
    section.bottom_margin = Inches(0.6)
    section.left_margin = Inches(0.7)
    section.right_margin = Inches(0.7)

    doc.styles["Normal"].font.name = "Arial"
    doc.styles["Normal"].font.size = Pt(10)

    p = doc.add_paragraph()
    p.alignment = 1
    run = p.add_run("ПРОЕКТ ПРЕДПИСАНИЯ")
    run.bold = True
    run.font.size = Pt(15)

    p = doc.add_paragraph()
    p.alignment = 1
    p.add_run("об устранении выявленных нарушений требований безопасности").bold = True

    doc.add_paragraph(f"Дата: {date.today():%d.%m.%Y}")
    doc.add_paragraph(f"Организация: {enterprise}")
    doc.add_paragraph(f"Подразделение: {subdivision}")
    doc.add_paragraph(f"Рабочее место / объект: {workplace}")
    doc.add_paragraph("Номер: ____________________")
    doc.add_paragraph()

    confirmed = [
        item for item in findings
        if item.get("status") == "confirmed" and item.get("legal_basis")
    ]

    if not confirmed:
        doc.add_paragraph(
            "Автоматически подтверждённых нарушений нет. "
            "Документ сформирован как проект и требует проверки специалистом."
        )
    else:
        table = doc.add_table(rows=1, cols=6)
        table.style = "Table Grid"
        headers = [
            "№",
            "Выявленное нарушение",
            "Нормативное основание",
            "Корректирующее мероприятие",
            "Срок",
            "Ответственный",
        ]
        for cell, header in zip(table.rows[0].cells, headers):
            cell.text = header

        due = deadline or (
            date.today() + timedelta(days=PRESCRIPTION_DEFAULT_DEADLINE_DAYS)
        ).strftime("%d.%m.%Y")

        for index, item in enumerate(confirmed, 1):
            basis = "; ".join(
                f"{x.get('document', '')} — {x.get('point', '')} "
                f"[{x.get('source_id', '')}]"
                for x in item.get("legal_basis", [])
            )
            cells = table.add_row().cells
            cells[0].text = str(index)
            cells[1].text = str(item.get("violation") or "")
            cells[2].text = basis
            cells[3].text = str(
                item.get("corrective_action")
                or "Устранить выявленное несоответствие и повторно проверить."
            )
            cells[4].text = due
            cells[5].text = responsible

    doc.add_paragraph()
    doc.add_paragraph(
        "Примечание: документ сформирован автоматически как ПРОЕКТ. "
        "Перед официальным применением его должен проверить и утвердить "
        "уполномоченный специалист."
    )

    if photo_bytes:
        doc.add_paragraph("Фотографическое подтверждение:")
        try:
            doc.add_picture(BytesIO(photo_bytes), width=Inches(5.8))
        except Exception:
            pass

    doc.add_paragraph()
    doc.add_paragraph("Составил: ____________________________")
    doc.add_paragraph("Проверил: ____________________________")
    doc.add_paragraph("Ответственный за устранение: ____________________________")
    doc.add_paragraph("Подпись: __________________  Дата: ______________")

    output = BytesIO()
    doc.save(output)
    return output.getvalue()
