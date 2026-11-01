from pathlib import Path
import re
import html

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak, KeepTogether
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "outputs" / "results.md"
OUT = ROOT / "outputs" / "Pneumonia_Detection_Report_EN.pdf"
FIGURES = ROOT / "outputs" / "figures"

pdfmetrics.registerFont(TTFont("Times", "C:/Windows/Fonts/times.ttf"))
pdfmetrics.registerFont(TTFont("Times-Bold", "C:/Windows/Fonts/timesbd.ttf"))
pdfmetrics.registerFont(TTFont("Times-Italic", "C:/Windows/Fonts/timesi.ttf"))

PAGE_W, PAGE_H = A4
LEFT = RIGHT = 2.1 * cm
TOP = 2.3 * cm
BOTTOM = 2.0 * cm
CONTENT_W = PAGE_W - LEFT - RIGHT

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(
    name="BodyCustom", fontName="Times", fontSize=10.3,
    leading=15, alignment=TA_LEFT, spaceAfter=7,
    textColor=colors.HexColor("#263238")
))
styles.add(ParagraphStyle(
    name="H1Custom", fontName="Times-Bold", fontSize=17,
    leading=21, textColor=colors.HexColor("#17365D"),
    spaceBefore=15, spaceAfter=9, keepWithNext=True
))
styles.add(ParagraphStyle(
    name="H2Custom", fontName="Times-Bold", fontSize=13,
    leading=17, textColor=colors.HexColor("#24527A"),
    spaceBefore=11, spaceAfter=6, keepWithNext=True
))
styles.add(ParagraphStyle(
    name="H3Custom", fontName="Times-Bold", fontSize=11,
    leading=14, textColor=colors.HexColor("#35627D"),
    spaceBefore=8, spaceAfter=4, keepWithNext=True
))
styles.add(ParagraphStyle(
    name="BulletCustom", parent=styles["BodyCustom"],
    leftIndent=14, firstLineIndent=-9, spaceAfter=4
))
styles.add(ParagraphStyle(
    name="SmallCustom", fontName="Times", fontSize=8.5,
    leading=11, alignment=TA_CENTER, textColor=colors.grey
))
styles.add(ParagraphStyle(
    name="CoverTitle", fontName="Times-Bold", fontSize=25,
    leading=32, alignment=TA_CENTER,
    textColor=colors.HexColor("#17365D")
))
styles.add(ParagraphStyle(
    name="CoverSubtitle", fontName="Times", fontSize=13,
    leading=19, alignment=TA_CENTER,
    textColor=colors.HexColor("#526575")
))
styles.add(ParagraphStyle(
    name="FigureCaption", fontName="Times-Bold", fontSize=10,
    leading=14, alignment=TA_CENTER, spaceBefore=8, spaceAfter=5,
    textColor=colors.HexColor("#17365D")
))

def inline_markup(text):
    text = html.escape(text, quote=False)
    text = re.sub(r"`([^`]+)`", r'<font name="Courier">\1</font>', text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*(.+?)\*", r"<i>\1</i>", text)
    text = re.sub(
        r"(https?://[^\s<]+)",
        r'<link href="\1" color="#24527A">\1</link>', text
    )
    return text

def page_decor(canvas, doc):
    canvas.saveState()
    if doc.page > 1:
        canvas.setStrokeColor(colors.HexColor("#D8E0E8"))
        canvas.line(LEFT, PAGE_H - 1.45*cm, PAGE_W-RIGHT, PAGE_H - 1.45*cm)
        canvas.setFont("Times", 8)
        canvas.setFillColor(colors.HexColor("#657786"))
        canvas.drawString(LEFT, PAGE_H - 1.15*cm, "PNEUMONIA DETECTION | RESEARCH REPORT")
        canvas.drawRightString(PAGE_W-RIGHT, 1.15*cm, f"Page {doc.page}")
        canvas.line(LEFT, 1.45*cm, PAGE_W-RIGHT, 1.45*cm)
    canvas.restoreState()

def make_table(rows):
    converted = []
    for row in rows:
        converted.append([
            Paragraph(inline_markup(cell.strip()), styles["BodyCustom"])
            for cell in row
        ])
    ncols = max(len(r) for r in rows)
    widths = [CONTENT_W / ncols] * ncols
    table = Table(converted, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#17365D")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Times-Bold"),
        ("GRID", (0,0), (-1,-1), 0.45, colors.HexColor("#C9D3DD")),
        ("ROWBACKGROUNDS", (0,1), (-1,-1),
         [colors.white, colors.HexColor("#F1F5F9")]),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
        ("RIGHTPADDING", (0,0), (-1,-1), 6),
        ("TOPPADDING", (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
    ]))
    return table

def add_figure(story, line):
    match = re.search(r"`(outputs/figures/[^`]+\.png)`", line)
    if not match:
        return False
    rel = match.group(1)
    path = ROOT / rel
    if not path.exists():
        story.append(Paragraph(
            f"Figure file not found: {html.escape(rel)}",
            styles["SmallCustom"]
        ))
        return True

    caption = re.sub(r"`outputs/figures/[^`]+`", "", line)
    caption = re.sub(r"^\s*-\s*", "", caption).strip()
    story.append(Spacer(1, 8))
    story.append(Paragraph(inline_markup(caption), styles["FigureCaption"]))

    img = Image(str(path), width=CONTENT_W, height=15.5*cm, kind="proportional")
    img.hAlign = "CENTER"
    story.append(img)
    story.append(Spacer(1, 12))
    return True

def build():
    if not REPORT.exists():
        raise FileNotFoundError(REPORT)

    text = REPORT.read_text(encoding="utf-8-sig")
    lines = text.splitlines()
    title = next(
        (line.lstrip("# ").strip() for line in lines if line.startswith("# ")),
        "Automated Pneumonia Detection from Chest X-ray Images"
    )

    doc = SimpleDocTemplate(
        str(OUT), pagesize=A4,
        rightMargin=RIGHT, leftMargin=LEFT,
        topMargin=TOP, bottomMargin=BOTTOM,
        title=title, author="Research Project"
    )

    story = []
    story.append(Spacer(1, 5.0*cm))
    story.append(Paragraph("RESEARCH PROJECT REPORT", styles["CoverSubtitle"]))
    story.append(Spacer(1, 1.2*cm))
    story.append(Paragraph(html.escape(title), styles["CoverTitle"]))
    story.append(Spacer(1, 1.0*cm))
    story.append(Paragraph(
        "Chest X-ray Image Classification Using Deep Learning "
        "and Conventional Machine Learning",
        styles["CoverSubtitle"]
    ))
    story.append(Spacer(1, 2.0*cm))
    story.append(Paragraph(
        "Experimental evaluation | MobileNetV2 | NIH supplementary data",
        styles["SmallCustom"]
    ))
    story.append(Spacer(1, 1.0*cm))
    for label, value in [
        ("Student Name", "Reyhane Zamzami"),
        ("Course", "Artificial Intelligence"),
        ("Supervisor", "Dr. Mehdi KorehPaz"),
        ("University", "Azad University (Sama)"),
        ("Submission Date", "1 November 2026"),
    ]:
        story.append(Paragraph(
            f"<b>{label}:</b> {value}",
            ParagraphStyle("CoverInfo", fontName="Times", fontSize=11,
                           leading=18, alignment=TA_CENTER)
        ))
    story.append(PageBreak())

    story.append(Paragraph("Contents", styles["H1Custom"]))
    for line in lines:
        if line.startswith("## "):
            heading = line[3:].strip()
            story.append(Paragraph(
                "• " + inline_markup(heading), styles["BodyCustom"]
            ))
    story.append(PageBreak())

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if not line or line.startswith("# "):
            i += 1
            continue

        if line.startswith("## "):
            story.append(Paragraph(inline_markup(line[3:]), styles["H1Custom"]))
            i += 1
            continue

        if line.startswith("### "):
            story.append(Paragraph(inline_markup(line[4:]), styles["H2Custom"]))
            i += 1
            continue

        if line.startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                current = lines[i].strip()
                cells = [c.strip() for c in current.strip("|").split("|")]
                if not all(re.fullmatch(r":?-{3,}:?", c or "---") for c in cells):
                    table_lines.append(cells)
                i += 1
            if table_lines:
                story.append(Spacer(1, 5))
                story.append(make_table(table_lines))
                story.append(Spacer(1, 9))
            continue

        if line.startswith("- ") or re.match(r"^\d+\.\s", line):
            if line.startswith("- "):
                content = "• " + line[2:]
            else:
                content = line
            story.append(Paragraph(inline_markup(content), styles["BulletCustom"]))
            if "outputs/figures/" in line:
                add_figure(story, line)
            i += 1
            continue

        if line.startswith("`") and line.endswith("`"):
            story.append(Paragraph(
                f'<font name="Courier">{html.escape(line.strip("`"))}</font>',
                styles["BodyCustom"]
            ))
            i += 1
            continue

        paragraph = [line]
        i += 1
        while i < len(lines):
            nxt = lines[i].strip()
            if (not nxt or nxt.startswith("#") or nxt.startswith("|")
                or nxt.startswith("- ") or re.match(r"^\d+\.\s", nxt)):
                break
            paragraph.append(nxt)
            i += 1
        story.append(Paragraph(
            inline_markup(" ".join(paragraph)), styles["BodyCustom"]
        ))

    doc.build(story, onFirstPage=page_decor, onLaterPages=page_decor)
    print(f"PDF created: {OUT}")
    print(f"Size: {OUT.stat().st_size:,} bytes")

if __name__ == "__main__":
    build()
