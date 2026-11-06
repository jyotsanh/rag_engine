"""Regenerate the binary documents served by the fixture site.

    uv run python tests/fixtures/build_documents.py

Outputs are committed under `tests/fixtures/site/files/` (and `images/`), so this only
needs re-running when the document content changes. Metadata dates are pinned so
re-runs produce stable content.
"""

import datetime as dt
from pathlib import Path

import pymupdf
from docx import Document
from openpyxl import Workbook
from pptx import Presentation
from pptx.util import Inches

SITE = Path(__file__).parent / "site"
FILES = SITE / "files"
IMAGES = SITE / "images"
PINNED = dt.datetime(2026, 1, 1, 0, 0, 0)


def build_pdf(path: Path) -> None:
    doc = pymupdf.open()
    pages = [
        (
            "Widget Specification",
            [
                "The WX-2048 widget is a modular sorter rated for 1200 items per minute.",
                "Operating temperature: -10 to 55 degrees Celsius.",
                "Power draw: 48 W nominal, 65 W peak during calibration.",
            ],
        ),
        (
            "Electrical Interface",
            [
                "The widget connects over a 24 V DC bus with an M12 A-coded connector.",
                "Pin 1 carries power, pin 3 is ground, pins 2 and 4 carry RS-485 data.",
            ],
        ),
    ]
    for title, lines in pages:
        page = doc.new_page()
        page.insert_text((72, 72), title, fontsize=20)
        y = 110
        for line in lines:
            page.insert_text((72, y), line, fontsize=11)
            y += 18
    doc.set_metadata({"title": "Widget Specification", "author": "Acme Widgets"})
    doc.save(path, garbage=4, deflate=True, no_new_id=True)


def build_scanned_pdf(path: Path) -> None:
    """A PDF with no text layer (only vector shapes), like a scan without OCR."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.draw_rect(pymupdf.Rect(72, 72, 300, 200), color=(0, 0, 0), fill=(0.8, 0.8, 0.8))
    page.draw_line((72, 250), (500, 250))
    doc.save(path, no_new_id=True)


def build_encrypted_pdf(path: Path) -> None:
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "Confidential pricing", fontsize=14)
    doc.save(
        path,
        encryption=pymupdf.PDF_ENCRYPT_AES_256,  # type: ignore  # set at runtime
        owner_pw="owner-secret",
        user_pw="user-secret",
        no_new_id=True,
    )


def build_docx(path: Path) -> None:
    doc = Document()
    doc.add_heading("Installation Guide", level=1)
    doc.add_paragraph("This guide covers mounting and wiring the WX-2048 widget.")
    doc.add_heading("Mounting", level=2)
    doc.add_paragraph("Use four M6 bolts torqued to 9 Nm.", style="List Bullet")
    doc.add_paragraph("Leave 50 mm clearance for airflow.", style="List Bullet")
    doc.add_heading("Wiring", level=2)
    table = doc.add_table(rows=3, cols=2)
    wiring = [("Pin", "Signal"), ("1", "+24 V"), ("3", "GND")]
    for row, (pin, signal) in zip(table.rows, wiring, strict=True):
        row.cells[0].text, row.cells[1].text = pin, signal
    props = doc.core_properties
    props.title, props.author, props.created, props.modified = (
        "Installation Guide",
        "Acme Widgets",
        PINNED,
        PINNED,
    )
    doc.save(str(path))


def build_pptx(path: Path) -> None:
    prs = Presentation()
    slides = [
        ("Product Roadmap 2026", "Widget Controller v3 ships in Q2 with fleet-wide OTA updates."),
        ("Q3: Open Widget Protocol 2.0", "Adds encrypted telemetry and frame compression."),
    ]
    for title, body in slides:
        slide = prs.slides.add_slide(prs.slide_layouts[1])
        slide.shapes.title.text = title
        slide.placeholders[1].text = body
        slide.notes_slide.notes_text_frame.text = f"Speaker notes: emphasise {title.lower()}."
    table_slide = prs.slides.add_slide(prs.slide_layouts[5])
    table_slide.shapes.title.text = "Release dates"
    rows = [("Release", "Date"), ("v3.0", "2026-05-01"), ("v3.1", "2026-08-15")]
    shape = table_slide.shapes.add_table(3, 2, Inches(1), Inches(2), Inches(6), Inches(1.5))
    for r, values in enumerate(rows):
        for c, value in enumerate(values):
            shape.table.cell(r, c).text = value
    prs.core_properties.created = PINNED
    prs.core_properties.modified = PINNED
    prs.save(str(path))


def build_xlsx(path: Path) -> None:
    wb = Workbook()
    prices = wb.active
    assert prices is not None
    prices.title = "Prices"
    for row in [
        ("SKU", "Description", "Unit price (USD)"),
        ("WX-2048", "Modular sorter", 1899),
        ("WX-4096", "High-throughput sorter", 3499),
        ("SNS-T1", "Torque sensor pack", 249),
    ]:
        prices.append(row)
    discounts = wb.create_sheet("Discounts")
    for row in [("Quantity", "Discount"), ("10+", "5%"), ("50+", "12%")]:
        discounts.append(row)
    wb.properties.created = PINNED
    wb.properties.modified = PINNED
    wb.save(path)


def build_png(path: Path) -> None:
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 16, 16), False)
    pix.set_rect(pix.irect, (200, 30, 30))
    pix.save(path)


def main() -> None:
    FILES.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)
    build_pdf(FILES / "widget-spec.pdf")
    build_scanned_pdf(FILES / "scanned.pdf")
    build_encrypted_pdf(FILES / "encrypted.pdf")
    build_docx(FILES / "install-guide.docx")
    build_pptx(FILES / "roadmap.pptx")
    build_xlsx(FILES / "pricing.xlsx")
    build_png(IMAGES / "logo.png")


if __name__ == "__main__":
    main()
