"""Report generation module for VeriScan (HTML, PDF via ReportLab, and JSON).

Every report export embeds the mandatory human-in-the-loop disclaimer.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Optional, Dict, Any
from io import BytesIO

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether
)
from reportlab.pdfgen import canvas

from veriscan.schemas import CaseReport, DISCLAIMER


class NumberedCanvas(canvas.Canvas):
    """Canvas that prints footer with page number and mandatory disclaimer on every page."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count: int):
        self.saveState()
        self.setFont("Helvetica", 7)
        self.setFillColor(colors.HexColor("#475569"))

        # Footer disclaimer
        disclaimer_text = DISCLAIMER
        # Wrap disclaimer across 2 lines if needed
        self.drawString(54, 30, disclaimer_text[:110])
        self.drawString(54, 20, disclaimer_text[110:])

        # Page number
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 20, page_str)

        # Header rule
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(54, 750, 558, 750)
        self.drawString(54, 755, "VeriScan — Pre-Screening Consistency Report")
        self.restoreState()


def generate_json_report(report: CaseReport) -> str:
    """Generate complete JSON export of CaseReport."""
    return json.dumps(report.model_dump(), indent=2)


def generate_html_report(report: CaseReport) -> str:
    """Generate clean standalone HTML representation of report."""
    triage_color = (
        "#10B981" if report.triage == "GREEN"
        else ("#F59E0B" if report.triage == "AMBER" else "#EF4444")
    )

    findings_html = ""
    for f in report.findings:
        findings_html += f"""
        <tr style="border-bottom: 1px solid #E2E8F0;">
            <td style="padding: 10px; font-weight: 600;">{f.field.upper()}</td>
            <td style="padding: 10px;"><span style="background: #F1F5F9; padding: 4px 8px; border-radius: 4px;">{f.verdict}</span></td>
            <td style="padding: 10px;">{f.severity}</td>
            <td style="padding: 10px;">{f.confidence:.2f}</td>
            <td style="padding: 10px;">{'; '.join(f.reasons)}</td>
            <td style="padding: 10px; color: #475569;">{f.suggested_action}</td>
        </tr>
        """

    consistent_html = ""
    for c in report.consistent_items:
        consistent_html += f"""
        <li style="margin-bottom: 6px;"><strong>{c.field.upper()}</strong> ({', '.join(c.docs)}): {'; '.join(c.reasons)}</li>
        """

    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>VeriScan Report - {report.case_id}</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #F8FAFC; color: #0F172A; margin: 0; padding: 30px; }}
            .container {{ max-width: 900px; margin: 0 auto; background: #FFFFFF; padding: 32px; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }}
            .disclaimer-banner {{ background: #FEF3C7; border-left: 4px solid #F59E0B; padding: 12px 16px; margin-bottom: 24px; font-size: 13px; color: #92400E; }}
            .triage-badge {{ display: inline-block; padding: 6px 14px; border-radius: 20px; font-weight: 700; color: white; background: {triage_color}; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 16px; font-size: 14px; }}
            th {{ text-align: left; background: #F8FAFC; padding: 10px; border-bottom: 2px solid #CBD5E1; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="disclaimer-banner">
                <strong>LEGAL DISCLAIMER:</strong> {report.disclaimer}
            </div>

            <header style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #E2E8F0; padding-bottom: 16px; margin-bottom: 24px;">
                <div>
                    <h1 style="margin: 0; font-size: 24px; color: #0B192C;">VeriScan Screening Report</h1>
                    <p style="margin: 4px 0 0 0; color: #64748B; font-size: 14px;">Case ID: {report.case_id} | OCR: {report.ocr_engine} | Date: {report.timestamp[:19]}</p>
                </div>
                <div>
                    <span class="triage-badge">{report.triage} (Risk: {report.risk_score:.2f})</span>
                </div>
            </header>

            <section style="margin-bottom: 24px;">
                <h2 style="font-size: 18px; border-bottom: 1px solid #E2E8F0; padding-bottom: 8px;">Executive Summary</h2>
                <p style="white-space: pre-line; line-height: 1.6;">{report.narrative}</p>
            </section>

            <section style="margin-bottom: 24px;">
                <h2 style="font-size: 18px; border-bottom: 1px solid #E2E8F0; padding-bottom: 8px;">Ranked Discrepancies & Findings ({len(report.findings)})</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Field</th>
                            <th>Verdict</th>
                            <th>Severity</th>
                            <th>Confidence</th>
                            <th>Reason</th>
                            <th>Suggested Action</th>
                        </tr>
                    </thead>
                    <tbody>
                        {findings_html if findings_html else "<tr><td colspan='6' style='padding: 12px; text-align: center; color: #10B981;'>No discrepancies flagged. All fields consistent.</td></tr>"}
                    </tbody>
                </table>
            </section>

            <section style="margin-bottom: 24px;">
                <h2 style="font-size: 18px; border-bottom: 1px solid #E2E8F0; padding-bottom: 8px;">Checked and Consistent Fields ({len(report.consistent_items)})</h2>
                <ul>
                    {consistent_html if consistent_html else "<li>None</li>"}
                </ul>
            </section>

            <footer style="margin-top: 40px; padding-top: 16px; border-top: 1px solid #E2E8F0; font-size: 12px; color: #64748B;">
                {report.disclaimer}
            </footer>
        </div>
    </body>
    </html>
    """
    return html


def generate_pdf_report(report: CaseReport) -> bytes:
    """Generate downloadable PDF report using ReportLab with disclaimer on every page."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#0B192C")
    )
    meta_style = ParagraphStyle(
        "MetaText",
        parent=styles["Normal"],
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#475569")
    )
    disclaimer_style = ParagraphStyle(
        "DisclaimerTop",
        parent=styles["Normal"],
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#92400E")
    )
    section_style = ParagraphStyle(
        "SectionHeader",
        parent=styles["Heading2"],
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#1E293B"),
        spaceBefore=14,
        spaceAfter=6
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#1E293B")
    )

    story = []

    # 1. Mandatory Top Disclaimer Strip
    disclaimer_box = Table(
        [[Paragraph(f"<b>SCREENING AID DISCLAIMER:</b> {report.disclaimer}", disclaimer_style)]],
        colWidths=[504]
    )
    disclaimer_box.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#FEF3C7")),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#F59E0B")),
        ('PADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(disclaimer_box)
    story.append(Spacer(1, 14))

    # 2. Header & Triage
    triage_bg = (
        "#10B981" if report.triage == "GREEN"
        else ("#F59E0B" if report.triage == "AMBER" else "#EF4444")
    )
    header_data = [
        [
            Paragraph(f"<b>VeriScan Consistency Report</b>", title_style),
            Paragraph(f"<font color='white'><b>{report.triage} (Risk {report.risk_score:.2f})</b></font>", ParagraphStyle(
                "TriageP", parent=styles["Normal"], alignment=1, fontSize=11, leading=14
            ))
        ],
        [
            Paragraph(f"Case ID: {report.case_id} | Engine: {report.ocr_engine} | Date: {report.timestamp[:19]}", meta_style),
            ""
        ]
    ]
    header_table = Table(header_data, colWidths=[384, 120])
    header_table.setStyle(TableStyle([
        ('BACKGROUND', (1, 0), (1, 0), colors.HexColor(triage_bg)),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 14))

    # 3. Narrative
    story.append(Paragraph("Executive Narrative", section_style))
    story.append(Paragraph(report.narrative.replace("\n", "<br/>"), body_style))
    story.append(Spacer(1, 14))

    # 4. Findings Table
    story.append(Paragraph(f"Ranked Discrepancies ({len(report.findings)})", section_style))
    table_rows = [
        ["Field", "Verdict", "Severity", "Conf.", "Reason & Action"]
    ]
    for f in report.findings:
        reasons_text = f"<b>Reason:</b> {'; '.join(f.reasons)}<br/><b>Action:</b> {f.suggested_action}"
        table_rows.append([
            f.field.upper(),
            f.verdict,
            f.severity,
            f"{f.confidence:.2f}",
            Paragraph(reasons_text, body_style)
        ])

    if len(table_rows) == 1:
        table_rows.append(["None", "MATCH", "INFO", "1.00", "All fields consistent across submitted proofs."])

    findings_table = Table(table_rows, colWidths=[70, 75, 55, 45, 259])
    findings_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor("#0B192C")),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('PADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(findings_table)
    story.append(Spacer(1, 14))

    # 5. Checked & Consistent List
    story.append(Paragraph(f"Checked and Consistent Fields ({len(report.consistent_items)})", section_style))
    consistent_rows = [["Field", "Documents", "Status & Rationale"]]
    for c in report.consistent_items:
        consistent_rows.append([
            c.field.upper(),
            ", ".join(c.docs),
            Paragraph("; ".join(c.reasons), body_style)
        ])
    if len(consistent_rows) == 1:
        consistent_rows.append(["-", "-", "None checked"])

    consistent_table = Table(consistent_rows, colWidths=[80, 100, 324])
    consistent_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#F8FAFC")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('PADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(consistent_table)

    # Build PDF with NumberedCanvas ensuring disclaimer on every page
    doc.build(story, canvasmaker=NumberedCanvas)
    buffer.seek(0)
    return buffer.getvalue()
