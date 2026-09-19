from io import BytesIO
import fitz  # type: ignore # PyMuPDF

from veriscan.schemas import CaseReport, DISCLAIMER
from veriscan.report import generate_html_report, generate_pdf_report, generate_json_report


class TestMandatoryDisclaimer:
    """Ensure the exact DISCLAIMER text is present in HTML, PDF, JSON, and schemas."""

    def test_disclaimer_constant_exact_string(self):
        expected = (
            "This report is an automated pre-screening aid. It flags possible inconsistencies for human review "
            "and does not determine authenticity, eligibility, or approval. Final judgment rests with the "
            "human reviewer. Only synthetic sample data is used in this demonstration."
        )
        assert DISCLAIMER == expected

    def test_disclaimer_present_in_json(self):
        report = CaseReport(case_id="case_test_01")
        json_output = generate_json_report(report)
        assert DISCLAIMER in json_output

    def test_disclaimer_present_in_html(self):
        report = CaseReport(case_id="case_test_01")
        html_output = generate_html_report(report)
        assert DISCLAIMER in html_output

    def test_disclaimer_present_in_pdf(self):
        report = CaseReport(case_id="case_test_01")
        pdf_bytes = generate_pdf_report(report)
        assert len(pdf_bytes) > 0

        # Read PDF pages using pypdf / PyPDF2 / PyMuPDF to assert disclaimer text
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        assert len(doc) >= 1
        for page in doc:
            text = page.get_text()
            # Verify core disclaimer keywords in footer of each page
            assert "automated pre-screening aid" in text
            assert "human reviewer" in text
            assert "synthetic sample data" in text
