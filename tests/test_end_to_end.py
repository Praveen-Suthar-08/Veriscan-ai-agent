"""End-to-end integration tests for VeriScan agent and reporting pipeline."""

from pathlib import Path
from veriscan.schemas import Document, DocumentField, Finding, CaseReport
from veriscan.agent import Orchestrator
from veriscan.store import CaseStore
from veriscan.report import generate_pdf_report, generate_html_report, generate_json_report


class TestEndToEndPipeline:
    """Test full cross-document screening flow and storage."""

    def test_consistent_documents_flow(self):
        doc1 = Document(
            doc_id="doc_1", filename="id_card.png", doc_type="id_card",
            fields={
                "name": DocumentField(doc_id="doc_1", name="name", value_raw="Rohit Sharma", value_norm="rohit sharma", ocr_conf=0.95),
                "dob": DocumentField(doc_id="doc_1", name="dob", value_raw="12/03/2004", value_norm="2004-03-12", ocr_conf=0.92),
                "address": DocumentField(doc_id="doc_1", name="address", value_raw="MG Road, Bengaluru 560001", value_norm="mg road bengaluru 560001", ocr_conf=0.94)
            }
        )
        doc2 = Document(
            doc_id="doc_2", filename="marksheet.png", doc_type="marksheet",
            fields={
                "name": DocumentField(doc_id="doc_2", name="name", value_raw="Sharma Rohit", value_norm="rohit sharma", ocr_conf=0.93),
                "dob": DocumentField(doc_id="doc_2", name="dob", value_raw="12 Mar 2004", value_norm="2004-03-12", ocr_conf=0.90)
            }
        )

        orchestrator = Orchestrator()
        findings, consistent_items = orchestrator.compare_documents([doc1, doc2])

        # Expect zero MISMATCH findings
        mismatches = [f for f in findings if f.verdict == "MISMATCH"]
        assert len(mismatches) == 0

        # Expect both name and dob in consistent items
        consistent_fields = [c.field for c in consistent_items]
        assert "name" in consistent_fields
        assert "dob" in consistent_fields

    def test_dob_mismatch_flagged(self):
        doc1 = Document(
            doc_id="doc_1", filename="id_card.png", doc_type="id_card",
            fields={
                "dob": DocumentField(doc_id="doc_1", name="dob", value_raw="12/03/2004", value_norm="2004-03-12", ocr_conf=0.95)
            }
        )
        doc2 = Document(
            doc_id="doc_2", filename="marksheet.png", doc_type="marksheet",
            fields={
                "dob": DocumentField(doc_id="doc_2", name="dob", value_raw="15/03/2004", value_norm="2004-03-15", ocr_conf=0.95)
            }
        )

        orchestrator = Orchestrator()
        findings, _ = orchestrator.compare_documents([doc1, doc2])

        mismatches = [f for f in findings if f.verdict == "MISMATCH"]
        assert len(mismatches) == 1
        assert mismatches[0].field == "dob"
        assert mismatches[0].severity == "HIGH"

    def test_sqlite_store_and_decision_lifecycle(self, tmp_path):
        db_file = tmp_path / "test_veriscan.db"
        store = CaseStore(db_path=db_file)

        report = CaseReport(case_id="case_integ_01", triage="AMBER", risk_score=0.45)
        store.save_case(report)

        retrieved = store.get_case("case_integ_01")
        assert retrieved is not None
        assert retrieved.triage == "AMBER"

        # Record human reviewer decision
        dec = store.record_decision(
            case_id="case_integ_01",
            finding_id="finding_dob",
            status="CONFIRMED",
            notes="Applicant confirmed birth year discrepancy."
        )
        assert dec.status == "CONFIRMED"

        decisions = store.get_decisions("case_integ_01")
        assert "finding_dob" in decisions
        assert decisions["finding_dob"].notes == "Applicant confirmed birth year discrepancy."

        # Verify audit log
        audit = store.get_audit_trail("case_integ_01")
        assert len(audit) >= 2  # Case creation + decision
