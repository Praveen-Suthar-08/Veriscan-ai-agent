"""End-to-End sample case tests for VeriScan.

Tests the full pipeline (OCR → Classify → Extract → Compare → Score) against
expected.json outcomes for all 4 bundled sample cases.

IMPORTANT: These tests use pre-cached token files (not live OCR) for speed.
For live OCR validation, run with --use-live-ocr flag (pytest -m live_ocr).
"""

from __future__ import annotations
import json
import sys
from pathlib import Path
import pytest  # type: ignore

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from veriscan.agent import Orchestrator
from veriscan.schemas import CaseReport
from veriscan.data.generate_mock import SAMPLES_DIR

SAMPLE_CASES = [
    "case_01_consistent",
    "case_02_benign",
    "case_03_mismatch",
    "case_04_degraded",
]


def load_expected(case_id: str) -> dict:
    """Load expected.json for a sample case."""
    exp_path = SAMPLES_DIR / case_id / "expected.json"
    assert exp_path.exists(), f"expected.json not found for {case_id}: {exp_path}"
    with open(exp_path, "r") as f:
        return json.load(f)


def run_pipeline_on_case(case_id: str) -> CaseReport:
    """Run the Orchestrator on a sample case using cached tokens."""
    import cv2  # type: ignore
    case_dir = SAMPLES_DIR / case_id
    truth_path = case_dir / "truth.json"
    with open(truth_path) as f:
        truth = json.load(f)

    image_files = []
    for doc_key, dinfo in truth["documents"].items():
        img_path = case_dir / dinfo["filename"]
        img = cv2.imread(str(img_path))
        if img is None:
            # Create a dummy blank image if file not found
            import numpy as np  # type: ignore
            img = np.ones((400, 600, 3), dtype="uint8") * 255
        image_files.append((dinfo["filename"], img))

    orch = Orchestrator(default_engine="tesseract")
    report = orch.process_case(
        case_id=case_id,
        image_files=image_files,
        cache_dir=case_dir
    )
    return report


class TestCase01Consistent:
    """Case 01: Fully consistent baseline — must be GREEN with no MISMATCH."""

    def test_triage_green(self):
        report = run_pipeline_on_case("case_01_consistent")
        exp = load_expected("case_01_consistent")
        assert report.triage == exp["expected_triage"], (
            f"Expected GREEN triage, got {report.triage}. "
            f"Findings: {[(f.field, f.verdict, f.severity) for f in report.findings]}"
        )

    def test_no_mismatch_findings(self):
        report = run_pipeline_on_case("case_01_consistent")
        exp = load_expected("case_01_consistent")
        mismatches = [f for f in report.findings if f.verdict == "MISMATCH"]
        assert len(mismatches) == 0, (
            f"Expected 0 MISMATCH findings, got {len(mismatches)}: "
            f"{[(f.field, f.verdict, f.severity, f.reasons) for f in mismatches]}"
        )

    def test_no_forbidden_field_mismatches(self):
        report = run_pipeline_on_case("case_01_consistent")
        exp = load_expected("case_01_consistent")
        for fn in exp.get("forbidden_mismatch_fields", []):
            for f in report.findings:
                assert not (f.field == fn and f.verdict == "MISMATCH"), (
                    f"Field '{fn}' must not have MISMATCH verdict. "
                    f"Got: {f.verdict}, reasons: {f.reasons}"
                )

    def test_risk_score_low(self):
        report = run_pipeline_on_case("case_01_consistent")
        exp = load_expected("case_01_consistent")
        max_risk = exp.get("expected_risk_max", 0.20)
        assert report.risk_score <= max_risk, (
            f"Risk score {report.risk_score:.3f} exceeds expected max {max_risk}. "
            f"Findings: {[(f.field, f.verdict, f.severity, f.confidence) for f in report.findings]}"
        )

    def test_no_issue_date_comparison(self):
        """Issue dates must never produce findings across documents."""
        report = run_pipeline_on_case("case_01_consistent")
        for f in report.findings:
            assert f.field != "issue_date", (
                f"issue_date must never be cross-compared. Got finding: {f}"
            )


class TestCase02Benign:
    """Case 02: Benign variants — GREEN triage, only MINOR_VARIANT allowed on name."""

    def test_triage_green(self):
        report = run_pipeline_on_case("case_02_benign")
        exp = load_expected("case_02_benign")
        assert report.triage == exp["expected_triage"], (
            f"Expected GREEN, got {report.triage}. "
            f"Findings: {[(f.field, f.verdict, f.severity) for f in report.findings]}"
        )

    def test_no_high_severity_findings(self):
        report = run_pipeline_on_case("case_02_benign")
        exp = load_expected("case_02_benign")
        max_high = exp.get("max_high_severity_findings", 0)
        high_findings = [f for f in report.findings if f.severity == "HIGH" and f.verdict == "MISMATCH"]
        assert len(high_findings) <= max_high, (
            f"Expected at most {max_high} HIGH MISMATCH findings, got {len(high_findings)}: "
            f"{[(f.field, f.verdict, f.severity, f.reasons) for f in high_findings]}"
        )

    def test_no_id_number_cross_type_mismatch(self):
        """IND- vs ROLL- vs CON- must not produce MISMATCH."""
        report = run_pipeline_on_case("case_02_benign")
        id_findings = [f for f in report.findings if f.field == "id_number" and f.verdict == "MISMATCH"]
        assert len(id_findings) == 0, (
            f"Cross-type id_number must not produce MISMATCH. Got: "
            f"{[(f.field, f.verdict, f.values_raw) for f in id_findings]}"
        )

    def test_no_issue_date_findings(self):
        """Issue dates (2021, 2022, 2023) must not produce findings."""
        report = run_pipeline_on_case("case_02_benign")
        issue_date_findings = [f for f in report.findings if f.field == "issue_date"]
        assert len(issue_date_findings) == 0, (
            f"issue_date must not appear in findings. Got: {issue_date_findings}"
        )

    def test_no_address_from_marksheet(self):
        """Marksheet must not produce a garbage address extracted from roll number context."""
        report = run_pipeline_on_case("case_02_benign")
        # Check each document in the report
        for doc in report.documents:
            if doc.doc_type == "marksheet":
                if "address" in doc.fields:
                    addr_val = doc.fields["address"].value_raw
                    # A marksheet address extracted via regex_pin_context is suspicious
                    assert doc.fields["address"].method != "regex_pin_context", (
                        f"Marksheet {doc.filename} has regex_pin_context address: '{addr_val}' "
                        f"(likely garbage from roll number). Method must be 'label_proximity'."
                    )


class TestCase03Mismatch:
    """Case 03: Known mismatches — DOB and address pincode must be flagged."""

    def test_triage_red(self):
        report = run_pipeline_on_case("case_03_mismatch")
        exp = load_expected("case_03_mismatch")
        assert report.triage in ["RED", "AMBER"], (
            f"Expected RED or AMBER triage for mismatch case, got {report.triage}."
        )

    def test_dob_mismatch_detected(self):
        """DOB mismatch (Sep 20 vs Sep 23) must be detected."""
        report = run_pipeline_on_case("case_03_mismatch")
        dob_findings = [f for f in report.findings if f.field == "dob" and f.verdict == "MISMATCH"]
        assert len(dob_findings) >= 1, (
            f"DOB MISMATCH must be detected (1998-09-20 vs 1998-09-23). "
            f"Current findings: {[(f.field, f.verdict) for f in report.findings]}"
        )

    def test_address_mismatch_detected(self):
        """Address pincode mismatch (560004 vs 560504) must be detected."""
        report = run_pipeline_on_case("case_03_mismatch")
        addr_findings = [f for f in report.findings if f.field == "address" and f.verdict in ["MISMATCH", "MINOR_VARIANT"]]
        assert len(addr_findings) >= 1, (
            f"Address mismatch must be detected (pincode 560004 vs 560504). "
            f"Current findings: {[(f.field, f.verdict) for f in report.findings]}"
        )

    def test_no_spurious_id_number_mismatch(self):
        """IND- vs ROLL- vs CON- must not produce MISMATCH even in mismatch case."""
        report = run_pipeline_on_case("case_03_mismatch")
        id_findings = [f for f in report.findings if f.field == "id_number" and f.verdict == "MISMATCH"]
        assert len(id_findings) == 0, (
            f"Cross-type id_number must not be compared. Got: "
            f"{[(f.values_raw) for f in id_findings]}"
        )

    def test_no_issue_date_findings(self):
        """Issue dates must not appear in findings."""
        report = run_pipeline_on_case("case_03_mismatch")
        assert not any(f.field == "issue_date" for f in report.findings), (
            "issue_date must never appear in findings"
        )


class TestCase04Degraded:
    """Case 04: Degraded scans — no MISMATCH allowed (consistent identity, just poor quality)."""

    def test_triage_not_red(self):
        report = run_pipeline_on_case("case_04_degraded")
        exp = load_expected("case_04_degraded")
        allowed = exp.get("expected_triage_options", ["GREEN", "AMBER"])
        assert report.triage in allowed, (
            f"Degraded scan with consistent identity must be GREEN or AMBER, got {report.triage}. "
            f"Findings: {[(f.field, f.verdict, f.severity) for f in report.findings]}"
        )

    def test_no_mismatch_from_poor_scan(self):
        """Poor scan quality must not produce MISMATCH — only LOW_CONFIDENCE."""
        report = run_pipeline_on_case("case_04_degraded")
        exp = load_expected("case_04_degraded")
        max_mm = exp.get("max_mismatch_findings", 0)
        mismatches = [f for f in report.findings if f.verdict == "MISMATCH"]
        assert len(mismatches) <= max_mm, (
            f"Degraded consistent case must produce ≤{max_mm} MISMATCH findings, got {len(mismatches)}: "
            f"{[(f.field, f.verdict, f.severity, f.reasons) for f in mismatches]}"
        )

    def test_no_issue_date_findings(self):
        report = run_pipeline_on_case("case_04_degraded")
        assert not any(f.field == "issue_date" for f in report.findings), (
            "issue_date must never appear in findings"
        )


class TestFieldRegistryRules:
    """Unit tests for field semantics registry (veriscan/fields.py)."""

    def test_issue_date_never_compared(self):
        from veriscan.fields import should_compare_field
        assert not should_compare_field("issue_date", "id_card", "marksheet")
        assert not should_compare_field("issue_date", "id_card", "id_card")
        assert not should_compare_field("bill_date", "address_proof", "id_card")

    def test_cross_comparable_fields(self):
        from veriscan.fields import should_compare_field
        assert should_compare_field("name", "id_card", "marksheet")
        assert should_compare_field("dob", "id_card", "marksheet")
        assert should_compare_field("guardian_name", "id_card", "marksheet")
        assert should_compare_field("address", "id_card", "address_proof")

    def test_id_number_different_types_not_compared(self):
        from veriscan.fields import should_compare_field
        # IND- vs ROLL-: different prefixes → not comparable
        assert not should_compare_field("id_number", "id_card", "marksheet", "IND-123", "ROLL-456")
        # IND- vs CON-: different prefixes → not comparable
        assert not should_compare_field("id_number", "id_card", "address_proof", "IND-123", "CON-456")

    def test_id_number_same_type_compared(self):
        from veriscan.fields import should_compare_field
        # Two id_cards with IND- prefix → comparable
        assert should_compare_field("id_number", "id_card", "id_card", "IND-123", "IND-456")

    def test_field_expected_on_doc_type(self):
        from veriscan.fields import is_field_expected_on_doc_type
        assert is_field_expected_on_doc_type("name", "id_card")
        assert is_field_expected_on_doc_type("dob", "marksheet")
        assert is_field_expected_on_doc_type("address", "address_proof")
        # Guardian name not expected on utility bill
        assert not is_field_expected_on_doc_type("guardian_name", "address_proof")
        assert not is_field_expected_on_doc_type("guardian_name", "bank_statement")


class TestDeduplication:
    """Test that suggested reviewer questions are deduplicated."""

    def test_no_duplicate_questions(self):
        from veriscan.narrative import generate_narrative
        from veriscan.schemas import Finding, ConsistentItem

        # Create 3 identical MISMATCH findings on DOB (like case_03 with 3 docs)
        findings = []
        for i in range(3):
            findings.append(Finding(
                id=f"dob_f{i}",
                field="dob",
                docs=[f"doc_{i}", f"doc_{i+1}"],
                values_raw={f"doc_{i}": "01/01/1990", f"doc_{i+1}": "01/01/1991"},
                values_norm={f"doc_{i}": "1990-01-01", f"doc_{i+1}": "1991-01-01"},
                verdict="MISMATCH",
                severity="HIGH",
                similarity=0.7,
                confidence=0.9,
            ))

        _, questions = generate_narrative(findings, [], 0.9, "RED")
        # Questions should be deduplicated
        assert len(questions) == len(set(questions)), (
            f"Questions contain duplicates: {questions}"
        )
