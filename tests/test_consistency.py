"""Unit tests for VeriScan consistency engine, comparators, and gating rules."""

from veriscan.schemas import DocumentField, Finding
from veriscan.consistency import (
    compare_names, compare_dates, compare_addresses, compare_id_numbers,
    compare_document_fields
)
from veriscan.scoring import calculate_case_risk


class TestConsistencyComparators:
    """Test individual field comparators."""

    def test_name_exact_match(self):
        verdict, sim, reasons, rules = compare_names(
            "Rohit Sharma", "Rohit Sharma", "rohit sharma", "rohit sharma"
        )
        assert verdict == "MATCH"
        assert sim == 1.0
        assert "RULE_NAME_EXACT_MATCH" in rules

    def test_name_initial_expansion(self):
        verdict, sim, reasons, rules = compare_names(
            "R. Sharma", "Rohit Sharma", "r sharma", "rohit sharma"
        )
        assert verdict == "MATCH"
        assert sim >= 0.92
        assert any("RULE_NAME_INITIAL_EXPAND" in r for r in rules)

    def test_name_order_inverted(self):
        verdict, sim, reasons, rules = compare_names(
            "Sharma Rohit", "Rohit Sharma", "rohit sharma", "rohit sharma"
        )
        assert verdict == "MATCH"
        assert sim == 1.0

    def test_name_minor_variant(self):
        # Slightly different spelling: e.g. "Venkatesh" vs "Venkatesan" before canonicalization or similar
        verdict, sim, reasons, rules = compare_names(
            "Ramesh K", "Ramesh Kumar", "k ramesh", "kumar ramesh"
        )
        assert verdict in ["MATCH", "MINOR_VARIANT"]
        assert sim >= 0.75

    def test_name_mismatch(self):
        verdict, sim, reasons, rules = compare_names(
            "Rohit Sharma", "Ananya Verma", "rohit sharma", "ananya verma"
        )
        assert verdict == "MISMATCH"
        assert sim < 0.75
        assert "RULE_NAME_SUBSTANTIAL_MISMATCH" in rules

    def test_date_exact_match(self):
        verdict, sim, reasons, rules = compare_dates(
            "12/03/2004", "12 Mar 2004", "2004-03-12", "2004-03-12"
        )
        assert verdict == "MATCH"
        assert sim == 1.0
        assert "RULE_DATE_EXACT_MATCH" in rules

    def test_date_day_month_swap_subtype(self):
        verdict, sim, reasons, rules = compare_dates(
            "12/03/2004", "03/12/2004", "2004-03-12", "2004-12-03"
        )
        assert verdict == "MISMATCH"
        assert "RULE_DATE_SUBTYPE_DAY_MONTH_SWAP" in rules

    def test_date_single_digit_typo_subtype(self):
        verdict, sim, reasons, rules = compare_dates(
            "12/03/2004", "15/03/2004", "2004-03-12", "2004-03-15"
        )
        assert verdict == "MISMATCH"
        assert "RULE_DATE_SUBTYPE_SINGLE_DIGIT_TYPO" in rules

    def test_date_year_off_subtype(self):
        verdict, sim, reasons, rules = compare_dates(
            "12/03/2004", "12/03/2005", "2004-03-12", "2005-03-12"
        )
        assert verdict == "MISMATCH"
        assert "RULE_DATE_SUBTYPE_YEAR_OFF" in rules

    def test_address_exact_match(self):
        verdict, sim, reasons, rules = compare_addresses(
            "12 MG Road, Bengaluru 560001", "12 MG Road, Bengaluru 560001",
            "12 mg road bengaluru 560001", "12 mg road bengaluru 560001"
        )
        assert verdict == "MATCH"
        assert sim == 1.0

    def test_address_pincode_mismatch_low_severity(self):
        fa = DocumentField(
            doc_id="doc_1", name="address",
            value_raw="MG Road, Bengaluru 560001", value_norm="mg road bengaluru 560001",
            ocr_conf=0.95
        )
        fb = DocumentField(
            doc_id="doc_2", name="address",
            value_raw="MG Road, Bengaluru 560099", value_norm="mg road bengaluru 560099",
            ocr_conf=0.95
        )
        finding = compare_document_fields("address", fa, fb)
        # In our configuration, pincode-only mismatch yields LOW severity
        assert finding.severity == "LOW"

    def test_id_exact_match(self):
        verdict, sim, reasons, rules = compare_id_numbers(
            "IND-94821092", "IND 9482 1092", "IND94821092", "IND94821092"
        )
        assert verdict == "MATCH"
        assert sim == 1.0

    def test_id_mismatch(self):
        verdict, sim, reasons, rules = compare_id_numbers(
            "IND-94821092", "IND-12345678", "IND94821092", "IND12345678"
        )
        assert verdict == "MISMATCH"
        assert sim == 0.0


class TestConfidenceGating:
    """Test that OCR confidence < 0.60 forces verdict to LOW_CONFIDENCE."""

    def test_low_ocr_confidence_gating(self):
        # Even if names are completely different, if OCR is low, it must be LOW_CONFIDENCE, never MISMATCH
        fa = DocumentField(
            doc_id="doc_1", name="name",
            value_raw="Rohit Sharma", value_norm="rohit sharma",
            ocr_conf=0.52  # Below 0.60 threshold
        )
        fb = DocumentField(
            doc_id="doc_2", name="name",
            value_raw="Ananya Verma", value_norm="ananya verma",
            ocr_conf=0.92
        )
        finding = compare_document_fields("name", fa, fb)
        assert finding.verdict == "LOW_CONFIDENCE"
        assert "RULE_OCR_CONFIDENCE_GATED" in finding.rule_ids


class TestRiskScoring:
    """Test heuristic case risk calculation."""

    def test_empty_findings_is_green(self):
        risk, triage = calculate_case_risk([])
        assert risk == 0.0
        assert triage == "GREEN"

    def test_only_matches_is_green(self):
        # Findings with MATCH should not elevate risk
        f = Finding(
            id="f1", field="name", docs=["d1", "d2"],
            values_raw={"d1": "A", "d2": "A"}, values_norm={"d1": "A", "d2": "A"},
            verdict="MATCH", similarity=1.0, severity="HIGH", confidence=1.0
        )
        risk, triage = calculate_case_risk([f])
        assert risk == 0.0
        assert triage == "GREEN"

    def test_high_severity_mismatch_elevates_to_red(self):
        f = Finding(
            id="f1", field="dob", docs=["d1", "d2"],
            values_raw={"d1": "1998-05-12", "d2": "2002-11-20"},
            values_norm={"d1": "1998-05-12", "d2": "2002-11-20"},
            verdict="MISMATCH", similarity=0.0, severity="HIGH", confidence=0.95
        )
        risk, triage = calculate_case_risk([f])
        assert risk > 0.60
        assert triage == "RED"
