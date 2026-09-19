"""Unit tests for VeriScan normalization module."""

from veriscan.normalize import (
    normalize_name, normalize_date, normalize_address,
    extract_address_components, normalize_id_number
)


class TestNameNormalization:
    """Test comprehensive Indian name normalization rules."""

    def test_exact_name(self):
        norm, rules = normalize_name("Rohit Sharma")
        assert norm == "rohit sharma"
        assert len(rules) >= 1

    def test_name_casefold(self):
        norm, rules = normalize_name("ROHIT SHARMA")
        assert norm == "rohit sharma"
        assert "RULE_NAME_CASEFOLD" in rules

    def test_name_order_independence(self):
        norm1, rules1 = normalize_name("Sharma Rohit")
        norm2, rules2 = normalize_name("Rohit Sharma")
        assert norm1 == norm2
        assert "RULE_NAME_TOKEN_ORDER_INDEPENDENT" in rules1 or "RULE_NAME_TOKEN_ORDER_INDEPENDENT" in rules2

    def test_strip_honorific_mr(self):
        norm, rules = normalize_name("Mr. Rohit Sharma")
        assert norm == "rohit sharma"
        assert "RULE_NAME_STRIP_HONORIFIC_MR" in rules

    def test_strip_honorific_dr(self):
        norm, rules = normalize_name("Dr. Priya Patel")
        assert norm == "patel priya"
        assert "RULE_NAME_STRIP_HONORIFIC_DR" in rules

    def test_strip_honorific_smt(self):
        norm, rules = normalize_name("Smt. Sunita Verma")
        assert norm == "sunita verma"
        assert "RULE_NAME_STRIP_HONORIFIC_SMT" in rules

    def test_strip_honorific_shri(self):
        norm, rules = normalize_name("Shri Aarav Kumar")
        assert norm == "aarav kumar"
        assert "RULE_NAME_STRIP_HONORIFIC_SHRI" in rules

    def test_strip_honorific_kumari(self):
        norm, rules = normalize_name("Kumari Sneha Reddy")
        assert norm == "reddy sneha"
        assert "RULE_NAME_STRIP_HONORIFIC_KUMARI" in rules

    def test_strip_relation_so(self):
        norm, rules = normalize_name("Vikram Singh S/O Mohan Singh")
        assert "singh" in norm
        assert "vikram" in norm
        assert any("RULE_NAME_STRIP_RELATION" in r for r in rules)

    def test_strip_relation_do(self):
        norm, rules = normalize_name("Ananya Rao D/O Ramesh Rao")
        assert "ananya" in norm
        assert "rao" in norm
        assert any("RULE_NAME_STRIP_RELATION" in r for r in rules)

    def test_strip_relation_wo(self):
        norm, rules = normalize_name("Kavita Devi W/O Suresh Kumar")
        assert "kavita" in norm
        assert any("RULE_NAME_STRIP_RELATION" in r for r in rules)

    def test_variant_mohammed_md(self):
        norm1, rules1 = normalize_name("Md. Irfan")
        norm2, rules2 = normalize_name("Mohammed Irfan")
        assert norm1 == norm2
        assert any("RULE_NAME_VARIANT" in r for r in rules1)

    def test_variant_mohd(self):
        norm1, rules1 = normalize_name("Mohd Irfan")
        norm2, rules2 = normalize_name("Mohammed Irfan")
        assert norm1 == norm2

    def test_variant_venkatesh_venkatesan(self):
        norm1, _ = normalize_name("Venkatesh Iyer")
        norm2, _ = normalize_name("Venkatesan Iyer")
        assert norm1 == norm2

    def test_variant_laxmi_lakshmi(self):
        norm1, _ = normalize_name("Laxmi Bai")
        norm2, _ = normalize_name("Lakshmi Bai")
        assert norm1 == norm2

    def test_variant_agarwal_agrawal(self):
        norm1, _ = normalize_name("Rohan Agarwal")
        norm2, _ = normalize_name("Rohan Agrawal")
        assert norm1 == norm2

    def test_variant_sharma_sarma(self):
        norm1, _ = normalize_name("Ajay Sharma")
        norm2, _ = normalize_name("Ajay Sarma")
        assert norm1 == norm2

    def test_variant_patel_patil(self):
        norm1, _ = normalize_name("Kiran Patel")
        norm2, _ = normalize_name("Kiran Patil")
        assert norm1 == norm2

    def test_variant_choudhary_chaudhary(self):
        norm1, _ = normalize_name("Sanjay Choudhary")
        norm2, _ = normalize_name("Sanjay Chaudhary")
        assert norm1 == norm2

    def test_unicode_nfkd_normalization(self):
        raw = "Róhït Shármá"
        norm, rules = normalize_name(raw)
        assert "rohit" in norm
        assert "sharma" in norm
        assert "RULE_NAME_UNICODE_NFKD" in rules

    def test_empty_name(self):
        norm, rules = normalize_name("")
        assert norm == ""
        assert "RULE_EMPTY_NAME" in rules


class TestDateNormalization:
    """Test multi-format date parsing and ambiguity detection."""

    def test_iso_date(self):
        norm, rules = normalize_date("2004-03-12")
        assert norm == "2004-03-12"
        assert "RULE_DATE_ISO_PARSE" in rules

    def test_slash_dmy(self):
        norm, rules = normalize_date("12/03/2004")
        assert norm == "2004-03-12"
        assert "RULE_DATE_DAY_FIRST_PARSE" in rules

    def test_dash_dmy(self):
        norm, rules = normalize_date("12-03-2004")
        assert norm == "2004-03-12"
        assert "RULE_DATE_DAY_FIRST_PARSE" in rules

    def test_dot_dmy(self):
        norm, rules = normalize_date("12.03.2004")
        assert norm == "2004-03-12"
        assert "RULE_DATE_DAY_FIRST_PARSE" in rules

    def test_named_month_short(self):
        norm, rules = normalize_date("12 Mar 2004")
        assert norm == "2004-03-12"
        assert "RULE_DATE_NAMED_MONTH_PARSE" in rules

    def test_named_month_full(self):
        norm, rules = normalize_date("12 March 2004")
        assert norm == "2004-03-12"
        assert "RULE_DATE_NAMED_MONTH_PARSE" in rules

    def test_two_digit_year_2000s(self):
        norm, rules = normalize_date("12-03-04")
        assert norm == "2004-03-12"
        assert "RULE_DATE_TWO_DIGIT_YEAR_EXPANDED" in rules

    def test_two_digit_year_1900s(self):
        norm, rules = normalize_date("12-03-98")
        assert norm == "1998-03-12"
        assert "RULE_DATE_TWO_DIGIT_YEAR_EXPANDED" in rules

    def test_ambiguous_day_month(self):
        # 05/06/2002: both <= 12 and d != m
        norm, rules = normalize_date("05/06/2002")
        assert norm == "2002-06-05"
        assert "RULE_DATE_AMBIGUOUS_DAY_MONTH" in rules

    def test_unambiguous_day_gt_12(self):
        # 25/06/2002: day is 25 (> 12) so unambiguous
        norm, rules = normalize_date("25/06/2002")
        assert norm == "2002-06-25"
        assert "RULE_DATE_AMBIGUOUS_DAY_MONTH" not in rules

    def test_strip_dob_label(self):
        norm, _ = normalize_date("DOB: 15/08/2000")
        assert norm == "2000-08-15"

    def test_strip_date_of_birth_label(self):
        norm, _ = normalize_date("Date of Birth: 22-11-1999")
        assert norm == "1999-11-22"

    def test_invalid_date_returns_raw(self):
        norm, rules = normalize_date("NotADateString")
        assert "RULE_DATE_PARSE_FAILED" in rules

    def test_empty_date(self):
        norm, rules = normalize_date("")
        assert norm == ""
        assert "RULE_DATE_EMPTY" in rules


class TestAddressNormalization:
    """Test address normalization, abbreviation expansion, and component extraction."""

    def test_expand_rd_to_road(self):
        norm, rules = normalize_address("12 MG Rd, Bengaluru")
        assert "road" in norm
        assert any("EXPAND_RD" in r for r in rules)

    def test_expand_st_to_street(self):
        norm, rules = normalize_address("44 Nehru St, Chennai")
        assert "street" in norm
        assert any("EXPAND_ST" in r for r in rules)

    def test_expand_nr_to_near(self):
        norm, rules = normalize_address("Plot 5 nr Bus Stand, Pune")
        assert "near" in norm
        assert any("EXPAND_NR" in r for r in rules)

    def test_expand_apt_to_apartment(self):
        norm, rules = normalize_address("Apt 302, Palm Heights, Mumbai")
        assert "apartment" in norm
        assert any("EXPAND_APT" in r for r in rules)

    def test_expand_blr_to_bengaluru(self):
        norm, rules = normalize_address("Koramangala, Blr - 560034")
        assert "bengaluru" in norm
        assert any("EXPAND_BLR" in r for r in rules)

    def test_expand_bangalore_to_bengaluru(self):
        norm, rules = normalize_address("Indiranagar, Bangalore 560038")
        assert "bengaluru" in norm
        assert any("EXPAND_BANGALORE" in r for r in rules)

    def test_expand_bom_to_mumbai(self):
        norm, rules = normalize_address("Bandra West, Bom - 400050")
        assert "mumbai" in norm

    def test_expand_no_to_number(self):
        norm, rules = normalize_address("Flat No. 12, Park Soc")
        assert "number" in norm

    def test_extract_pincode_rule(self):
        norm, rules = normalize_address("MG Road, Bengaluru 560001")
        assert any("RULE_ADDR_EXTRACTED_PIN_560001" in r for r in rules)

    def test_address_component_split(self):
        raw = "No 45, Brigade Road, Bengaluru - 560025"
        comp, rules = extract_address_components(raw)
        assert comp["pincode"] == "560025"
        assert comp["city"] == "bengaluru"
        assert "45" in comp["house_number"]
        assert "brigade road" in comp["street_locality"]
        assert "RULE_ADDR_COMPONENT_SPLIT" in rules

    def test_empty_address(self):
        norm, rules = normalize_address("")
        assert norm == ""
        assert "RULE_ADDR_EMPTY" in rules


class TestIdNormalization:
    """Test ID/Number separator stripping and OCR confusion fixes."""

    def test_strip_spaces(self):
        norm, rules = normalize_id_number("IND 9482 1092")
        assert norm == "IND94821092"
        assert "RULE_ID_STRIP_SEPARATORS" in rules

    def test_strip_hyphens(self):
        norm, rules = normalize_id_number("DL-1420-1100-1234")
        assert norm == "DL142011001234"
        assert "RULE_ID_STRIP_SEPARATORS" in rules

    def test_strip_slashes(self):
        norm, rules = normalize_id_number("ROLL/2022/9812")
        assert norm == "ROLL20229812"
        assert "RULE_ID_STRIP_SEPARATORS" in rules

    def test_ocr_confusion_letter_o_to_zero(self):
        # In numeric-only context, letter O replaced with 0
        norm, rules = normalize_id_number("98O45O12", is_numeric_only=True)
        assert norm == "98045012"
        assert "RULE_ID_OCR_NUMERIC_CONFUSION_FIX" in rules

    def test_ocr_confusion_letter_i_l_to_one(self):
        norm, rules = normalize_id_number("9I45l012", is_numeric_only=True)
        assert norm == "91451012"
        assert "RULE_ID_OCR_NUMERIC_CONFUSION_FIX" in rules

    def test_ocr_confusion_letter_s_to_five(self):
        norm, rules = normalize_id_number("98S45S12", is_numeric_only=True)
        assert norm == "98545512"
        assert "RULE_ID_OCR_NUMERIC_CONFUSION_FIX" in rules

    def test_ocr_confusion_letter_b_to_eight(self):
        norm, rules = normalize_id_number("9B45B012", is_numeric_only=True)
        assert norm == "98458012"
        assert "RULE_ID_OCR_NUMERIC_CONFUSION_FIX" in rules

    def test_no_confusion_fix_if_not_numeric_only(self):
        # In alphanumeric ID, letter O should NOT be changed
        norm, rules = normalize_id_number("PAN98O45", is_numeric_only=False)
        assert norm == "PAN98O45"
        assert "RULE_ID_OCR_NUMERIC_CONFUSION_FIX" not in rules

    def test_empty_id(self):
        norm, rules = normalize_id_number("")
        assert norm == ""
        assert "RULE_ID_EMPTY" in rules
