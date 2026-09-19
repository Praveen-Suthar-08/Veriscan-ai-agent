"""Tests for layout-agnostic field extraction and source_text_span hardening (Key Feature 7)."""

import pytest
from pathlib import Path
from veriscan.ocr import load_tokens_cache
from veriscan.extract import extract_fields_from_tokens
from veriscan.schemas import DocumentField

SAMPLES_DIR = Path(__file__).parent.parent / "data" / "samples"


def test_source_text_span_populated_on_all_bundled_samples():
    """
    Step 1 Test:
    For every field extracted from every bundled sample:
    a. Assert source_text_span is not None and is a non-empty string.
    b. Assert source_text_span is a substring of the concatenated OCR text for that document.
    c. Assert source_line_bbox is a list of 4 numbers.
    d. Assert source_line_conf is between 0.0 and 1.0.
    e. Assert Field.evidence_line() returns a string containing doc_name, raw span and norm value.
    """
    sample_cases = [
        "case_01_consistent", "case_02_benign", "case_03_mismatch",
        "case_04_degraded", "case_05_name_mismatch", "case_06_address_mismatch"
    ]

    for case_id in sample_cases:
        case_dir = SAMPLES_DIR / case_id
        assert case_dir.exists(), f"Sample case {case_id} not found"

        token_files = list(case_dir.glob("*_tokens.json"))
        assert len(token_files) >= 2, f"Expected token files in {case_id}"

        for tf in token_files:
            tokens = load_tokens_cache(tf)
            assert tokens is not None and len(tokens) > 0

            # Document name from filename stem
            doc_name = tf.stem.replace("_tokens", ".png")
            doc_type = "id_card" if "id_card" in tf.stem else ("marksheet" if "marksheet" in tf.stem else "utility_bill")

            fields = extract_fields_from_tokens(tokens, doc_id=doc_name, doc_type=doc_type)
            assert len(fields) >= 2, f"Expected extracted fields for {tf.name}"

            full_doc_text = " ".join(t.text for t in tokens)

            for f_name, field in fields.items():
                # a. source_text_span is not None and is a non-empty string
                assert field.source_text_span is not None, f"Field {f_name} in {tf.name} has source_text_span=None"
                assert len(field.source_text_span.strip()) > 0

                # b. source_text_span is a substring of the concatenated OCR text
                # (or every token of source_text_span exists in full_doc_text)
                for w in field.source_text_span.split():
                    assert w in full_doc_text, f"Token '{w}' from source_text_span not found in OCR text for {tf.name}"

                # c. source_line_bbox is a list of 4 numbers
                assert isinstance(field.source_line_bbox, list), f"Field {f_name} source_line_bbox not a list"
                assert len(field.source_line_bbox) == 4
                assert all(isinstance(v, (int, float)) for v in field.source_line_bbox)

                # d. source_line_conf is between 0.0 and 1.0
                assert field.source_line_conf is not None
                assert 0.0 <= field.source_line_conf <= 1.0

                # e. Field.evidence_line() returns formatted string
                ev_line = field.evidence_line(doc_name)
                assert doc_name in ev_line
                assert field.source_text_span in ev_line
                assert field.value_norm in ev_line
