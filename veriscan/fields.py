"""Field Semantics Registry for VeriScan.

Defines which fields are cross-comparable, which are document-type-local,
and which carry date-sanity-only checks (never compared for equality).

Structure:
- CROSS_COMPARABLE: fields compared across all document types
- DOC_TYPE_ID_FIELDS: typed identifier fields — compared ONLY within same type
- DATE_SANITY_ONLY: date fields never compared for equality (only sanity checks)
- EXPECTED_FIELDS_BY_TYPE: fields expected per document type (for MISSING logic)
"""

from __future__ import annotations
from typing import Dict, Set, Optional

# ──────────────────────────────────────────────
# Cross-comparable fields (compared across docs)
# ──────────────────────────────────────────────
CROSS_COMPARABLE: Set[str] = {"name", "guardian_name", "dob", "address"}

# ──────────────────────────────────────────────────────────────────────────────
# Typed identifier fields: each is specific to a document type.
# ONLY compared between two documents of the SAME type (or matching ID scheme).
# Key = canonical field name; Value = set of doc types that carry it.
# ──────────────────────────────────────────────────────────────────────────────
DOC_TYPE_ID_FIELDS: Dict[str, Set[str]] = {
    "id_number": {"id_card"},           # IND-..., DL-..., PAN-...
    "roll_no": {"marksheet"},            # ROLL-...
    "consumer_no": {"address_proof"},    # CON-...
    "account_no": {"bank_statement"},    # account numbers
    "application_ref": {"application_form"},
}

# ──────────────────────────────────────────────────────────────────────────────
# Normalised identifier prefixes that signal the document type.
# If an extracted id_number has one of these prefixes, we know its type.
# ──────────────────────────────────────────────────────────────────────────────
ID_PREFIX_TO_DOC_TYPE: Dict[str, str] = {
    "IND": "id_card",
    "DL":  "id_card",
    "PAN": "id_card",
    "PAN": "id_card",
    "ROLL": "marksheet",
    "CON": "address_proof",
    "ACC": "bank_statement",
    "ACCT": "bank_statement",
    "APP": "application_form",
    "REF": "application_form",
}

# ──────────────────────────────────────────────────────────────────────────────
# Date fields that carry SANITY checks only — NEVER compared for equality.
# Different documents legitimately have different issue/bill dates.
# ──────────────────────────────────────────────────────────────────────────────
DATE_SANITY_ONLY: Set[str] = {
    "issue_date",
    "bill_date",
    "statement_date",
    "valid_until",
    "expiry_date",
}

# ──────────────────────────────────────────────────────────────────────────────
# Expected fields per document type.
# Used to decide if a MISSING finding is legitimate (field expected on doc_b).
# ──────────────────────────────────────────────────────────────────────────────
EXPECTED_FIELDS_BY_TYPE: Dict[str, Set[str]] = {
    "id_card":          {"name", "guardian_name", "dob", "address", "id_number", "issue_date"},
    "marksheet":        {"name", "guardian_name", "dob", "roll_no", "issue_date"},
    "address_proof":    {"name", "address", "consumer_no", "bill_date"},
    "utility_bill":     {"name", "address", "consumer_no", "bill_date"},
    "bank_statement":   {"name", "address", "account_no", "statement_date"},
    "application_form": {"name", "guardian_name", "dob", "address", "application_ref"},
    "unknown":          {"name"},
}

# All cross-comparable fields unified
ALL_COMPARABLE: Set[str] = CROSS_COMPARABLE


def get_id_prefix(id_value: str) -> Optional[str]:
    """Extract the alphabetic prefix from an identifier string like 'IND-12345' or 'ROLL-98765'."""
    import re
    m = re.match(r"^([A-Z]{2,6})[-\s]?[0-9]", id_value.upper().strip())
    return m.group(1) if m else None


def infer_id_doc_type(id_value: str) -> Optional[str]:
    """Infer the document type from an identifier's prefix."""
    prefix = get_id_prefix(id_value)
    if prefix:
        return ID_PREFIX_TO_DOC_TYPE.get(prefix)
    return None


def are_id_numbers_comparable(
    id_a: str, id_b: str,
    doc_type_a: str, doc_type_b: str
) -> bool:
    """
    Return True only when two id_number values can legitimately be compared.
    Criteria:
      1. Same document type on both sides (two id_cards), OR
      2. Both IDs share the same normalised prefix (e.g. IND- vs IND-)
    False for IND- vs ROLL-, IND- vs CON-, etc.
    """
    # Same doc type that is an ID-carrying type
    id_carrying_types = {"id_card", "application_form"}
    if doc_type_a == doc_type_b and doc_type_a in id_carrying_types:
        return True

    # Same prefix
    prefix_a = get_id_prefix(id_a)
    prefix_b = get_id_prefix(id_b)
    if prefix_a and prefix_b and prefix_a == prefix_b:
        return True

    return False


def should_compare_field(
    field_name: str,
    doc_type_a: str,
    doc_type_b: str,
    value_a: str = "",
    value_b: str = "",
) -> bool:
    """
    Return True if two field values from two different documents should be compared.

    Rules:
    - DATE_SANITY_ONLY fields: never compared for equality → always False here
    - CROSS_COMPARABLE fields: always True
    - id_number: only if comparable by type/prefix
    - Other typed identifier fields: only if both docs are the right type
    """
    if field_name in DATE_SANITY_ONLY:
        return False

    if field_name in CROSS_COMPARABLE:
        return True

    if field_name == "id_number":
        return are_id_numbers_comparable(value_a, value_b, doc_type_a, doc_type_b)

    # Check typed identifier fields
    for fn, types in DOC_TYPE_ID_FIELDS.items():
        if field_name == fn:
            return (doc_type_a in types and doc_type_b in types)

    # Unknown field: skip to be conservative
    return False


def is_field_expected_on_doc_type(field_name: str, doc_type: str) -> bool:
    """Return True if the field is expected to appear on this document type."""
    expected = EXPECTED_FIELDS_BY_TYPE.get(doc_type, EXPECTED_FIELDS_BY_TYPE["unknown"])
    return field_name in expected
