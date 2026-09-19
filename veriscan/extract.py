"""Layout-agnostic field extraction engine for VeriScan.

Uses fuzzy label-proximity matching, regular expressions, and type validators
to locate and extract key fields across arbitrary document layouts.
"""

from __future__ import annotations
import re
from typing import List, Dict, Tuple, Optional, Any
import numpy as np  # type: ignore

try:
    from rapidfuzz import fuzz  # type: ignore
except ImportError:
    fuzz = None

from veriscan.schemas import Token, DocumentField
from veriscan.normalize import (
    normalize_name, normalize_date, normalize_address, normalize_id_number
)

# Canonical label synonyms for layout-agnostic matching
LABEL_SYNONYMS: Dict[str, List[str]] = {
    "name": [
        "name", "full name", "applicant name", "candidate name",
        "student name", "account holder", "consumer name", "name of holder", "holder name"
    ],
    "guardian_name": [
        "father's name", "father name", "father", "guardian name",
        "guardian", "s/o", "d/o", "w/o", "parent's name", "husband name"
    ],
    "dob": [
        "date of birth", "dob", "birth date", "d.o.b", "born on", "birth date"
    ],
    "address": [
        "address", "residential address", "permanent address", "present address",
        "service address", "billing address", "premises", "residence"
    ],
    "id_number": [
        "id no", "id number", "roll no", "roll number", "account no",
        "account number", "consumer no", "card no", "registration no",
        "reg no", "unique id", "aadhaar no", "pan no", "voter id"
    ],
    "issue_date": [
        "issue date", "date of issue", "issued on", "bill date", "date"
    ]
}


def _find_matching_label(token_text: str) -> Optional[str]:
    """Check if token string matches any target field label."""
    clean = re.sub(r"[^\w\s]", "", token_text.lower()).strip()
    if not clean:
        return None

    for field, synonyms in LABEL_SYNONYMS.items():
        for syn in synonyms:
            if clean == syn:
                return field
            if fuzz and fuzz.ratio(clean, syn) >= 88:
                return field
    return None


def _calculate_bbox_union(tokens: List[Token]) -> Optional[List[int]]:
    """Compute bounding box covering multiple tokens."""
    if not tokens:
        return None
    x1s = [t.bbox[0] for t in tokens if t.bbox]
    y1s = [t.bbox[1] for t in tokens if t.bbox]
    x2s = [t.bbox[0] + t.bbox[2] for t in tokens if t.bbox]
    y2s = [t.bbox[1] + t.bbox[3] for t in tokens if t.bbox]
    if not x1s:
        return None
    x = min(x1s)
    y = min(y1s)
    w = max(x2s) - x
    h = max(y2s) - y
    return [x, y, w, h]


def _get_line_context(
    val_tokens: List[Token],
    all_tokens: List[Token],
    fallback_line_tokens: Optional[List[Token]] = None
) -> Tuple[Optional[str], Optional[List[int]], Optional[float]]:
    """
    Given value tokens and all document tokens:
    1. Find all tokens sharing the same line_id(s) as the value tokens.
    2. Concatenate their text in reading order -> source_text_span.
    3. Compute the union bounding box -> source_line_bbox.
    4. Compute mean confidence of those tokens -> source_line_conf.
    """
    if not val_tokens:
        if fallback_line_tokens:
            sorted_line = sorted(fallback_line_tokens, key=lambda t: t.bbox[0])
            span = " ".join([t.text for t in sorted_line]).strip()
            bbox = _calculate_bbox_union(sorted_line)
            conf = float(np.mean([t.conf for t in sorted_line])) if sorted_line else None
            return (span if span else None), bbox, (round(conf, 3) if conf is not None else None)
        return None, None, None

    # Identify the line_ids of val_tokens
    line_ids = {t.line_id for t in val_tokens if hasattr(t, "line_id")}
    
    # Collect all tokens from the document having any of these line_ids
    line_tokens = [t for t in all_tokens if getattr(t, "line_id", None) in line_ids]
    
    if not line_tokens:
        line_tokens = fallback_line_tokens or val_tokens

    # Sort tokens in reading order (by y then x)
    sorted_line = sorted(line_tokens, key=lambda t: (t.bbox[1] // 15, t.bbox[0]))
    span = " ".join([t.text for t in sorted_line]).strip()
    bbox = _calculate_bbox_union(sorted_line)
    conf = float(np.mean([t.conf for t in sorted_line])) if sorted_line else None
    
    return (span if span else None), bbox, (round(conf, 3) if conf is not None else None)


def extract_fields_from_tokens(
    tokens: List[Token],
    doc_id: str,
    doc_type: str = "unknown"
) -> Dict[str, DocumentField]:
    """
    Extract key document fields using spatial proximity and semantic regex rules.
    Populates source_text_span, source_line_bbox, and source_line_conf for every extracted field.
    """
    fields: Dict[str, DocumentField] = {}
    if not tokens:
        return fields

    # Group tokens by vertical lines (proximity on y-axis)
    lines: List[List[Token]] = []
    sorted_tokens = sorted(tokens, key=lambda t: (t.bbox[1] // 15, t.bbox[0]))

    current_line: List[Token] = []
    last_y = -1
    for t in sorted_tokens:
        cy = t.bbox[1]
        if last_y == -1 or abs(cy - last_y) < 18:
            current_line.append(t)
            last_y = cy
        else:
            if current_line:
                lines.append(sorted(current_line, key=lambda x: x.bbox[0]))
            current_line = [t]
            last_y = cy
    if current_line:
        lines.append(sorted(current_line, key=lambda x: x.bbox[0]))

    # Full text for regex fallback
    full_text = " ".join([t.text for t in tokens])

    # 1. Label Proximity Extraction
    for line_idx, line in enumerate(lines):
        line_str = " ".join([t.text for t in line])
        for token_idx, tok in enumerate(line):
            # Check 1-to-3 token n-grams for label match
            for n in [1, 2, 3]:
                if token_idx + n <= len(line):
                    ngram = " ".join([t.text for t in line[token_idx:token_idx + n]])
                    matched_field = _find_matching_label(ngram)
                    if matched_field and matched_field not in fields:
                        # Value is either to the right on the same line, or on the immediate next line below
                        val_tokens = line[token_idx + n:]
                        target_line = line
                        # Clean separator tokens like ':' or '-'
                        while val_tokens and val_tokens[0].text in [":", "-", ";", "="]:
                            val_tokens = val_tokens[1:]

                        if not val_tokens and line_idx + 1 < len(lines):
                            # Look on next line
                            next_line = lines[line_idx + 1]
                            val_tokens = next_line
                            target_line = next_line

                        if val_tokens:
                            # Truncate if another label appears in val_tokens
                            filtered_val_tokens = []
                            for vt in val_tokens:
                                if _find_matching_label(vt.text):
                                    break
                                filtered_val_tokens.append(vt)

                            if filtered_val_tokens:
                                val_raw = " ".join([vt.text for vt in filtered_val_tokens]).strip()
                                # Clean leading punctuation
                                val_raw = re.sub(r"^[:\-\s]+", "", val_raw).strip()
                                if val_raw:
                                    avg_conf = float(np.mean([vt.conf for vt in filtered_val_tokens]))
                                    bbox = _calculate_bbox_union(filtered_val_tokens)

                                    # Compute source_text_span, source_line_bbox, source_line_conf
                                    src_span, src_lbbox, src_lconf = _get_line_context(
                                        filtered_val_tokens, tokens, fallback_line_tokens=target_line
                                    )

                                    # Normalize according to field type
                                    if matched_field == "name" or matched_field == "guardian_name":
                                        val_norm, rules = normalize_name(val_raw)
                                    elif matched_field in ["dob", "issue_date"]:
                                        val_norm, rules = normalize_date(val_raw)
                                    elif matched_field == "address":
                                        val_norm, rules = normalize_address(val_raw)
                                    else:
                                        val_norm, rules = normalize_id_number(val_raw)

                                    fields[matched_field] = DocumentField(
                                        doc_id=doc_id,
                                        name=matched_field,
                                        value_raw=val_raw,
                                        value_norm=val_norm,
                                        ocr_conf=round(avg_conf, 3),
                                        bbox=bbox,
                                        method="label_proximity",
                                        rules_applied=rules,
                                        source_text_span=src_span,
                                        source_line_bbox=src_lbbox,
                                        source_line_conf=src_lconf
                                    )

    # 2. Regex fallback for DOB if not captured
    if "dob" not in fields:
        # Common date patterns
        date_matches = list(re.finditer(
            r"\b(\d{1,2}[-/. ](?:\d{1,2}|[A-Za-z]{3,9})[-/. ]\d{2,4})\b", full_text
        ))
        for dm in date_matches:
            raw_d = dm.group(1).strip()
            norm_d, rules = normalize_date(raw_d)
            if norm_d and "RULE_DATE_PARSE_FAILED" not in rules:
                matching_tokens = [t for t in tokens if t.text in raw_d or raw_d in t.text]
                avg_conf = float(np.mean([t.conf for t in matching_tokens])) if matching_tokens else 0.85
                src_span, src_lbbox, src_lconf = _get_line_context(matching_tokens, tokens)
                fields["dob"] = DocumentField(
                    doc_id=doc_id,
                    name="dob",
                    value_raw=raw_d,
                    value_norm=norm_d,
                    ocr_conf=round(avg_conf, 3),
                    bbox=_calculate_bbox_union(matching_tokens),
                    method="regex",
                    rules_applied=rules,
                    source_text_span=src_span,
                    source_line_bbox=src_lbbox,
                    source_line_conf=src_lconf
                )
                break

    # 3. Regex fallback for 6-digit Pincode & Address if address not captured
    if "address" not in fields and doc_type not in ["marksheet"]:
        # Ensure document is not an academic transcript/marksheet by keyword check
        is_academic = any(k in full_text.lower() for k in ["marksheet", "statement of marks", "roll no", "examination board"])
        if not is_academic:
            pin_matches = list(re.finditer(r"(?<![A-Za-z0-9\-_])([1-9][0-9]{5})\b", full_text))
            for pm in pin_matches:
                pincode = pm.group(1)
                start_pos = max(0, pm.start() - 15)
                prefix_ctx = full_text[start_pos:pm.start()].lower()
                if any(bad_prefix in prefix_ctx for bad_prefix in ["roll", "id-", "id ", "no.", "no ", "reg", "acc"]):
                    continue

                pin_tokens = [t for t in tokens if pincode in t.text]
                if pin_tokens:
                    pin_idx = tokens.index(pin_tokens[0])
                    start_idx = max(0, pin_idx - 8)
                    end_idx = min(len(tokens), pin_idx + 2)
                    addr_slice = tokens[start_idx:end_idx]
                    raw_addr = " ".join([t.text for t in addr_slice])

                    addr_keywords = {
                        "road", "rd", "street", "st", "marg", "lane", "nagar", "layout",
                        "sector", "block", "flat", "apartment", "apt", "floor", "bldg",
                        "building", "city", "town", "district", "state", "pin", "pincode",
                        "postal", "no", "near", "opposite", "belgaum", "bangalore",
                        "bengaluru", "mumbai", "delhi", "chennai", "kolkata", "hyderabad", "pune"
                    }
                    has_addr_marker = any(k in raw_addr.lower().split() for k in addr_keywords) or any(k in raw_addr.lower() for k in ["road", "marg", "nagar", "street"])
                    if has_addr_marker:
                        norm_addr, rules = normalize_address(raw_addr)
                        src_span, src_lbbox, src_lconf = _get_line_context(addr_slice, tokens)
                        fields["address"] = DocumentField(
                            doc_id=doc_id,
                            name="address",
                            value_raw=raw_addr,
                            value_norm=norm_addr,
                            ocr_conf=round(float(np.mean([t.conf for t in addr_slice])), 3),
                            bbox=_calculate_bbox_union(addr_slice),
                            method="regex_pin_context",
                            rules_applied=rules,
                            source_text_span=src_span,
                            source_line_bbox=src_lbbox,
                            source_line_conf=src_lconf
                        )
                        break

    # 4. Regex fallback for ID Number if not captured
    if "id_number" not in fields:
        id_match = re.search(r"\b([A-Z]{2,4}[-\s]?[0-9]{6,12})\b", full_text)
        if id_match:
            raw_id = id_match.group(1)
            norm_id, rules = normalize_id_number(raw_id)
            matching_tokens = [t for t in tokens if raw_id in t.text or any(part in t.text for part in raw_id.split())]
            avg_conf = float(np.mean([t.conf for t in matching_tokens])) if matching_tokens else 0.85
            src_span, src_lbbox, src_lconf = _get_line_context(matching_tokens, tokens)
            fields["id_number"] = DocumentField(
                doc_id=doc_id,
                name="id_number",
                value_raw=raw_id,
                value_norm=norm_id,
                ocr_conf=round(avg_conf, 3),
                bbox=_calculate_bbox_union(matching_tokens),
                method="regex_id",
                rules_applied=rules,
                source_text_span=src_span,
                source_line_bbox=src_lbbox,
                source_line_conf=src_lconf
            )

    return fields
