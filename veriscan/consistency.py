"""Cross-document consistency engine for VeriScan.

Compares normalized fields across documents and yields deterministic verdicts:
MATCH, MINOR_VARIANT, MISMATCH, MISSING, LOW_CONFIDENCE.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Any, Optional
import re
from pathlib import Path
import yaml

try:
    from rapidfuzz import fuzz, distance
except ImportError:
    fuzz = None
    distance = None

try:
    import jellyfish
except ImportError:
    jellyfish = None

from veriscan.schemas import Finding, ConfidenceBreakdown, DocumentField
from veriscan.normalize import (
    normalize_name, normalize_date, normalize_address,
    extract_address_components, normalize_id_number
)

# Load configuration
_CONFIG_PATH = Path(__file__).parent / "config.yaml"
_CONFIG: Dict[str, Any] = {}
if _CONFIG_PATH.exists():
    try:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            _CONFIG = yaml.safe_load(f) or {}
    except Exception:
        _CONFIG = {}

MIN_OCR_CONF_GATE = _CONFIG.get("ocr", {}).get("min_confidence", 0.60)
NAME_MATCH_THRESH = _CONFIG.get("thresholds", {}).get("name", {}).get("match", 0.92)
NAME_VARIANT_THRESH = _CONFIG.get("thresholds", {}).get("name", {}).get("minor_variant", 0.75)
ADDR_MATCH_THRESH = _CONFIG.get("thresholds", {}).get("address", {}).get("match", 0.85)
ADDR_VARIANT_THRESH = _CONFIG.get("thresholds", {}).get("address", {}).get("minor_variant", 0.70)

SEVERITY_MAP = _CONFIG.get("severity", {
    "name": "HIGH",
    "full_name": "HIGH",
    "dob": "HIGH",
    "date_of_birth": "HIGH",
    "id_number": "HIGH",
    "guardian_name": "MEDIUM",
    "father_name": "MEDIUM",
    "address": "MEDIUM",
    "pincode_only": "LOW",
    "missing_optional": "INFO"
})


def _fallback_levenshtein_ratio(s1: str, s2: str) -> float:
    """Pure python fallback if rapidfuzz/jellyfish not yet loaded."""
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    # Basic token overlap
    tokens1 = set(s1.split())
    tokens2 = set(s2.split())
    if not tokens1 or not tokens2:
        return 0.0
    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)
    return intersection / union


def compare_names(
    raw_a: str, raw_b: str, norm_a: str, norm_b: str
) -> Tuple[str, float, List[str], List[str]]:
    """
    Compare names using multi-metric blend:
    - RapidFuzz token_set_ratio
    - Normalized Levenshtein similarity
    - Jaro-Winkler
    - Metaphone equality
    - Initial-expansion match (e.g. 'r sharma' vs 'rohit sharma')
    Returns: (verdict, similarity, reasons, rule_ids)
    """
    reasons: List[str] = []
    rule_ids: List[str] = []

    if norm_a == norm_b:
        reasons.append("Normalized names match exactly.")
        rule_ids.append("RULE_NAME_EXACT_MATCH")
        return "MATCH", 1.0, reasons, rule_ids

    tokens_a = norm_a.split()
    tokens_b = norm_b.split()

    # Check for initial expansion match
    is_initial_match = False
    if len(tokens_a) == len(tokens_b) and len(tokens_a) > 0:
        matches = 0
        for ta, tb in zip(tokens_a, tokens_b):
            if ta == tb:
                matches += 1
            elif (len(ta) == 1 and tb.startswith(ta)) or (len(tb) == 1 and ta.startswith(tb)):
                matches += 1
                rule_ids.append(f"RULE_NAME_INITIAL_EXPAND_{ta.upper()}_{tb.upper()}")
        if matches == len(tokens_a):
            is_initial_match = True
            reasons.append("Initials match expanded name tokens.")

    # Calculate metrics
    if fuzz and jellyfish:
        token_set_sim = fuzz.token_set_ratio(norm_a, norm_b) / 100.0
        lev_sim = 1.0 - (distance.Levenshtein.normalized_distance(norm_a, norm_b))
        jaro_sim = jellyfish.jaro_winkler_similarity(norm_a, norm_b)

        # Metaphone check
        meta_a = " ".join([jellyfish.metaphone(t) for t in tokens_a if t])
        meta_b = " ".join([jellyfish.metaphone(t) for t in tokens_b if t])
        meta_sim = 1.0 if meta_a == meta_b else (0.8 if fuzz.token_set_ratio(meta_a, meta_b) > 85 else 0.0)
    else:
        token_set_sim = _fallback_levenshtein_ratio(norm_a, norm_b)
        lev_sim = token_set_sim
        jaro_sim = token_set_sim
        meta_sim = 1.0 if norm_a == norm_b else 0.0

    # Weighted blend
    if is_initial_match:
        similarity = max(0.93, 0.35 * token_set_sim + 0.25 * lev_sim + 0.20 * jaro_sim + 0.20)
        reasons.append(f"Name tokens correspond with recognized initial expansion (similarity: {similarity:.2f}).")
    else:
        similarity = (
            0.35 * token_set_sim +
            0.25 * lev_sim +
            0.25 * jaro_sim +
            0.15 * meta_sim
        )
        similarity = min(1.0, max(0.0, similarity))

    # Verdict assignment based on thresholds
    if similarity >= NAME_MATCH_THRESH:
        verdict = "MATCH"
        reasons.append(f"High name similarity score of {similarity:.2f} confirms consistent identity.")
        rule_ids.append("RULE_NAME_HIGH_SIMILARITY_MATCH")
    elif similarity >= NAME_VARIANT_THRESH:
        verdict = "MINOR_VARIANT"
        reasons.append(
            f"Moderate name variation detected ({similarity:.2f}) - likely minor transliteration or abbreviation."
        )
        rule_ids.append("RULE_NAME_MINOR_VARIANT")
    else:
        verdict = "MISMATCH"
        reasons.append(
            f"Significant name difference ({similarity:.2f}) between '{raw_a}' and '{raw_b}'."
        )
        rule_ids.append("RULE_NAME_SUBSTANTIAL_MISMATCH")

    return verdict, round(similarity, 3), reasons, rule_ids


def compare_dates(
    raw_a: str, raw_b: str, norm_a: str, norm_b: str
) -> Tuple[str, float, List[str], List[str]]:
    """
    Compare date fields.
    Exact ISO equality = MATCH.
    Otherwise MISMATCH with specific subtype:
    - single_digit_typo (Levenshtein dist == 1 on ISO)
    - day_month_swap
    - year_off
    - different
    """
    reasons: List[str] = []
    rule_ids: List[str] = []

    if norm_a == norm_b and norm_a:
        reasons.append("Dates match exactly in ISO-8601 format.")
        rule_ids.append("RULE_DATE_EXACT_MATCH")
        return "MATCH", 1.0, reasons, rule_ids

    # Subtype diagnosis
    similarity = 0.0
    # Parse ISO components if available
    iso_pat = r"^(\d{4})-(\d{2})-(\d{2})$"
    ma = re.match(iso_pat, norm_a)
    mb = re.match(iso_pat, norm_b)

    if ma and mb:
        ya, ma_val, da = int(ma.group(1)), int(ma.group(2)), int(ma.group(3))
        yb, mb_val, db = int(mb.group(1)), int(mb.group(2)), int(mb.group(3))

        # Check year off (same day and month, year within 5 years)
        if ma_val == mb_val and da == db and ya != yb and abs(ya - yb) <= 5:
            reasons.append(
                f"Year difference detected with identical day and month: {ya} vs {yb}."
            )
            rule_ids.append("RULE_DATE_SUBTYPE_YEAR_OFF")
            return "MISMATCH", 0.50, reasons, rule_ids

        # Check day-month swap: e.g. 2004-03-12 vs 2004-12-03
        if ya == yb and ma_val == db and da == mb_val:
            reasons.append(
                f"Likely day-month transposition detected: '{norm_a}' vs '{norm_b}' (Day {da} vs Month {mb_val})."
            )
            rule_ids.append("RULE_DATE_SUBTYPE_DAY_MONTH_SWAP")
            return "MISMATCH", 0.70, reasons, rule_ids

        # Check single digit typo (e.g. 1998-05-12 vs 1998-05-15)
        diff_chars = sum(1 for ca, cb in zip(norm_a, norm_b) if ca != cb)
        if diff_chars == 1:
            reasons.append(
                f"Single digit difference detected: '{norm_a}' vs '{norm_b}' (probable OCR or clerical typo)."
            )
            rule_ids.append("RULE_DATE_SUBTYPE_SINGLE_DIGIT_TYPO")
            return "MISMATCH", 0.65, reasons, rule_ids


    reasons.append(f"Inconsistent dates detected: '{raw_a}' vs '{raw_b}'.")
    rule_ids.append("RULE_DATE_SUBTYPE_DIFFERENT")
    return "MISMATCH", 0.0, reasons, rule_ids


def compare_addresses(
    raw_a: str, raw_b: str, norm_a: str, norm_b: str
) -> Tuple[str, float, List[str], List[str]]:
    """
    Compare address strings using component weighting:
    - Pincode: 0.30
    - City: 0.20
    - Street / Locality: 0.30
    - House Number: 0.20
    """
    reasons: List[str] = []
    rule_ids: List[str] = []

    if norm_a == norm_b and norm_a:
        reasons.append("Normalized address matches completely.")
        rule_ids.append("RULE_ADDR_EXACT_MATCH")
        return "MATCH", 1.0, reasons, rule_ids

    comp_a, _ = extract_address_components(raw_a)
    comp_b, _ = extract_address_components(raw_b)

    # 1. Pincode comparison
    pin_a = comp_a.get("pincode", "")
    pin_b = comp_b.get("pincode", "")
    pin_score = 1.0 if (pin_a and pin_a == pin_b) else (0.5 if not pin_a or not pin_b else 0.0)

    # 2. City comparison
    city_a = comp_a.get("city", "")
    city_b = comp_b.get("city", "")
    city_score = 1.0 if (city_a and city_a == city_b) else (0.5 if not city_a or not city_b else 0.0)

    # 3. Street / Locality comparison
    st_a = comp_a.get("street_locality", "")
    st_b = comp_b.get("street_locality", "")
    if st_a and st_b:
        st_score = (fuzz.token_set_ratio(st_a, st_b) / 100.0) if fuzz else _fallback_levenshtein_ratio(st_a, st_b)
    else:
        st_score = 0.5

    # 4. House Number comparison
    h_a = comp_a.get("house_number", "")
    h_b = comp_b.get("house_number", "")
    h_score = 1.0 if (h_a and h_a == h_b) else (0.5 if not h_a or not h_b else 0.0)

    # Weighted calculation
    weights = _CONFIG.get("thresholds", {}).get("address", {}).get("weights", {
        "pincode": 0.30, "city": 0.20, "street_locality": 0.30, "house_number": 0.20
    })

    similarity = (
        weights["pincode"] * pin_score +
        weights["city"] * city_score +
        weights["street_locality"] * st_score +
        weights["house_number"] * h_score
    )
    similarity = min(1.0, max(0.0, similarity))

    # Diagnostic reasons
    if pin_a and pin_b and pin_a != pin_b:
        reasons.append(f"Pincode mismatch: {pin_a} vs {pin_b}.")
        rule_ids.append("RULE_ADDR_PINCODE_MISMATCH")

    if city_a and city_b and city_a != city_b:
        reasons.append(f"City mismatch: {city_a} vs {city_b}.")
        rule_ids.append("RULE_ADDR_CITY_MISMATCH")

    if similarity >= ADDR_MATCH_THRESH:
        verdict = "MATCH"
        reasons.append(f"Address components align with similarity score {similarity:.2f}.")
        rule_ids.append("RULE_ADDR_HIGH_MATCH")
    elif similarity >= ADDR_VARIANT_THRESH:
        verdict = "MINOR_VARIANT"
        reasons.append(f"Minor address formatting or locality variations ({similarity:.2f}).")
        rule_ids.append("RULE_ADDR_MINOR_VARIANT")
    else:
        verdict = "MISMATCH"
        reasons.append(f"Significant address inconsistency ({similarity:.2f}).")
        rule_ids.append("RULE_ADDR_MISMATCH")

    return verdict, round(similarity, 3), reasons, rule_ids


def compare_id_numbers(
    raw_a: str, raw_b: str, norm_a: str, norm_b: str
) -> Tuple[str, float, List[str], List[str]]:
    """Compare ID/Roll/Document numbers. Exact equality required after normalization."""
    reasons: List[str] = []
    rule_ids: List[str] = []

    if norm_a == norm_b and norm_a:
        reasons.append("ID numbers match exactly after normalization.")
        rule_ids.append("RULE_ID_EXACT_MATCH")
        return "MATCH", 1.0, reasons, rule_ids

    reasons.append(f"ID number discrepancy: '{norm_a}' vs '{norm_b}'.")
    rule_ids.append("RULE_ID_MISMATCH")
    return "MISMATCH", 0.0, reasons, rule_ids


def compare_document_fields(
    field_name: str,
    field_a: DocumentField,
    field_b: DocumentField
) -> Finding:
    """
    Compare a shared field across two documents and generate a Finding object.
    Applies OCR confidence gating: if either OCR conf < min_confidence, forces LOW_CONFIDENCE.
    """
    fn = field_name.lower()
    raw_a, raw_b = field_a.value_raw, field_b.value_raw
    norm_a, norm_b = field_a.value_norm, field_b.value_norm

    # 1. Determine severity
    severity = SEVERITY_MAP.get(fn, "MEDIUM")

    # 2. Field-specific comparator
    if "name" in fn and "father" not in fn and "guardian" not in fn:
        verdict, similarity, reasons, rule_ids = compare_names(raw_a, raw_b, norm_a, norm_b)
    elif "dob" in fn or "date" in fn:
        verdict, similarity, reasons, rule_ids = compare_dates(raw_a, raw_b, norm_a, norm_b)
    elif "address" in fn:
        verdict, similarity, reasons, rule_ids = compare_addresses(raw_a, raw_b, norm_a, norm_b)
        # Check if only pincode mismatch
        if verdict == "MISMATCH" and "RULE_ADDR_PINCODE_MISMATCH" in rule_ids and "RULE_ADDR_CITY_MISMATCH" not in rule_ids:
            severity = SEVERITY_MAP.get("pincode_only", "LOW")
    elif "id" in fn or "number" in fn or "roll" in fn:
        verdict, similarity, reasons, rule_ids = compare_id_numbers(raw_a, raw_b, norm_a, norm_b)
    else:
        # Generic name/text comparison
        verdict, similarity, reasons, rule_ids = compare_names(raw_a, raw_b, norm_a, norm_b)

    # 3. Gating rule: If OCR confidence < threshold on either document, verdict is LOW_CONFIDENCE
    conf_a = field_a.ocr_conf
    conf_b = field_b.ocr_conf
    if conf_a < MIN_OCR_CONF_GATE or conf_b < MIN_OCR_CONF_GATE:
        old_verdict = verdict
        verdict = "LOW_CONFIDENCE"
        reasons.append(
            f"OCR confidence ({min(conf_a, conf_b):.2f}) is below reliable threshold ({MIN_OCR_CONF_GATE:.2f}). "
            f"Preliminary verdict was {old_verdict}."
        )
        rule_ids.append("RULE_OCR_CONFIDENCE_GATED")

    # 4. Compute confidence & breakdown
    # For MISMATCH: higher confidence means more certain it is a genuine discrepancy
    similarity_signal = (1.0 - similarity) if verdict in ["MISMATCH", "LOW_CONFIDENCE"] else similarity
    calibrated_conf = round(conf_a * conf_b * max(0.5, similarity_signal), 3)

    confidence_breakdown = ConfidenceBreakdown(
        ocr_a=round(conf_a, 3),
        ocr_b=round(conf_b, 3),
        similarity_signal=round(similarity_signal, 3),
        calibrated_conf=calibrated_conf
    )

    # 5. Suggested reviewer action
    if verdict == "MATCH":
        suggested_action = "No action required (consistent)"
    elif verdict == "MINOR_VARIANT":
        suggested_action = "Review spelling/formatting variation; confirm if acceptable"
    elif verdict == "LOW_CONFIDENCE":
        suggested_action = "Inspect source image closely or request clearer document re-upload"
    else:  # MISMATCH
        suggested_action = f"Verify discrepancy between {field_a.doc_id} and {field_b.doc_id}; require explanation or re-upload"

    finding_id = f"finding_{field_a.doc_id}_{field_b.doc_id}_{fn}"

    return Finding(
        id=finding_id,
        field=fn,
        docs=[field_a.doc_id, field_b.doc_id],
        values_raw={field_a.doc_id: raw_a, field_b.doc_id: raw_b},
        values_norm={field_a.doc_id: norm_a, field_b.doc_id: norm_b},
        verdict=verdict,
        similarity=similarity,
        severity=severity,
        confidence=calibrated_conf,
        confidence_breakdown=confidence_breakdown,
        reasons=reasons,
        rule_ids=rule_ids,
        suggested_action=suggested_action,
        bboxes={
            field_a.doc_id: field_a.bbox,
            field_b.doc_id: field_b.bbox
        }
    )
