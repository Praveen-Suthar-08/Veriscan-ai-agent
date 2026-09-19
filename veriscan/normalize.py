"""Pure functional normalization layer for VeriScan.

Every normalization function returns a tuple of (normalized_value, [rule_ids_applied]).
No external side-effects or network calls.
"""

from __future__ import annotations
import re
import unicodedata
from typing import Tuple, List, Dict, Optional, Set, Any
from pathlib import Path
import yaml

# Load default config for variants and abbreviations
_CONFIG_PATH = Path(__file__).parent / "config.yaml"
_CONFIG: Dict[str, Any] = {}
if _CONFIG_PATH.exists():
    try:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            _CONFIG = yaml.safe_load(f) or {}
    except Exception:
        _CONFIG = {}

# Precompile lookup structures
NAME_VARIANTS_LIST = _CONFIG.get("name_variants", [
    ["mohammed", "muhammad", "mohd", "md"],
    ["venkatesh", "venkatesan"],
    ["lakshmi", "laxmi"],
    ["vijay", "vijaya"],
    ["prasad", "prasada"],
    ["suresh", "suresha"],
    ["chandra", "chandran"],
    ["subramanian", "subramaniam"],
    ["srinivasan", "srinivas"],
    ["agarwal", "agrawal", "aggarwal"],
    ["sharma", "sarma"],
    ["choudhary", "chaudhary", "chowdhury"],
    ["patil", "patel"],
    ["kumar", "kr"],
    ["singh", "sing"],
    ["devi", "dabi"],
    ["gupta", "gupt"],
    ["reddy", "reddi"]
])

NAME_VARIANT_MAP: Dict[str, str] = {}
for group in NAME_VARIANTS_LIST:
    if group:
        canonical = group[0].lower()
        for variant in group:
            NAME_VARIANT_MAP[variant.lower()] = canonical

HONORIFICS = set(_CONFIG.get("honorifics", [
    "mr", "mrs", "ms", "shri", "shree", "smt", "dr", "prof", "kumari", "km", "master"
]))

RELATION_MARKERS = set(_CONFIG.get("relation_markers", [
    "s/o", "so", "d/o", "do", "w/o", "wo", "c/o", "co"
]))

_raw_abbr = _CONFIG.get("address_abbreviations", {
    "rd": "road", "st": "street", "ave": "avenue", "nr": "near", "opp": "opposite",
    "apt": "apartment", "flt": "flat", "no": "number", "no.": "number", "#": "number",
    "hno": "house number", "blr": "bengaluru", "bangalore": "bengaluru", "bom": "mumbai",
    "bombay": "mumbai", "cal": "kolkata", "calcutta": "kolkata", "madras": "chennai",
    "po": "post office", "dist": "district", "marg": "road", "ngr": "nagar",
    "soc": "society", "col": "colony", "bldg": "building", "sec": "sector",
    "ext": "extension", "cross": "cross", "main": "main"
})
ADDRESS_ABBREVIATIONS: Dict[str, str] = {}
for k, v in _raw_abbr.items():
    if k is False:
        ADDRESS_ABBREVIATIONS["no"] = str(v).lower()
    else:
        ADDRESS_ABBREVIATIONS[str(k).lower()] = str(v).lower()
ADDRESS_ABBREVIATIONS["no"] = "number"
ADDRESS_ABBREVIATIONS["no."] = "number"


# Month name mapping
MONTH_NAMES = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12
}


def normalize_name(raw_name: str) -> Tuple[str, List[str]]:
    """
    Normalize human name:
    - Unicode NFKD normalization
    - Lowercase / casefold
    - Strip honorifics (mr, mrs, shri, smt, dr, etc.)
    - Remove punctuation
    - Handle relation markers (s/o, d/o, w/o) by extracting applicant name
    - Expand / canonicalize known Indian spelling variants (e.g. md -> mohammed)
    - Return sorted unique token representation and list of rule IDs applied.
    """
    if not raw_name or not raw_name.strip():
        return "", ["RULE_EMPTY_NAME"]

    rules: List[str] = []
    text = raw_name.strip()

    # 1. Unicode NFKD normalization and strip combining diacritics
    nfkd_text = unicodedata.normalize("NFKD", text)
    stripped_combining = "".join(c for c in nfkd_text if not unicodedata.combining(c))
    if stripped_combining != text:
        rules.append("RULE_NAME_UNICODE_NFKD")
    text = stripped_combining


    # 2. Casefold
    lowered = text.casefold()
    if lowered != text:
        rules.append("RULE_NAME_CASEFOLD")
    text = lowered

    # 3. Handle relation markers (e.g. "Rohit Sharma S/O Mohan Sharma" -> "Rohit Sharma")
    for marker in [" s/o ", " d/o ", " w/o ", " c/o ", " so ", " do ", " wo ", " co "]:
        if marker in f" {text} ":
            idx = f" {text} ".find(marker)
            text = f" {text} "[:idx].strip()
            rules.append(f"RULE_NAME_STRIP_RELATION_{marker.strip().upper()}")
            break

    # 4. Remove punctuation except spaces
    cleaned = re.sub(r"[^\w\s]", " ", text)
    if cleaned != text:
        rules.append("RULE_NAME_REMOVE_PUNCTUATION")
    text = cleaned

    # 5. Tokenize and filter
    tokens = text.split()
    filtered_tokens: List[str] = []
    for tok in tokens:
        # Check honorific
        if tok in HONORIFICS:
            rules.append(f"RULE_NAME_STRIP_HONORIFIC_{tok.upper()}")
            continue
        # Check relation tokens
        if tok in RELATION_MARKERS:
            rules.append(f"RULE_NAME_STRIP_RELATION_{tok.upper()}")
            continue
        # Standardize known variants
        if tok in NAME_VARIANT_MAP:
            canonical = NAME_VARIANT_MAP[tok]
            rules.append(f"RULE_NAME_VARIANT_{tok.upper()}_TO_{canonical.upper()}")
            filtered_tokens.append(canonical)
        else:
            filtered_tokens.append(tok)

    if not filtered_tokens:
        return "", rules

    # 6. Sort tokens for order-independence
    sorted_tokens = sorted(filtered_tokens)
    if sorted_tokens != filtered_tokens:
        rules.append("RULE_NAME_TOKEN_ORDER_INDEPENDENT")

    normalized_name = " ".join(sorted_tokens)
    return normalized_name, sorted(list(set(rules)))


def normalize_date(raw_date: str) -> Tuple[str, List[str]]:
    """
    Parse and normalize a date string into ISO-8601 (YYYY-MM-DD).
    - Supports formats: DD/MM/YYYY, DD-MM-YYYY, DD.MM.YYYY, YYYY-MM-DD, DD Month YYYY, etc.
    - Day-first default for Indian document conventions.
    - Flags ambiguous dates where both day <= 12 and month <= 12 without named month.
    """
    if not raw_date or not raw_date.strip():
        return "", ["RULE_DATE_EMPTY"]

    rules: List[str] = []
    s = raw_date.strip().lower()
    s = unicodedata.normalize("NFKD", s)

    # Remove extraneous label tokens like 'dob', 'date of birth', 'born on', ':'
    s = re.sub(r"\b(dob|date of birth|birth date|d\.o\.b|date|dt)\b[:\s]*", "", s).strip()
    s = re.sub(r"^[^\w]+|[^\w]+$", "", s)

    year: Optional[int] = None
    month: Optional[int] = None
    day: Optional[int] = None

    # Check for text month: e.g. "12 Mar 2004", "12-March-2004", "March 12, 2004"
    match_named = re.search(
        r"\b(\d{1,2})[-/\s.,]+([a-z]{3,9})[-/\s.,]+(\d{2,4})\b", s
    )
    if not match_named:
        match_named_rev = re.search(
            r"\b([a-z]{3,9})[-/\s.,]+(\d{1,2})[-/\s.,]+(\d{2,4})\b", s
        )
    else:
        match_named_rev = None

    if match_named:
        d_str, m_str, y_str = match_named.groups()
        if m_str in MONTH_NAMES:
            day = int(d_str)
            month = MONTH_NAMES[m_str]
            year = int(y_str)
            rules.append("RULE_DATE_NAMED_MONTH_PARSE")
    elif match_named_rev:
        m_str, d_str, y_str = match_named_rev.groups()
        if m_str in MONTH_NAMES:
            day = int(d_str)
            month = MONTH_NAMES[m_str]
            year = int(y_str)
            rules.append("RULE_DATE_NAMED_MONTH_PREFIX_PARSE")
    else:
        # Numeric parsing
        # Try ISO: YYYY-MM-DD or YYYY/MM/DD
        match_iso = re.search(r"\b(\d{4})[-/. ](\d{1,2})[-/. ](\d{1,2})\b", s)
        if match_iso:
            year = int(match_iso.group(1))
            month = int(match_iso.group(2))
            day = int(match_iso.group(3))
            rules.append("RULE_DATE_ISO_PARSE")
        else:
            # Standard DD-MM-YYYY or DD/MM/YYYY
            match_dmy = re.search(r"\b(\d{1,2})[-/. ](\d{1,2})[-/. ](\d{2,4})\b", s)
            if match_dmy:
                d_val = int(match_dmy.group(1))
                m_val = int(match_dmy.group(2))
                y_val = int(match_dmy.group(3))

                # Day-first default
                day = d_val
                month = m_val
                year = y_val
                rules.append("RULE_DATE_DAY_FIRST_PARSE")

                # Check ambiguity
                if d_val <= 12 and m_val <= 12 and d_val != m_val:
                    rules.append("RULE_DATE_AMBIGUOUS_DAY_MONTH")

    if year is not None and month is not None and day is not None:
        # Expand 2-digit year (e.g. 04 -> 2004, 98 -> 1998)
        if year < 100:
            if year <= 30:
                year += 2000
            else:
                year += 1900
            rules.append("RULE_DATE_TWO_DIGIT_YEAR_EXPANDED")

        # Validate range
        if 1 <= month <= 12 and 1 <= day <= 31 and 1900 <= year <= 2035:
            iso_date = f"{year:04d}-{month:02d}-{day:02d}"
            rules.append("RULE_DATE_ISO8601_CANONICAL")
            return iso_date, sorted(list(set(rules)))

    # Fallback return raw if unparseable
    return s, ["RULE_DATE_PARSE_FAILED"]


def normalize_address(raw_address: str) -> Tuple[str, List[str]]:
    """
    Normalize raw address string:
    - Lowercase, unicode normalization
    - Expand common abbreviations (rd->road, st->street, blr->bengaluru, etc.)
    - Drop filler tokens
    - Extract and canonicalize 6-digit Indian PIN code
    - Standardize whitespace and separator tokens.
    """
    if not raw_address or not raw_address.strip():
        return "", ["RULE_ADDR_EMPTY"]

    rules: List[str] = []
    text = raw_address.strip().lower()
    text = unicodedata.normalize("NFKD", text)

    # 1. Extract pincode if present
    pin_match = re.search(r"\b([1-9][0-9]{5})\b", text)
    pincode = ""
    if pin_match:
        pincode = pin_match.group(1)
        rules.append(f"RULE_ADDR_EXTRACTED_PIN_{pincode}")

    # 2. Replace punctuation with space
    cleaned = re.sub(r"[^\w\s]", " ", text)
    if cleaned != text:
        rules.append("RULE_ADDR_CLEAN_PUNCTUATION")
    text = cleaned

    # 3. Token substitution for abbreviations
    tokens = text.split()
    expanded_tokens: List[str] = []
    for tok in tokens:
        if tok in ADDRESS_ABBREVIATIONS:
            exp = ADDRESS_ABBREVIATIONS[tok]
            expanded_tokens.append(exp)
            rules.append(f"RULE_ADDR_EXPAND_{tok.upper()}_TO_{exp.upper()}")
        else:
            expanded_tokens.append(tok)

    normalized_str = " ".join(expanded_tokens)
    return normalized_str, sorted(list(set(rules)))


def extract_address_components(raw_address: str) -> Tuple[Dict[str, str], List[str]]:
    """
    Split normalized address into structured components:
    - house_number
    - street_locality
    - city
    - pincode
    """
    norm_str, rules = normalize_address(raw_address)
    components: Dict[str, str] = {
        "house_number": "",
        "street_locality": "",
        "city": "",
        "pincode": ""
    }

    if not norm_str:
        return components, rules

    tokens = norm_str.split()

    # 1. Extract 6-digit PIN code
    for i, tok in enumerate(tokens):
        if re.fullmatch(r"[1-9][0-9]{5}", tok):
            components["pincode"] = tok
            tokens.pop(i)
            break

    # 2. Extract city if known Indian major city appears
    known_cities = [
        "bengaluru", "mumbai", "delhi", "chennai", "kolkata", "hyderabad",
        "pune", "ahmedabad", "jaipur", "lucknow", "chandigarh", "bhopal",
        "indore", "patna", "nagpur", "kochi", "coimbatore", "mysuru"
    ]
    city_found = ""
    for city in known_cities:
        if city in tokens:
            city_found = city
            tokens = [t for t in tokens if t != city]
            break
    components["city"] = city_found

    # 3. Extract house number (first token if contains digits, or if prefixed with number/no/flat/hno)
    house_no = ""
    if tokens:
        if re.search(r"\d", tokens[0]):
            house_no = tokens[0]
            tokens = tokens[1:]
        elif len(tokens) > 1 and tokens[0] in ["number", "no", "flat", "plot", "house", "hno", "apt"] and re.search(r"\d", tokens[1]):
            house_no = f"{tokens[0]} {tokens[1]}"
            tokens = tokens[2:]
    components["house_number"] = house_no

    # 4. Remaining tokens comprise street and locality
    components["street_locality"] = " ".join(tokens)

    rules.append("RULE_ADDR_COMPONENT_SPLIT")
    return components, sorted(list(set(rules)))


def normalize_id_number(raw_id: str, is_numeric_only: bool = False) -> Tuple[str, List[str]]:
    """
    Normalize ID / Roll / Document numbers:
    - Strip separators (whitespace, dashes, slashes)
    - Uppercase
    - OCR confusion fixes (O<->0, I/l<->1, S<->5, B<->8) applied inside numeric-typed fields
    """
    if not raw_id or not raw_id.strip():
        return "", ["RULE_ID_EMPTY"]

    rules: List[str] = []
    text = raw_id.strip().upper()

    # Strip whitespace, dashes, slashes
    cleaned = re.sub(r"[\s\-_/.]+", "", text)
    if cleaned != text:
        rules.append("RULE_ID_STRIP_SEPARATORS")
    text = cleaned

    if is_numeric_only:
        # Apply OCR confusion fixes for numbers
        replacements = {
            "O": "0", "D": "0", "Q": "0",
            "I": "1", "L": "1", "|": "1",
            "Z": "2",
            "S": "5",
            "B": "8"
        }
        res_chars = []
        applied_fix = False
        for ch in text:
            if ch in replacements:
                res_chars.append(replacements[ch])
                applied_fix = True
            else:
                res_chars.append(ch)
        if applied_fix:
            rules.append("RULE_ID_OCR_NUMERIC_CONFUSION_FIX")
        text = "".join(res_chars)

    return text, sorted(list(set(rules)))
