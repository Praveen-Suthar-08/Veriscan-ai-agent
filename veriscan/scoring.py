"""Heuristic risk scoring and triage logic for VeriScan.

Computes finding-level confidence and aggregates case-level risk score:
case_risk = 1 - prod(1 - severity_weight_i * finding_conf_i)
"""

from __future__ import annotations
from typing import List, Dict, Any, Tuple
from pathlib import Path
import yaml

from veriscan.schemas import Finding

_CONFIG_PATH = Path(__file__).parent / "config.yaml"
_CONFIG: Dict[str, Any] = {}
if _CONFIG_PATH.exists():
    try:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            _CONFIG = yaml.safe_load(f) or {}
    except Exception:
        _CONFIG = {}

RISK_WEIGHTS = _CONFIG.get("risk_weights", {
    "HIGH": 1.0,
    "MEDIUM": 0.6,
    "LOW": 0.3,
    "INFO": 0.0
})

TRIAGE_CONFIG = _CONFIG.get("triage", {
    "green_max": 0.20,
    "amber_max": 0.60
})


def calculate_case_risk(findings: List[Finding]) -> Tuple[float, str]:
    """
    Calculate aggregated case risk score and triage level.

    Only MISMATCH and LOW_CONFIDENCE findings elevate risk.

    Confidence formula for each finding:
      - For MISMATCH: finding_conf = ocr_a * ocr_b * (1 - similarity)
        This correctly weights HIGH dissimilarity + HIGH OCR confidence more.
      - For LOW_CONFIDENCE: raw finding confidence, capped at 0.5

    Aggregation:
      case_risk = 1 - Product_i (1 - weight_i * finding_conf_i)

    Triage rules (deterministic, overriding the formula for edge cases):
      - Only MINOR_VARIANT or MATCH → GREEN (risk forced <= 0.15)
      - Any HIGH MISMATCH with conf >= 0.7 → RED
      - Any HIGH MISMATCH with conf < 0.7 → AMBER
      - Only LOW-severity findings → AMBER max
      - LOW_CONFIDENCE only → AMBER max
    """
    if not findings:
        return 0.0, "GREEN"

    has_high_mismatch = any(
        f.verdict == "MISMATCH" and f.severity == "HIGH" for f in findings
    )
    has_high_mismatch_conf = any(
        f.verdict == "MISMATCH" and f.severity == "HIGH" and f.confidence >= 0.70 for f in findings
    )
    has_medium_mismatch = any(
        f.verdict == "MISMATCH" and f.severity == "MEDIUM" for f in findings
    )
    only_low_or_info = all(
        f.severity in ("LOW", "INFO") or f.verdict in ("MINOR_VARIANT",) for f in findings
    )
    only_low_conf = all(f.verdict == "LOW_CONFIDENCE" for f in findings)
    has_real_mismatch = any(f.verdict == "MISMATCH" for f in findings)

    # Multiplicative risk model
    prod_complement = 1.0
    for f in findings:
        if f.verdict == "MISMATCH":
            weight = RISK_WEIGHTS.get(f.severity.upper(), 0.3)
            # Dissimilarity: how different are the values (1 - sim is the signal)
            dissimilarity = 1.0 - min(1.0, max(0.0, f.similarity))
            finding_conf = min(1.0, f.confidence * dissimilarity) if dissimilarity > 0 else f.confidence * 0.1
            term = 1.0 - (weight * finding_conf)
            term = max(0.05, term)
            prod_complement *= term
        elif f.verdict == "LOW_CONFIDENCE":
            # Moderate risk contribution for poor scan quality
            lc_conf = min(0.5, f.confidence)
            term = 1.0 - (0.4 * lc_conf)
            term = max(0.05, term)
            prod_complement *= term

    case_risk = 1.0 - prod_complement
    case_risk = round(min(1.0, max(0.0, case_risk)), 3)

    # Apply triage rules
    if has_high_mismatch_conf:
        triage = "RED"
    elif has_high_mismatch:
        triage = "AMBER"
        case_risk = max(case_risk, 0.45)
    elif has_medium_mismatch:
        triage = "AMBER"
    elif only_low_or_info and not has_real_mismatch:
        triage = "GREEN"
        case_risk = min(case_risk, 0.15)
    elif only_low_conf:
        triage = "AMBER"
        case_risk = max(case_risk, 0.22)
    else:
        green_max = TRIAGE_CONFIG.get("green_max", 0.20)
        amber_max = TRIAGE_CONFIG.get("amber_max", 0.60)
        if case_risk < green_max:
            triage = "GREEN"
        elif case_risk <= amber_max:
            triage = "AMBER"
        else:
            triage = "RED"

    return case_risk, triage

