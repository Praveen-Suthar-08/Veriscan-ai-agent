"""Document classification module for VeriScan.

Classifies documents into canonical types based on weighted keyword heuristics:
- id_card
- marksheet
- address_proof
- bank_statement
- application_form
- unknown

Returns classification with confidence score.
"""

from __future__ import annotations
from typing import List, Dict, Tuple, Optional
from veriscan.schemas import Token

# Definitive (primary) keywords — strong 5-point signals
PRIMARY_KEYWORDS: Dict[str, List[str]] = {
    "id_card": [
        "identity card", "id card", "aadhaar", "aadhar", "pan card",
        "voter id", "voter card", "driving licence", "driving license",
        "permanent account number", "national id", "unique identification authority",
        "election commission", "uid", "uidai"
    ],
    "marksheet": [
        "marksheet", "mark sheet", "statement of marks", "board of secondary",
        "marks obtained", "cgpa", "sgpa", "grade card", "passing certificate",
        "academic transcript", "board examination", "university examination",
        "roll no", "roll number", "roll no.", "admit card", "result card"
    ],
    "address_proof": [
        "electricity bill", "electric bill", "utility bill", "water bill",
        "consumer number", "power distribution", "discom", "gas bill",
        "meter number", "lpg connection", "broadband bill", "telephone bill",
        "bill amount", "units consumed", "reading date", "due date", "service address"
    ],
    "bank_statement": [
        "bank statement", "account statement", "passbook", "account holder",
        "ifsc", "savings account", "current account", "account number",
        "opening balance", "closing balance", "statement period", "branch code"
    ],
    "application_form": [
        "application form", "admission form", "enrollment form",
        "registration form", "declaration form", "applicant name",
        "candidate name", "signature of applicant", "place of birth",
        "personal details", "admission number"
    ]
}

# Secondary (supporting) keywords — 1-point signals each
SECONDARY_KEYWORDS: Dict[str, List[str]] = {
    "id_card": ["dob", "date of birth", "father", "address", "sex", "gender", "photo", "govt of india"],
    "marksheet": [
        "examination", "roll", "subject", "grade", "marks", "result",
        "pass", "fail", "semester", "academic year", "institution",
        "theory", "practical", "total marks", "school", "college", "university"
    ],
    "address_proof": [
        "consumer", "connection", "connection id", "bill date", "amount due",
        "billing period", "address", "meter", "cylinder", "subscriber", "client"
    ],
    "bank_statement": [
        "debit", "credit", "transaction", "balance", "deposit",
        "withdrawal", "branch", "micr", "statement", "bank"
    ],
    "application_form": [
        "candidate", "declaration", "father name", "mother name", "applicant",
        "purpose", "category", "enclosures", "self declaration", "place date"
    ]
}


def classify_document(tokens: List[Token]) -> str:
    """
    Classify document type. Returns the most likely type string.
    Uses weighted keyword scoring: primary=5pts, secondary=1pt per keyword.
    Returns 'unknown' if score < 3.
    """
    result = classify_document_with_confidence(tokens)
    return result["doc_type"]


def classify_document_with_confidence(tokens: List[Token]) -> Dict:
    """
    Classify document type with confidence score.
    Returns: {"doc_type": str, "confidence": float (0-1), "top_2": list[str]}
    """
    if not tokens:
        return {"doc_type": "unknown", "confidence": 0.0, "top_2": []}

    full_text = " ".join([t.text.lower() for t in tokens])

    scores: Dict[str, float] = {k: 0.0 for k in PRIMARY_KEYWORDS}

    # Primary keywords: 5 points each
    for doc_type, keywords in PRIMARY_KEYWORDS.items():
        for kw in keywords:
            if kw in full_text:
                scores[doc_type] += 5

    # Secondary keywords: 1 point each (multi-word = 2 pts)
    for doc_type, keywords in SECONDARY_KEYWORDS.items():
        for kw in keywords:
            if kw in full_text:
                scores[doc_type] += 2 if " " in kw else 1

    # Rank types by score
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    best_type, best_score = ranked[0]
    second_type = ranked[1][0] if len(ranked) > 1 else "unknown"

    # Threshold: at least 3 points to classify
    if best_score < 3:
        return {"doc_type": "unknown", "confidence": 0.0, "top_2": []}

    # Confidence: ratio of best score to theoretical max (number of primary kws * 5)
    max_possible = len(PRIMARY_KEYWORDS.get(best_type, [])) * 5 + len(SECONDARY_KEYWORDS.get(best_type, [])) * 2
    confidence = min(1.0, best_score / max(1, max_possible))

    return {
        "doc_type": best_type,
        "confidence": round(confidence, 3),
        "top_2": [best_type, second_type]
    }
