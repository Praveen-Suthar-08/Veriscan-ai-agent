"""Pydantic data models for VeriScan Document & Identity Consistency-Checking Agent."""

from __future__ import annotations
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field  # type: ignore
from datetime import datetime, timezone

DISCLAIMER: str = (
    "This report is an automated pre-screening aid. It flags possible inconsistencies for human review "
    "and does not determine authenticity, eligibility, or approval. Final judgment rests with the "
    "human reviewer. Only synthetic sample data is used in this demonstration."
)


class BBox(BaseModel):
    """Bounding box [x, y, w, h]."""
    x: int
    y: int
    w: int
    h: int

    def to_list(self) -> List[int]:
        return [self.x, self.y, self.w, self.h]


class Token(BaseModel):
    """Individual OCR token with text, confidence, and bounding box."""
    text: str
    conf: float = Field(ge=0.0, le=1.0)
    bbox: List[int] = Field(default_factory=lambda: [0, 0, 0, 0])  # [x, y, w, h]
    line_id: int = 0


class DocumentField(BaseModel):
    """Extracted field from a single document."""
    doc_id: str
    name: str
    value_raw: str
    value_norm: str
    ocr_conf: float = Field(ge=0.0, le=1.0)
    bbox: Optional[List[int]] = None
    page: int = 1
    method: str = "rule"  # "rule", "regex", "retry", "llm_gap_fill"
    rules_applied: List[str] = Field(default_factory=list)


class QualityMetrics(BaseModel):
    """Image quality assessment metrics."""
    blur_score: float = 0.0  # Variance of Laplacian
    is_blurry: bool = False
    skew_angle: float = 0.0  # Degrees
    is_skewed: bool = False
    resolution_dpi: int = 200
    is_low_res: bool = False
    warnings: List[str] = Field(default_factory=list)


class Document(BaseModel):
    """Container for an ingested and processed document."""
    doc_id: str
    filename: str
    doc_type: str = "unknown"  # id_card, marksheet, address_proof, bank_statement, application_form, unknown
    pages: int = 1
    tokens: List[Token] = Field(default_factory=list)
    fields: Dict[str, DocumentField] = Field(default_factory=dict)
    quality: QualityMetrics = Field(default_factory=QualityMetrics)
    raw_image_path: Optional[str] = None


class ConfidenceBreakdown(BaseModel):
    """Granular factors contributing to a finding's confidence."""
    ocr_a: float = 1.0
    ocr_b: float = 1.0
    similarity_signal: float = 1.0
    calibrated_conf: float = 1.0


class Finding(BaseModel):
    """Flagged discrepancy, minor variant, or missing field between documents."""
    id: str
    field: str
    docs: List[str]  # e.g., ["doc_1", "doc_2"]
    values_raw: Dict[str, str]  # doc_id -> raw string
    values_norm: Dict[str, str]  # doc_id -> normalized string
    verdict: str  # MATCH, MINOR_VARIANT, MISMATCH, MISSING, LOW_CONFIDENCE
    similarity: float = Field(ge=0.0, le=1.0)
    severity: str = "INFO"  # HIGH, MEDIUM, LOW, INFO
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_breakdown: ConfidenceBreakdown = Field(default_factory=ConfidenceBreakdown)
    reasons: List[str] = Field(default_factory=list)
    rule_ids: List[str] = Field(default_factory=list)
    suggested_action: str = "Verify manually against original document"
    bboxes: Dict[str, Optional[List[int]]] = Field(default_factory=dict)


class ConsistentItem(BaseModel):
    """Records fields that were checked and confirmed consistent."""
    id: str
    field: str
    docs: List[str]
    values_raw: Dict[str, str]
    values_norm: Dict[str, str]
    verdict: str = "MATCH"
    similarity: float = 1.0
    confidence: float = 1.0
    reasons: List[str] = Field(default_factory=list)
    rule_ids: List[str] = Field(default_factory=list)


class ToolTraceStep(BaseModel):
    """Orchestrator execution trace step for auditing."""
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    tool: str
    args_summary: str
    result_summary: str
    duration_ms: float = 0.0


class ReviewerDecision(BaseModel):
    """Human reviewer action and notes recorded for a finding."""
    finding_id: str
    case_id: str
    status: str = "PENDING"  # PENDING, CONFIRMED, DISMISSED, NEEDS_REUPLOAD
    notes: str = ""
    reviewer_id: str = "reviewer_1"
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class CaseReport(BaseModel):
    """Complete case screening report."""
    case_id: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    ocr_engine: str = "easyocr"
    documents: List[Document] = Field(default_factory=list)
    quality_warnings: List[str] = Field(default_factory=list)
    findings: List[Finding] = Field(default_factory=list)
    consistent_items: List[ConsistentItem] = Field(default_factory=list)
    risk_score: float = 0.0  # Heuristic risk score in [0.0, 1.0]
    triage: str = "GREEN"  # GREEN, AMBER, RED
    narrative: str = ""
    suggested_questions: List[str] = Field(default_factory=list)
    tool_trace: List[ToolTraceStep] = Field(default_factory=list)
    reviewer_decisions: Dict[str, ReviewerDecision] = Field(default_factory=dict)
    llm_provider: str = "template"
    disclaimer: str = DISCLAIMER
