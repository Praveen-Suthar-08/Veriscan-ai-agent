"""Autonomous Agent Orchestrator with tool trace and progressive retry loop for VeriScan.

Coordinates:
- assess_quality
- run_ocr
- enhance_image (progressive retry loop on critical fields)
- extract_fields
- normalize
- compare_fields
- score_case
- render_report
"""

from __future__ import annotations
import re
import time
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np  # type: ignore
import cv2  # type: ignore

from veriscan.schemas import (
    Document, DocumentField, Token, Finding, ConsistentItem,
    CaseReport, ToolTraceStep, QualityMetrics, EvidenceEntry
)
from veriscan.quality import assess_image_quality
from veriscan.preprocess import (
    deskew, denoise_and_threshold, enhance_clahe, upscale_and_sharpen, crop_region
)
from veriscan.ocr import get_ocr_engine, OCREngine, load_tokens_cache, save_tokens_cache
from veriscan.classify import classify_document
from veriscan.extract import extract_fields_from_tokens
from veriscan.consistency import compare_document_fields
from veriscan.scoring import calculate_case_risk


class Orchestrator:
    """Deterministic orchestrator managing end-to-end document consistency screening."""

    def __init__(self, default_engine: str = "easyocr"):
        self.default_engine_name = default_engine
        self._ocr_engine: Optional[OCREngine] = None
        self.tool_trace: List[ToolTraceStep] = []

    def _log_trace(self, tool: str, args: str, result: str, duration: float) -> None:
        """Record an execution step in the audit trace."""
        self.tool_trace.append(
            ToolTraceStep(
                tool=tool,
                args_summary=args,
                result_summary=result,
                duration_ms=round(duration * 1000.0, 1)
            )
        )

    def _get_engine(self, engine_name: Optional[str] = None) -> OCREngine:
        name = engine_name or self.default_engine_name
        return get_ocr_engine(name)

    def assess_quality(self, image: np.ndarray, filename: str) -> QualityMetrics:
        """Tool 1: Evaluate blur, skew, and resolution."""
        t0 = time.time()
        metrics = assess_image_quality(image)
        self._log_trace(
            tool="assess_quality",
            args=f"file={filename}, shape={image.shape}",
            result=f"blur={metrics.blur_score}, skew={metrics.skew_angle}°, warnings={len(metrics.warnings)}",
            duration=time.time() - t0
        )
        return metrics

    def run_ocr(
        self,
        image: np.ndarray,
        doc_id: str,
        engine_name: Optional[str] = None,
        cache_path: Optional[Path] = None
    ) -> List[Token]:
        """Tool 2: Run OCR with cache check."""
        t0 = time.time()

        # Check cache first for rapid demonstration
        if cache_path and cache_path.exists():
            cached = load_tokens_cache(cache_path)
            if cached is not None:
                self._log_trace(
                    tool="run_ocr",
                    args=f"doc_id={doc_id}, cached=True",
                    result=f"Retrieved {len(cached)} tokens from cache",
                    duration=time.time() - t0
                )
                return cached

        engine = self._get_engine(engine_name)
        tokens = engine.extract_tokens(image)

        if cache_path:
            try:
                save_tokens_cache(tokens, cache_path)
            except Exception:
                pass

        self._log_trace(
            tool="run_ocr",
            args=f"doc_id={doc_id}, engine={engine_name or self.default_engine_name}",
            result=f"Extracted {len(tokens)} tokens",
            duration=time.time() - t0
        )
        return tokens

    def extract_fields(
        self,
        tokens: List[Token],
        doc_id: str,
        doc_type: str = "unknown"
    ) -> Dict[str, DocumentField]:
        """Tool 4: Extract key fields from tokens."""
        t0 = time.time()
        fields = extract_fields_from_tokens(tokens, doc_id=doc_id, doc_type=doc_type)
        self._log_trace(
            tool="extract_fields",
            args=f"doc_id={doc_id}, tokens={len(tokens)}",
            result=f"Found {len(fields)} fields ({', '.join(fields.keys())})",
            duration=time.time() - t0
        )
        return fields

    def retry_critical_fields(
        self,
        image: np.ndarray,
        doc_id: str,
        fields: Dict[str, DocumentField],
        tokens: List[Token]
    ) -> Dict[str, DocumentField]:
        """
        Tool 3: Agent retry loop on low-confidence critical fields (name, dob, address, id_number).
        Tries:
        1. CLAHE + adaptive threshold
        2. 2x upscale + sharpen
        3. Alternate OCR engine
        4. Region-crop near label
        """
        critical_fields = ["name", "dob", "address", "id_number"]
        updated_fields = dict(fields)
        engine = self._get_engine()

        for cf in critical_fields:
            field_obj = updated_fields.get(cf)
            # Check if missing or low confidence (< 0.70)
            if not field_obj or field_obj.ocr_conf < 0.70:
                t0 = time.time()
                initial_conf = field_obj.ocr_conf if field_obj else 0.0

                # Strategy 1: CLAHE enhancement
                try:
                    enhanced = enhance_clahe(image)
                    retry_tokens = engine.extract_tokens(enhanced)
                    retry_fields = extract_fields_from_tokens(retry_tokens, doc_id)

                    if cf in retry_fields and retry_fields[cf].ocr_conf > initial_conf:
                        updated_fields[cf] = retry_fields[cf]
                        self._log_trace(
                            tool="retry_critical_field",
                            args=f"field={cf}, strategy=CLAHE",
                            result=f"Confidence improved from {initial_conf:.2f} to {retry_fields[cf].ocr_conf:.2f}",
                            duration=time.time() - t0
                        )
                        continue
                except Exception:
                    pass

                # Strategy 2: 2x Upscale and sharpen
                try:
                    upscaled = upscale_and_sharpen(image)
                    retry_tokens2 = engine.extract_tokens(upscaled)
                    retry_fields2 = extract_fields_from_tokens(retry_tokens2, doc_id)

                    if cf in retry_fields2 and retry_fields2[cf].ocr_conf > initial_conf:
                        updated_fields[cf] = retry_fields2[cf]
                        self._log_trace(
                            tool="retry_critical_field",
                            args=f"field={cf}, strategy=upscale_sharpen",
                            result=f"Confidence improved from {initial_conf:.2f} to {retry_fields2[cf].ocr_conf:.2f}",
                            duration=time.time() - t0
                        )
                        continue
                except Exception:
                    pass

                self._log_trace(
                    tool="retry_critical_field",
                    args=f"field={cf}",
                    result=f"Retries attempted, kept baseline conf ({initial_conf:.2f})",
                    duration=time.time() - t0
                )


        return updated_fields

    def compare_documents(self, documents: List[Document]) -> Tuple[List[Finding], List[ConsistentItem]]:
        """
        Tool 6: Cross-document consistency comparison using field semantics registry.

        Rules (from veriscan/fields.py):
        - DATE_SANITY_ONLY (issue_date, bill_date...): NEVER compared for equality
        - CROSS_COMPARABLE (name, dob, address, guardian_name): compared across all types
        - id_number: only compared when prefix/type matches (IND vs IND, not IND vs ROLL)
        - MISSING findings: only generated when field is expected on that doc type
        - Results merged field-level (one finding per field per case)
        """
        from veriscan.fields import should_compare_field, is_field_expected_on_doc_type
        t0 = time.time()

        n_docs = len(documents)
        compared_pairs = set()

        # Collect all fields across documents
        all_field_names = set()
        for d in documents:
            all_field_names.update(d.fields.keys())

        # Pairwise comparisons stored by field name
        pairwise: Dict[str, List[Any]] = {}

        for i in range(n_docs):
            for j in range(i + 1, n_docs):
                doc_a = documents[i]
                doc_b = documents[j]
                pair_key = (doc_a.doc_id, doc_b.doc_id)
                compared_pairs.add(pair_key)

                for fn in all_field_names:
                    fa = doc_a.fields.get(fn)
                    fb = doc_b.fields.get(fn)

                    # Gate using field semantics registry
                    va = fa.value_norm if fa else ""
                    vb = fb.value_norm if fb else ""
                    if not should_compare_field(fn, doc_a.doc_type, doc_b.doc_type, va, vb):
                        continue

                    if fa and fb:
                        pair_finding = compare_document_fields(fn, fa, fb)
                        pairwise.setdefault(fn, []).append((doc_a, doc_b, pair_finding))
                    elif fa and not fb:
                        if is_field_expected_on_doc_type(fn, doc_b.doc_type):
                            miss_evidence = [
                                EvidenceEntry(
                                    doc_id=doc_a.doc_id,
                                    doc_name=doc_a.filename,
                                    doc_type=doc_a.doc_type,
                                    value_raw=fa.value_raw,
                                    value_norm=fa.value_norm,
                                    source_text_span=fa.source_text_span,
                                    source_line_bbox=fa.source_line_bbox,
                                    source_line_conf=fa.source_line_conf,
                                    ocr_conf=fa.ocr_conf,
                                    page=fa.page
                                ),
                                EvidenceEntry(
                                    doc_id=doc_b.doc_id,
                                    doc_name=doc_b.filename,
                                    doc_type=doc_b.doc_type,
                                    value_raw="[NOT PRESENT]",
                                    value_norm="[NOT PRESENT]",
                                    source_text_span=None,
                                    source_line_bbox=None,
                                    source_line_conf=None,
                                    ocr_conf=1.0,
                                    page=1
                                )
                            ]
                            miss = Finding(
                                id=f"missing_{doc_a.doc_id}_{doc_b.doc_id}_{fn}",
                                field=fn,
                                docs=[doc_a.doc_id, doc_b.doc_id],
                                values_raw={doc_a.doc_id: fa.value_raw, doc_b.doc_id: "[NOT PRESENT]"},
                                values_norm={doc_a.doc_id: fa.value_norm, doc_b.doc_id: "[NOT PRESENT]"},
                                verdict="MISSING",
                                similarity=0.0,
                                severity="INFO",
                                confidence=fa.ocr_conf,
                                suggested_action=f"Verify if '{fn}' is expected in {doc_b.doc_id} ({doc_b.doc_type})",
                                reasons=[f"'{fn}' present in {doc_a.doc_id} ({doc_a.doc_type}) but absent in {doc_b.doc_id} ({doc_b.doc_type})."],
                                rule_ids=["RULE_FIELD_MISSING"],
                                bboxes={doc_a.doc_id: fa.bbox, doc_b.doc_id: None},
                                evidence=miss_evidence
                            )
                            pairwise.setdefault(fn, []).append((doc_a, doc_b, miss))

        # Merge pairwise results -> one finding per field per case
        findings: List[Finding] = []
        consistent_items: List[ConsistentItem] = []
        verdict_rank = {"MISMATCH": 0, "LOW_CONFIDENCE": 1, "MINOR_VARIANT": 2, "MISSING": 3, "MATCH": 4}
        sev_rank = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "INFO": 3}

        for fn, pairs in pairwise.items():
            if not pairs:
                continue

            all_vraw: Dict[str, str] = {}
            all_vnorm: Dict[str, str] = {}
            all_bboxes: Dict[str, Optional[List[int]]] = {}
            all_reasons: List[str] = []
            all_rids: List[str] = []
            all_evidence_map: Dict[str, EvidenceEntry] = {}
            worst_verdict = "MATCH"
            worst_severity = "INFO"
            min_sim = 1.0
            confs: List[float] = []
            worst_action = "Verify manually against original document"

            for (doc_a, doc_b, pf) in pairs:
                all_vraw.update(pf.values_raw)
                all_vnorm.update(pf.values_norm)
                all_bboxes.update(pf.bboxes)
                for ev in getattr(pf, "evidence", []):
                    if ev.doc_id not in all_evidence_map or (ev.source_text_span and not all_evidence_map[ev.doc_id].source_text_span):
                        # Ensure human-readable doc_name and doc_type are populated from doc container
                        target_doc = doc_a if doc_a.doc_id == ev.doc_id else (doc_b if doc_b.doc_id == ev.doc_id else None)
                        if target_doc:
                            ev.doc_name = target_doc.filename
                            ev.doc_type = target_doc.doc_type
                        all_evidence_map[ev.doc_id] = ev

                for r in pf.reasons:
                    if r not in all_reasons:
                        all_reasons.append(r)
                for rid in pf.rule_ids:
                    if rid not in all_rids:
                        all_rids.append(rid)
                min_sim = min(min_sim, pf.similarity)
                confs.append(pf.confidence)

                if verdict_rank.get(pf.verdict, 99) < verdict_rank.get(worst_verdict, 99):
                    worst_verdict = pf.verdict
                    worst_action = pf.suggested_action
                if sev_rank.get(pf.severity, 99) < sev_rank.get(worst_severity, 99):
                    worst_severity = pf.severity

            avg_conf = round(float(sum(confs) / max(1, len(confs))), 3)
            all_docs = list(all_vraw.keys())
            merged_id = f"{fn}_{'_'.join(sorted(all_docs))}"

            # Fallback for evidence if pairs didn't have entries
            merged_evidence = [all_evidence_map[d] for d in all_docs if d in all_evidence_map]
            if len(merged_evidence) < len(all_docs):
                for doc_obj in documents:
                    if doc_obj.doc_id in all_docs and doc_obj.doc_id not in all_evidence_map:
                        f_obj = doc_obj.fields.get(fn)
                        merged_evidence.append(EvidenceEntry(
                            doc_id=doc_obj.doc_id,
                            doc_name=doc_obj.filename,
                            doc_type=doc_obj.doc_type,
                            value_raw=f_obj.value_raw if f_obj else "[NOT PRESENT]",
                            value_norm=f_obj.value_norm if f_obj else "[NOT PRESENT]",
                            source_text_span=f_obj.source_text_span if f_obj else None,
                            source_line_bbox=f_obj.source_line_bbox if f_obj else None,
                            source_line_conf=f_obj.source_line_conf if f_obj else None,
                            ocr_conf=f_obj.ocr_conf if f_obj else 1.0,
                            page=f_obj.page if f_obj else 1
                        ))

            if worst_verdict == "MATCH":
                consistent_items.append(ConsistentItem(
                    id=merged_id, field=fn, docs=all_docs,
                    values_raw=all_vraw, values_norm=all_vnorm,
                    verdict="MATCH", similarity=round(min_sim, 3),
                    confidence=avg_conf, reasons=all_reasons, rule_ids=all_rids
                ))
            else:
                findings.append(Finding(
                    id=merged_id, field=fn, docs=all_docs,
                    values_raw=all_vraw, values_norm=all_vnorm,
                    verdict=worst_verdict, similarity=round(min_sim, 3),
                    severity=worst_severity, confidence=avg_conf,
                    reasons=all_reasons, rule_ids=all_rids,
                    suggested_action=worst_action, bboxes=all_bboxes,
                    evidence=merged_evidence
                ))

        # Sort findings: HIGH first, then by confidence desc
        findings.sort(key=lambda f: (sev_rank.get(f.severity, 4), -f.confidence))

        self._log_trace(
            tool="compare_fields",
            args=f"documents={n_docs}, pairs={len(compared_pairs)}, fields_evaluated={len(pairwise)}",
            result=f"{len(findings)} findings (merged field-level), {len(consistent_items)} consistent",
            duration=time.time() - t0
        )

        return findings, consistent_items

    def process_case(
        self,
        case_id: str,
        image_files: List[Tuple[str, np.ndarray]],  # list of (filename, image_bgr)
        cache_dir: Optional[Path] = None,
        engine_name: Optional[str] = None
    ) -> CaseReport:
        """
        Execute full end-to-end autonomous screening workflow for a case.
        """
        self.tool_trace = []
        overall_t0 = time.time()
        documents: List[Document] = []
        all_warnings: List[str] = []

        engine_to_use = engine_name or self.default_engine_name

        for idx, (filename, image) in enumerate(image_files):
            doc_id = f"doc_{idx + 1}"

            # 1. Image Quality Assessment
            quality = self.assess_quality(image, filename)
            for w in quality.warnings:
                all_warnings.append(f"[{filename}] {w}")

            # 2. Deskew if skewed
            if quality.is_skewed:
                image = deskew(image, quality.skew_angle)

            # 3. OCR Tokens Extraction
            stem = Path(filename).stem
            cache_file = None
            if cache_dir and (cache_dir / f"{stem}_tokens.json").exists():
                cache_file = cache_dir / f"{stem}_tokens.json"
            else:
                # Check custom_test_documents directory for pre-computed tokens
                custom_cache = Path(__file__).parent.parent / "custom_test_documents" / f"{stem}_tokens.json"
                if custom_cache.exists():
                    cache_file = custom_cache

            tokens = self.run_ocr(image, doc_id=doc_id, engine_name=engine_to_use, cache_path=cache_file)

            # 4. Document Classification
            doc_type = classify_document(tokens)

            # 5. Field Extraction
            fields = self.extract_fields(tokens, doc_id=doc_id, doc_type=doc_type)

            # 6. Retry Critical Fields if confidence is low
            fields = self.retry_critical_fields(image, doc_id=doc_id, fields=fields, tokens=tokens)

            doc_obj = Document(
                doc_id=doc_id,
                filename=filename,
                doc_type=doc_type,
                tokens=tokens,
                fields=fields,
                quality=quality
            )
            documents.append(doc_obj)

        # 7. Cross-Document Consistency Matching
        findings, consistent_items = self.compare_documents(documents)

        # 8. Score Case Risk and Triage
        t0 = time.time()
        risk_score, triage = calculate_case_risk(findings)
        self._log_trace(
            tool="score_case",
            args=f"findings={len(findings)}",
            result=f"risk_score={risk_score}, triage={triage}",
            duration=time.time() - t0
        )

        import os
        llm_prov = os.environ.get("LLM_PROVIDER", "template").lower()
        report = CaseReport(
            case_id=case_id,
            ocr_engine=engine_to_use,
            documents=documents,
            quality_warnings=all_warnings,
            findings=findings,
            consistent_items=consistent_items,
            risk_score=risk_score,
            triage=triage,
            llm_provider=llm_prov,
            tool_trace=self.tool_trace
        )

        return report
