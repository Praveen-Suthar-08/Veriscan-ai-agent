"""Narrative generation engine for VeriScan.

Generates plain-language executive summaries and suggested reviewer questions.
Uses LLM provider abstraction (Template, Anthropic, Gemini, Ollama) with:
- Deterministic TemplateProvider fallback by default
- Hash-based narrative caching
- Zero network required by default
- Zero ability to alter findings or verdicts
"""

from __future__ import annotations
import os
from typing import List, Dict, Any, Tuple, Optional
from veriscan.schemas import Finding, ConsistentItem
from veriscan.llm import (
    get_llm_provider, TemplateProvider, AnthropicProvider, GeminiProvider, OllamaProvider,
    compute_findings_hash, get_cached_narrative, set_cached_narrative,
    build_masked_findings_payload, get_last_fallback_note
)


def generate_deterministic_narrative(
    findings: List[Finding],
    consistent_items: List[ConsistentItem],
    risk_score: float,
    triage: str
) -> Tuple[str, List[str]]:
    """
    Generate deterministic, transparent executive narrative and reviewer questions
    via TemplateProvider. Zero external dependencies, zero hallucinations.
    """
    template_provider = TemplateProvider()
    return template_provider.generate_from_structured_data(
        findings, consistent_items, risk_score, triage
    )


def generate_narrative(
    findings: List[Finding],
    consistent_items: List[ConsistentItem],
    risk_score: float,
    triage: str,
    provider_override: Optional[str] = None
) -> Tuple[str, List[str]]:
    """
    Generate narrative and reviewer questions using the configured LLM provider.
    - Uses hash caching to avoid redundant calls.
    - Sends strictly structured, masked findings (no images, masked IDs).
    - Falls back to TemplateProvider on network timeout, error, or missing keys.
    """
    # 1. Build structured masked payload
    payload = build_masked_findings_payload(findings, consistent_items, risk_score, triage)
    f_hash = compute_findings_hash(payload)

    # 2. Check cache first
    cached = get_cached_narrative(f_hash)
    if cached is not None:
        summary, questions, _ = cached
        return summary, questions

    # 3. Determine provider
    provider_name = provider_override or os.environ.get("LLM_PROVIDER", "template").lower()
    provider = get_llm_provider(provider_name)

    # If TemplateProvider, generate directly
    if isinstance(provider, TemplateProvider):
        summary, questions = provider.generate_from_structured_data(
            findings, consistent_items, risk_score, triage
        )
        set_cached_narrative(f_hash, summary, questions, "template")
        return summary, questions

    # Remote provider (Anthropic, Gemini, Ollama)
    try:
        prompt = f"""
You are an expert KYC and identity verification analyst writing an objective, evidence-backed narrative for a human reviewer.
STRICT RULES:
1. You must ONLY restate and summarize the structured findings provided below.
2. DO NOT add, invent, assume, or omit any findings.
3. DO NOT make an approval or rejection verdict.
4. Output your answer in two sections:
   ### Summary
   (A 2-3 paragraph objective breakdown of matches, discrepancies, and scan issues)
   ### Reviewer Questions
   (2-4 bullet points suggesting exact verification questions for the applicant)

Structured Case Data:
- Triage: {triage}
- Risk Score: {risk_score:.2f}
- Findings: {payload['findings']}
- Consistent Fields: {payload['consistent_fields']}
"""
        response_text = provider.generate(prompt)

        # Parse sections
        parts = response_text.split("### Reviewer Questions")
        summary = parts[0].replace("### Summary", "").strip()
        questions: List[str] = []
        if len(parts) > 1:
            for q_line in parts[1].strip().split("\n"):
                q_clean = q_line.strip().lstrip("-*123456789. ")
                if q_clean:
                    questions.append(q_clean)

        # Deduplicate parsed questions
        questions = list(dict.fromkeys(questions))

        if not questions:
            questions = ["Review discrepancy details and confirm documentation with the applicant."]

        if not summary:
            # Fallback if empty
            summary, questions = generate_deterministic_narrative(findings, consistent_items, risk_score, triage)

        set_cached_narrative(f_hash, summary, questions, provider_name)
        return summary, questions

    except Exception:
        # Graceful fallback to deterministic template narrative
        summary, questions = generate_deterministic_narrative(findings, consistent_items, risk_score, triage)
        set_cached_narrative(f_hash, summary, questions, "template_fallback")
        return summary, questions
