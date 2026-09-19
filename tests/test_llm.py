"""Automated unit tests for LLM provider abstraction, caching, masking, and fallback."""

import os
from veriscan.schemas import Finding, ConsistentItem
from veriscan.llm import (
    TemplateProvider, AnthropicProvider, GeminiProvider, OllamaProvider,
    get_llm_provider, compute_findings_hash, get_cached_narrative,
    set_cached_narrative, build_masked_findings_payload
)
from veriscan.narrative import generate_narrative, generate_deterministic_narrative


def test_template_provider_deterministic():
    provider = TemplateProvider()
    findings = [
        Finding(
            id="f1", field="dob", docs=["doc1", "doc2"],
            values_raw={"doc1": "12/03/2004", "doc2": "15/03/2004"},
            values_norm={"doc1": "2004-03-12", "doc2": "2004-03-15"},
            verdict="MISMATCH", similarity=0.65, severity="HIGH", confidence=0.85,
            reasons=["Single digit difference detected"], suggested_action="Verify DOB"
        )
    ]
    summary, questions = provider.generate_from_structured_data(
        findings=findings, consistent_items=[], risk_score=0.85, triage="RED"
    )
    assert "DOB" in summary
    assert "RED" in summary
    assert len(questions) >= 1
    assert "date of birth" in questions[0].lower()


def test_provider_factory_default(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "template")
    provider = get_llm_provider()
    assert isinstance(provider, TemplateProvider)


def test_fallback_on_missing_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    provider_anthropic = get_llm_provider("anthropic")
    assert isinstance(provider_anthropic, TemplateProvider)

    provider_gemini = get_llm_provider("gemini")
    assert isinstance(provider_gemini, TemplateProvider)


def test_findings_hash_and_cache():
    test_data = {"test": 123, "fields": ["name", "dob"]}
    h1 = compute_findings_hash(test_data)
    h2 = compute_findings_hash(test_data)
    assert h1 == h2

    set_cached_narrative(h1, "Summary Cached", ["Question 1"], "template")
    retrieved = get_cached_narrative(h1)
    assert retrieved is not None
    assert retrieved[0] == "Summary Cached"
    assert retrieved[1] == ["Question 1"]


def test_id_masking_in_payload():
    findings = [
        Finding(
            id="f_id", field="id_number", docs=["doc1", "doc2"],
            values_raw={"doc1": "1234 5678 9012", "doc2": "1234 5678 9999"},
            values_norm={"doc1": "123456789012", "doc2": "123456789999"},
            verdict="MISMATCH", similarity=0.0, severity="HIGH", confidence=0.90,
            reasons=["ID number discrepancy"], suggested_action="Check ID"
        )
    ]
    payload = build_masked_findings_payload(findings, [], 0.90, "RED")
    masked_finding = payload["findings"][0]

    # Verify original raw numbers are masked
    assert masked_finding["values_raw"]["doc1"] != "1234 5678 9012"
    assert "****" in masked_finding["values_raw"]["doc1"]
    assert masked_finding["values_norm"]["doc1"] != "123456789012"
    assert "****" in masked_finding["values_norm"]["doc1"]


def test_narrative_generation_never_alters_findings():
    findings = [
        Finding(
            id="f_name", field="name", docs=["doc1", "doc2"],
            values_raw={"doc1": "Rohit Sharma", "doc2": "R Sharma"},
            values_norm={"doc1": "rohit sharma", "doc2": "rohit sharma"},
            verdict="MINOR_VARIANT", similarity=0.88, severity="MEDIUM", confidence=0.92,
            reasons=["Initial expansion match"], suggested_action="Confirm"
        )
    ]
    original_verdict = findings[0].verdict
    original_conf = findings[0].confidence

    summary, questions = generate_narrative(findings, [], 0.0, "GREEN")
    assert findings[0].verdict == original_verdict
    assert findings[0].confidence == original_conf
    assert "NAME" in summary or "Rohit" in summary or "Accepted Minor Variations" in summary
