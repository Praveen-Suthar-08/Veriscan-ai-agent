"""LLM provider abstraction for VeriScan.

Provides:
- Abstract LLMProvider interface: generate(prompt) -> str
- TemplateProvider: default deterministic generator requiring zero network
- AnthropicProvider: Anthropic Claude (ANTHROPIC_API_KEY, ANTHROPIC_MODEL)
- GeminiProvider: Google Gemini (GEMINI_API_KEY, GEMINI_MODEL)
- OllamaProvider: Local Ollama (OLLAMA_URL, OLLAMA_MODEL)
- Provider selection via environment variable LLM_PROVIDER
- Graceful fallback to TemplateProvider on error or missing keys
- Narrative caching by hash of findings JSON
- Strict data masking (ID numbers masked, no images sent)
"""

from __future__ import annotations
import os
import json
import hashlib
from pathlib import Path
import urllib.request
import urllib.error
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List

# Auto-load .env if present
def _load_env_file():
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k, v = k.strip(), v.strip().strip("\"'")
                        if k and k not in os.environ:
                            os.environ[k] = v
        except Exception:
            pass

_load_env_file()

# In-memory narrative cache by findings hash
_NARRATIVE_CACHE: Dict[str, Tuple[str, List[str], str]] = {}
_LAST_FALLBACK_NOTE: Optional[str] = None


def compute_findings_hash(findings_data: Any) -> str:
    """Compute a deterministic SHA-256 hash of structured findings."""
    serialized = json.dumps(findings_data, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def get_cached_narrative(findings_hash: str) -> Optional[Tuple[str, List[str], str]]:
    """Retrieve cached narrative (summary, questions, provider_used) if present."""
    return _NARRATIVE_CACHE.get(findings_hash)


def set_cached_narrative(
    findings_hash: str,
    summary: str,
    questions: List[str],
    provider_used: str
) -> None:
    """Cache narrative output for a given findings hash."""
    _NARRATIVE_CACHE[findings_hash] = (summary, questions, provider_used)


def get_last_fallback_note() -> Optional[str]:
    """Return warning note if an LLM provider failed and fell back to template."""
    return _LAST_FALLBACK_NOTE


def _mask_id_value(val: str) -> str:
    """Mask sensitive ID numbers (e.g. 1234 5678 9012 -> **** **** 9012)."""
    clean = str(val).strip()
    if len(clean) <= 4:
        return "****"
    return "*" * (len(clean) - 4) + clean[-4:]


def build_masked_findings_payload(
    findings: List[Any],
    consistent_items: List[Any],
    risk_score: float,
    triage: str
) -> Dict[str, Any]:
    """
    Format findings into structured, masked payload for LLM narration.
    Masks ID numbers and never includes images or unvalidated data.
    """
    masked_findings = []
    for f in findings:
        f_dict = f.model_dump() if hasattr(f, "model_dump") else dict(f)
        field_name = str(f_dict.get("field", "")).lower()
        
        # Mask raw and norm values if ID field
        if "id" in field_name or "roll" in field_name or "number" in field_name:
            if "values_raw" in f_dict and isinstance(f_dict["values_raw"], dict):
                f_dict["values_raw"] = {
                    k: _mask_id_value(v) for k, v in f_dict["values_raw"].items()
                }
            if "values_norm" in f_dict and isinstance(f_dict["values_norm"], dict):
                f_dict["values_norm"] = {
                    k: _mask_id_value(v) for k, v in f_dict["values_norm"].items()
                }

        masked_findings.append({
            "field": f_dict.get("field"),
            "docs": f_dict.get("docs"),
            "verdict": f_dict.get("verdict"),
            "severity": f_dict.get("severity"),
            "confidence": f_dict.get("confidence"),
            "values_raw": f_dict.get("values_raw"),
            "values_norm": f_dict.get("values_norm"),
            "reasons": f_dict.get("reasons"),
            "suggested_action": f_dict.get("suggested_action"),
        })

    masked_consistent = []
    for c in consistent_items:
        c_dict = c.model_dump() if hasattr(c, "model_dump") else dict(c)
        field_name = str(c_dict.get("field", "")).lower()
        if "id" in field_name:
            continue
        masked_consistent.append({
            "field": c_dict.get("field"),
            "docs": c_dict.get("docs"),
            "reasons": c_dict.get("reasons"),
        })

    return {
        "triage": triage,
        "risk_score": round(risk_score, 3),
        "findings": masked_findings,
        "consistent_fields": masked_consistent
    }


class LLMProvider(ABC):
    """Abstract interface for all LLM providers."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate text completion from prompt."""
        pass


class TemplateProvider(LLMProvider):
    """Default deterministic template provider requiring zero network access."""

    def generate(self, prompt: str) -> str:
        # Prompt usually contains structured JSON or is called via helper
        return (
            "### Summary\n"
            "Screening completed deterministically using rule-based reconciliation.\n\n"
            "### Reviewer Questions\n"
            "- Verify all flagged items against physical original proofs."
        )

    def generate_from_structured_data(
        self,
        findings: List[Any],
        consistent_items: List[Any],
        risk_score: float,
        triage: str
    ) -> Tuple[str, List[str]]:
        """Generate clear, structured narrative from findings."""
        mismatches = [f for f in findings if getattr(f, "verdict", "") == "MISMATCH"]
        minor_variants = [f for f in findings if getattr(f, "verdict", "") == "MINOR_VARIANT"]
        low_confs = [f for f in findings if getattr(f, "verdict", "") == "LOW_CONFIDENCE"]

        lines: List[str] = []

        # Executive summary
        if triage == "GREEN":
            lines.append(
                f"**Executive Summary (Triage: GREEN, Risk: {risk_score:.2f}):**\n"
                f"The automated pre-screening did not identify critical identity discrepancies. "
                f"All examined identity fields (name, date of birth, address, ID numbers) are substantially consistent "
                f"across submitted documents."
            )
        elif triage == "AMBER":
            lines.append(
                f"**Executive Summary (Triage: AMBER, Risk: {risk_score:.2f}):**\n"
                f"Moderate discrepancies or scan quality limitations were detected across documents. "
                f"Human reviewer verification is recommended before proceeding."
            )
        else:  # RED
            lines.append(
                f"**Executive Summary (Triage: RED, Risk: {risk_score:.2f}):**\n"
                f"High-severity cross-document inconsistencies were identified. "
                f"Crucial identity markers disagree between submitted proofs and require manual investigation."
            )

        if mismatches:
            lines.append("\n**Key Inconsistencies Flagged:**")
            for m in mismatches:
                reasons_str = "; ".join(getattr(m, "reasons", []))
                docs_str = ", ".join(getattr(m, "docs", []))
                lines.append(
                    f"- **{m.field.upper()}** ({m.severity} Severity): Discrepancy between {docs_str}. "
                    f"Details: {reasons_str}."
                )

        if low_confs:
            lines.append("\n**Low-Confidence / Quality Caveats:**")
            for lc in low_confs:
                reasons_str = "; ".join(getattr(lc, "reasons", []))
                lines.append(
                    f"- **{lc.field.upper()}**: OCR confidence was insufficient for deterministic verification. "
                    f"Details: {reasons_str}."
                )

        if minor_variants:
            lines.append("\n**Accepted Minor Variations:**")
            for mv in minor_variants:
                reasons_str = "; ".join(getattr(mv, "reasons", []))
                lines.append(f"- **{mv.field.upper()}**: {reasons_str}.")

        # Suggested Reviewer Questions — deduplicated, severity-ordered, max 5
        seen_keys: set = set()
        ordered_questions: List[str] = []

        def _add_question(key: str, text: str) -> None:
            if key not in seen_keys and len(ordered_questions) < 5:
                seen_keys.add(key)
                ordered_questions.append(text)

        # HIGH severity mismatches first
        for m in sorted(mismatches, key=lambda x: 0 if x.severity == "HIGH" else (1 if x.severity == "MEDIUM" else 2)):
            f_name = getattr(m, "field", "").lower()
            if "dob" in f_name or "date" in f_name:
                _add_question("dob", "Can the applicant confirm the official date of birth with a primary birth certificate or matriculation certificate?")
            elif "name" in f_name:
                _add_question("name", "Is there legal documentation (gazette notification / marriage certificate) explaining the name variance between documents?")
            elif "address" in f_name:
                _add_question("address", "Does the difference in address reflect permanent vs present residence, or is a recent utility bill / address proof required?")
            elif "id" in f_name:
                _add_question("id", "Can the applicant re-submit an unredacted, high-resolution copy of the official ID proof?")

        # LOW_CONFIDENCE items
        for lc in low_confs:
            _add_question(f"lc_{lc.field}", f"Can a higher-resolution or uncompressed scan of the document containing '{lc.field}' be requested?")

        questions: List[str] = ordered_questions

        if not questions:
            questions.append("No follow-up questions required. Record conforms to consistency rules.")

        return "\n".join(lines), questions



class AnthropicProvider(LLMProvider):
    """Anthropic Claude LLM provider."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.model = model or os.environ.get("ANTHROPIC_MODEL", "claude-3-haiku-20240307")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set.")

    def generate(self, prompt: str) -> str:
        try:
            anthropic = __import__("anthropic")
            client = anthropic.Anthropic(api_key=self.api_key)
            response = client.messages.create(
                model=self.model,
                max_tokens=600,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text.strip()
        except Exception:
            # Fallback to direct HTTPS request
            url = "https://api.anthropic.com/v1/messages"
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }
            body = json.dumps({
                "model": self.model,
                "max_tokens": 600,
                "temperature": 0.0,
                "messages": [{"role": "user", "content": prompt}]
            }).encode("utf-8")
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=12) as res:
                data = json.loads(res.read().decode("utf-8"))
                return data["content"][0]["text"].strip()


class GeminiProvider(LLMProvider):
    """Google Gemini LLM provider."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.model = model or os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not set.")

    def generate(self, prompt: str) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        headers = {"content-type": "application/json"}
        payload = json.dumps({
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": 600}
        }).encode("utf-8")

        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=12) as res:
            data = json.loads(res.read().decode("utf-8"))
            candidates = data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts:
                    return parts[0].get("text", "").strip()
            return "No narrative generated by Gemini."


class OllamaProvider(LLMProvider):
    """Local or Hosted Ollama LLM provider."""

    def __init__(self, url: Optional[str] = None, model: Optional[str] = None, api_key: Optional[str] = None):
        self.url = url or os.environ.get("OLLAMA_URL", "http://localhost:11434")
        self.model = model or os.environ.get("OLLAMA_MODEL", "llama3")
        self.api_key = api_key or os.environ.get("OLLAMA_API_KEY", "")

    def generate(self, prompt: str) -> str:
        endpoint = f"{self.url.rstrip('/')}/api/generate"
        headers = {"content-type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0}
        }).encode("utf-8")

        req = urllib.request.Request(endpoint, data=payload, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=10) as res:
            data = json.loads(res.read().decode("utf-8"))
            return data.get("response", "").strip()


def get_llm_provider(name: Optional[str] = None) -> LLMProvider:
    """
    Factory function returning the configured LLM provider.
    Select by env LLM_PROVIDER=template|anthropic|gemini|ollama (default: template).
    """
    provider_name = (name or os.environ.get("LLM_PROVIDER", "template")).lower().strip()

    if provider_name == "anthropic":
        try:
            return AnthropicProvider()
        except Exception as e:
            global _LAST_FALLBACK_NOTE
            _LAST_FALLBACK_NOTE = f"AnthropicProvider failed ({e}). Reverted to TemplateProvider."
            return TemplateProvider()

    elif provider_name == "gemini":
        try:
            return GeminiProvider()
        except Exception as e:
            _LAST_FALLBACK_NOTE = f"GeminiProvider failed ({e}). Reverted to TemplateProvider."
            return TemplateProvider()

    elif provider_name == "ollama":
        try:
            return OllamaProvider()
        except Exception as e:
            _LAST_FALLBACK_NOTE = f"OllamaProvider failed ({e}). Reverted to TemplateProvider."
            return TemplateProvider()

    else:
        return TemplateProvider()
