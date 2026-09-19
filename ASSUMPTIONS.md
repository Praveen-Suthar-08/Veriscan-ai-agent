# VeriScan — Assumptions and Engineering Decisions

This document records architectural, algorithmic, and engineering assumptions made during the autonomous build of **VeriScan** for the National AI Hackathon 2026, Track C (Problem C2: Document & Identity Consistency-Checking Agent).

---

## 1. Regulatory and Compliance Boundary
- **Pre-screening Aid Exclusivity:** VeriScan is strictly a screening and consistency reconciliation aid. It provides no verdicts on applicant authenticity, document genuineness, legal eligibility, or final application approvals.
- **Human-in-the-Loop:** All flagged discrepancies, minor variants, and triage ratings are subject to human reviewer confirmation, dismissal, or request for re-upload.
- **Mandatory Disclaimer:** The exact disclaimer string is required across all user-facing interfaces, PDF exports, JSON payloads, and documentation.
- **Zero Real PII:** All operational, bundled, and benchmark documents are 100% synthetically generated with Faker (`en_IN`) and explicit sample watermarks.

---

## 2. OCR and Ingestion
- **Dual-Engine Architecture:** VeriScan abstracts OCR behind `OCREngine` with implementations for both `EasyOCR` and `Tesseract` (`pytesseract`).
- **Default Engine Selection:** EasyOCR is selected as the initial default due to self-contained neural text detection and recognition across varied typography without requiring system-level binary dependencies. Tesseract is available as a selectable or fallback engine.
- **Confidence Calibration and Low-Confidence Gating:** If OCR token confidence on either compared document falls below the threshold (default: 0.60), the consistency verdict is categorized as `LOW_CONFIDENCE` rather than asserting a false `MISMATCH`.
- **Pre-processing and Retry Pipeline:** If a critical field (Name, DOB, ID, Address) has OCR confidence below 0.70 or fails schema validation, the agent triggers progressive retries:
  1. Contrast-Limited Adaptive Histogram Equalization (CLAHE) + adaptive threshold.
  2. 2x bicubic upscale + unsharp mask sharpening.
  3. Alternate OCR engine invocation.
  4. Bounding-box localized re-OCR near detected label regions.

---

## 3. Normalization and Consistency Logic
- **Pure Functions:** Normalization functions are pure, side-effect free, and return `(normalized_value, [rule_ids_applied])` to enable full explainability in the "Checked and consistent" audit view.
- **Names:**
  - Token-set matching handles order inversion (`Sharma, Rohit` vs `Rohit Sharma`).
  - Initials are expanded (e.g. `R. Kumar` matches `Rohit Kumar` when the initial letter agrees).
  - Configurable phonetic and transliteration variant dictionary handles common Indian name spellings (e.g., `Mohd` / `Mohammed` / `Muhammad`, `Laxmi` / `Lakshmi`).
  - RapidFuzz token set ratio, Levenshtein, Jaro-Winkler, and Metaphone are combined in a weighted similarity formula:
    - `>= 0.92`: `MATCH`
    - `0.75 – 0.92`: `MINOR_VARIANT`
    - `< 0.75`: `MISMATCH`
- **Dates of Birth:**
  - Default parsing order is day-first (`DD/MM/YYYY`) matching standard Indian document conventions.
  - Dates are normalized to canonical ISO-8601 (`YYYY-MM-DD`).
  - Non-matches are categorized into subtypes: `single_digit_typo`, `day_month_swap`, `year_off`, or `different`.
- **Addresses:**
  - Standardized expansion of common Indian address abbreviations (`rd`, `st`, `nr`, `opp`, `apt`, `blr`, `po`, etc.).
  - Component-weighted evaluation: Pincode (30%), City/State (20%), Street/Locality (30%), House/Premises (20%).
  - Pincode mismatch alone is weighted as `LOW` severity when cities/states align.
- **ID Numbers:**
  - Separators (spaces, hyphens) stripped.
  - OCR confusion corrections (`O` <-> `0`, `I`/`l` <-> `1`, `S` <-> `5`, `B` <-> `8`) are applied strictly within numeric fields.

---

## 4. Scoring and Triage
- **Finding Confidence:**
  $$\text{conf} = \text{ocr\_a} \times \text{ocr\_b} \times \text{similarity\_signal}$$
- **Case Risk Score:**
  $$\text{case\_risk} = 1 - \prod_{i \in \text{findings}} (1 - w_i \times \text{conf}_i)$$
  where weights $w_i$: High = 1.0, Medium = 0.6, Low = 0.3, Info = 0.0.
- **Triage Thresholds:**
  - `GREEN`: Risk < 0.20
  - `AMBER`: 0.20 <= Risk <= 0.60
  - `RED`: Risk > 0.60
- **Risk Score Semantics:** Explicitly presented in the UI and report as an empirical heuristic risk indicator, never a probability.

---

## 5. Offline and Hackathon Demo Reliability
- **Pre-cached OCR for Bundled Demo Cases:** 4 bundled cases (`consistent`, `benign_variants`, `dob_pincode_mismatch`, `degraded_scan`) include pre-computed OCR caches to guarantee instantaneous, glitch-free judge demonstrations.
- **Offline Narrative Engine:** A deterministic narrative generator is the default narrator. Anthropic Claude LLM generation is optional and strictly restates structured findings when an API key is provided.
