# VeriScan — 3-Minute Hackathon Demo Script
### Autonomous Document & Identity Consistency-Checking Agent

**Judge Value Proposition:**  
*"VeriScan is a pre-screening agent that reconciles names, dates, addresses, and IDs across uploaded documents, isolating real discrepancies while ignoring benign cultural variants, and hands reviewers a ranked, evidence-backed report with visual bounding-box highlights. It's an intelligent screening aid, never a verdict. It runs 100% locally with zero cloud API keys required."*

---

## ⏱️ Minute 0:00 – 0:45: The Problem & Benign Variant Intelligence
1. **Open the App:**  
   Launch `streamlit run app/main.py`. Point out the sleek cyber-command interface, the **Agent Status: Autonomous Core Active** indicator, and the **Mandatory Legal Pre-Screening Disclaimer** displayed prominently.
2. **Scenario 1: Benign Variants (`case_02_benign`):**
   - Select **Case 02: Benign Variants Only** from the intake dropdown. Click **🚀 Load and Screen Sample Case**.
   - Show the staged progress indicators: *Quality check ➔ OCR ➔ Extraction & Retry ➔ Consistency ➔ Report*.
   - **The Reveal:** Triage is **GREEN** (Risk Score: 0.00) with **0 Mismatches**!
   - Navigate to the **"✅ Checked & Consistent"** tab:
     - Show how `R. Sharma` matched `Rohit Sharma` via initial expansion.
     - Show how `Rd` expanded to `Road` and `Blr` to `Bengaluru`.
     - *Key Judge Takeaway:* "Naive systems flag 68% of these as discrepancies. VeriScan transparently shows its work and accepts benign variants without wasting reviewer time."

---

## ⏱️ Minute 0:45 – 1:45: Genuine Discrepancies & Evidence Overlays
1. **Scenario 2: Real Discrepancies (`case_03_mismatch`):**
   - Select **Case 03: Discrepancies (DOB & Pincode Mismatch)**. Click Screen.
   - **The Reveal:** Triage switches immediately to **RED** (Risk Score: 0.88).
2. **Examine Ranked Flags:**
   - **DOB Mismatch (HIGH Severity):** Shows `2004-03-12` vs `2004-03-15`. Subtype correctly diagnosed as *Single Digit Typo*.
   - **Confidence Breakdown Bar:** Point to `0.89` overall confidence, decomposed into OCR token confidences and signal certainty.
3. **Interactive Evidence Viewer:**
   - Click **"🔍 Highlight Evidence in Viewer"** on the DOB card.
   - The right-hand document viewer instantly highlights the exact physical bounding box in red on the source document image!
4. **Human Reviewer Actions:**
   - Directly on the card, select **Confirm Issue** and enter a note: *"Applicant confirmed clerical typo on marksheet."*
   - Click Confirm. Notice the status updates to **CONFIRMED** and is written directly to the SQLite audit log.

---

## ⏱️ Minute 1:45 – 2:30: Scan Degradation & Safe Gating ("Not Sure is Valid")
1. **Scenario 3: Degraded Scan (`case_04_degraded`):**
   - Select **Case 04: Degraded Scan**. Click Screen.
   - Show the **Image Quality Warnings** tab: flags Gaussian blur and skew angle.
   - Show the **Agent Execution Trace** tab: point to the progressive retry loop:
     *`retry_critical_field -> CLAHE enhancement -> 2x upscale & sharpen`*.
   - Point to the **LOW_CONFIDENCE** flag:
     - *Key Judge Takeaway:* "Because the scan quality was below 0.60, VeriScan did NOT hallucinate a false mismatch. It safely reported LOW_CONFIDENCE, suggesting a clearer re-upload."

---

## ⏱️ Minute 2:30 – 3:00: Custom Uploads, PDF Export & Empirical Proof
1. **Option B Custom Document Upload:**
   - Show judges the `custom_test_documents/` folder with 10 synthetic documents.
   - Drag and drop `01_Aadhaar_Aarav_Sharma.png` + `04_Marksheet_Aarav_Sharma_DOB_Mismatch.png` into Option B.
   - Proves the system handles arbitrary drag-and-drop uploads in real-time, not just hardcoded demo cases.
2. **Download Official PDF Report:**
   - Switch to **Export Reports (PDF & JSON)**. Click **"Download Official PDF Report"**.
   - Open the generated PDF: Show the triage badge, findings breakdown, and the **mandatory legal disclaimer printed on every single page**.
3. **Model Evaluation Benchmarks Tab:**
   - Navigate to Screen 4 (**"Model Evaluation Benchmarks"**).
   - Show judges the empirical charts generated from the 40+ dev cases and 12 held-out layout cases:
     - Full Pipeline F1: **0.95** vs Naive F1: **0.64**.
     - Benign False Positive Rate reduced from **68.4% to 0.0%**.
     - Held-Out Layout F1: **0.94** proving the extraction is layout-agnostic, not hardcoded.
4. **Close:**  
   *"VeriScan brings deterministic rigor, zero hallucination risk, and complete human-in-the-loop explainability to high-stakes document verification."*
