# Custom Test Documents — VeriScan
### 10 Synthetic Test Documents for Option B Upload Testing

> **MANDATORY LEGAL DISCLAIMER:**  
> **All documents in this folder are 100% synthetic mock records generated using Pillow and Faker (en_IN). Each file contains the prominent diagonal watermark: `SAMPLE – NOT A REAL DOCUMENT`. No real personal data or government templates are used.**

---

## 📁 Document Inventory & Test Scenarios

This directory contains **10 realistic, diverse documents** designed to test VeriScan's layout-agnostic OCR, cultural name normalizer, multi-format date parser, address component reconciliation, and scan quality retry loops.

| # | File Name | Document Type | Key Test Focus | Expected Triage Outcome |
|:---:|---|---|---|:---:|
| **01** | `01_Aadhaar_Aarav_Sharma.png` | Aadhaar / National ID Card | Baseline Record: Aarav Sharma, DOB: 15/08/1999, Bengaluru 560001 | — |
| **02** | `02_Marksheet_Aarav_Sharma_Consistent.png` | Class 10/12 Marksheet | **Consistent Baseline:** Matches Doc 01 exactly (DOB formatted as `15-Aug-1999`) | 🟢 **GREEN** (0 Mismatches) |
| **03** | `03_Utility_Bill_Aarav_Sharma_Benign_Variant.png` | State Electricity Supply Bill | **Benign Variant Test:** Name abbreviated as `A. Sharma`, address as `42 MG Rd, Blr` | 🟢 **GREEN** (Initial & Abbrev accepted) |
| **04** | `04_Marksheet_Aarav_Sharma_DOB_Mismatch.png` | Academic Marksheet | **Real Discrepancy Test:** DOB is `18-Aug-1999` (Single-digit clerical typo vs `15/08/1999`) | 🔴 **RED** (High-Severity DOB Mismatch) |
| **05** | `05_ID_Card_Priya_Patel.png` | Driving Licence ID Card | Baseline Record: Priya Patel, Mumbai 400001, DOB: 22/01/2001 | — |
| **06** | `06_Application_Form_Priya_Patel_Consistent.png` | University Application Form | **Consistent Baseline:** Matches Doc 05 exactly (DOB formatted as `2001-01-22`) | 🟢 **GREEN** (0 Mismatches) |
| **07** | `07_Utility_Bill_Priya_Patel_Address_Mismatch.png` | Electricity Bill | **Pincode & Variant Test:** Surname spelled `Priya Patil` (benign variant) but pincode is `400050` vs `400001` | 🟡 **AMBER** (Address Pincode Discrepancy) |
| **08** | `08_Bank_Statement_Vikram_Verma.png` | Commercial Bank Statement | Baseline Record: Vikram Verma, New Delhi 110001 | — |
| **09** | `09_ID_Card_Vikram_Verma_Consistent.png` | Voter Identity Card | **Consistent Baseline:** Matches Doc 08 | 🟢 **GREEN** (0 Mismatches) |
| **10** | `10_ID_Card_Vikram_Verma_Degraded_Scan.png` | Degraded Mobile Scan | **Scan Quality Stress Test:** 3.5° rotation & Gaussian blur. Triggers CLAHE + 2x upscale retry loop | 🟡 **AMBER** (`LOW_CONFIDENCE` gating) |

---

## 🧪 Recommended Test Combinations for Live Judging

To test in the Streamlit web application:
1. Navigate to **`1. Case Intake & Screening`**.
2. Scroll to **Option B: Upload Custom Document Set**.
3. Drag and drop any of the following combinations:

### Scenario 1: Clean Cross-Reconciliation (Baseline)
- **Upload:** `01_Aadhaar_Aarav_Sharma.png` + `02_Marksheet_Aarav_Sharma_Consistent.png`
- **What happens:** VeriScan reconciles Name, DOB (`15/08/1999` vs `15-Aug-1999`), and Father Name.
- **Outcome:** **GREEN Triage** (Risk Score: `0.00`). All fields filed under *"Checked & Consistent"*.

### Scenario 2: Benign Cultural Variants (Zero Over-Flagging)
- **Upload:** `01_Aadhaar_Aarav_Sharma.png` + `03_Utility_Bill_Aarav_Sharma_Benign_Variant.png`
- **What happens:** Tests initial expansion (`A. Sharma` ➔ `Aarav Sharma`) and address abbreviation expansion (`Rd` ➔ `Road`, `Blr` ➔ `Bengaluru`).
- **Outcome:** **GREEN Triage** (Risk Score: `0.00`). Proves VeriScan does not waste reviewer time on harmless formatting differences.

### Scenario 3: Real Discrepancy Detection (Single-Digit DOB Typo)
- **Upload:** `01_Aadhaar_Aarav_Sharma.png` + `04_Marksheet_Aarav_Sharma_DOB_Mismatch.png`
- **What happens:** Detects `1999-08-15` vs `1999-08-18`. Diagnoses subtype: *Single-Digit Typo*.
- **Outcome:** **RED Triage** (Risk Score: `0.88`). Flags issue with high severity and enables red bounding-box evidence overlay.

### Scenario 4: Address Discrepancy & Transliteration Variant
- **Upload:** `05_ID_Card_Priya_Patel.png` + `07_Utility_Bill_Priya_Patel_Address_Mismatch.png`
- **What happens:** Accepts `Patil` as a benign transliteration variant of `Patel`, but flags the postal code mismatch (`400001` vs `400050`).
- **Outcome:** **AMBER Triage** (Low-Severity Pincode finding).

### Scenario 5: Blurry/Rotated Scan & Progressive Retry Loop
- **Upload:** `09_ID_Card_Vikram_Verma_Consistent.png` + `10_ID_Card_Vikram_Verma_Degraded_Scan.png`
- **What happens:** Agent detects blur (Laplacian variance < 100) and rotation. Automatically executes progressive retry (`CLAHE contrast enhancement` ➔ `2x upscale & sharpen`).
- **Outcome:** Emits `LOW_CONFIDENCE` rather than hallucinating a false mismatch (*"Not sure is a valid answer"*).
