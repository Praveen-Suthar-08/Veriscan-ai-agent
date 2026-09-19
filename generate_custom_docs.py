"""Generate 10 custom test documents for VeriScan upload demonstration.
All documents are 100% synthetic, realistic, and watermarked: SAMPLE – NOT A REAL DOCUMENT.
"""

import os
from pathlib import Path
from PIL import Image, ImageFilter
from veriscan.data.generate_mock import (
    _render_id_card, _render_marksheet, _render_utility_bill,
    _render_bank_statement, _render_application_form, _draw_watermark,
    WATERMARK_TEXT
)

CUSTOM_DIR = Path(__file__).parent / "custom_test_documents"
CUSTOM_DIR.mkdir(exist_ok=True)

docs_spec = [
    # 1. Aarav Sharma ID Card
    ("01_Aadhaar_Aarav_Sharma.png", _render_id_card, {
        "name": "Aarav Sharma", "guardian": "Suresh Sharma", "dob": "15/08/1999",
        "address": "42 MG Road, Bengaluru 560001", "id_number": "AADH-4920-1849", "issue_date": "2020-03-15"
    }),
    # 2. Aarav Sharma Consistent Marksheet
    ("02_Marksheet_Aarav_Sharma_Consistent.png", _render_marksheet, {
        "name": "Aarav Sharma", "guardian": "Suresh Sharma", "dob": "15-Aug-1999",
        "id_number": "ROLL-849201", "issue_date": "2017-05-20"
    }),
    # 3. Aarav Sharma Benign Variant Bill (Initial & Abbreviation)
    ("03_Utility_Bill_Aarav_Sharma_Benign_Variant.png", _render_utility_bill, {
        "name": "A. Sharma", "address": "42 MG Rd, Blr 560001",
        "id_number": "CON-9402184", "issue_date": "2023-11-04"
    }),
    # 4. Aarav Sharma Marksheet with DOB Typo Mismatch (Single-digit typo 15 vs 18)
    ("04_Marksheet_Aarav_Sharma_DOB_Mismatch.png", _render_marksheet, {
        "name": "Aarav Sharma", "guardian": "Suresh Sharma", "dob": "18-Aug-1999",
        "id_number": "ROLL-849201", "issue_date": "2017-05-20"
    }),
    # 5. Priya Patel ID Card
    ("05_ID_Card_Priya_Patel.png", _render_id_card, {
        "name": "Priya Patel", "guardian": "Kishore Patel", "dob": "22/01/2001",
        "address": "15 Marine Drive, Mumbai 400001", "id_number": "DL-MH-928104", "issue_date": "2021-02-10"
    }),
    # 6. Priya Patel Consistent Application Form
    ("06_Application_Form_Priya_Patel_Consistent.png", _render_application_form, {
        "name": "Priya Patel", "guardian": "Kishore Patel", "dob": "2001-01-22",
        "address": "15 Marine Drive, Mumbai 400001", "id_number": "APP-2024-8192", "issue_date": "2024-01-05"
    }),
    # 7. Priya Patel Bill with Pincode Discrepancy & Name Variant (Patil vs Patel)
    ("07_Utility_Bill_Priya_Patel_Address_Mismatch.png", _render_utility_bill, {
        "name": "Priya Patil", "address": "15 Marine Drive, Mumbai 400050",
        "id_number": "CON-8472910", "issue_date": "2023-12-01"
    }),
    # 8. Vikram Verma Bank Statement
    ("08_Bank_Statement_Vikram_Verma.png", _render_bank_statement, {
        "name": "Vikram Verma", "address": "77 Connaught Place, New Delhi 110001",
        "id_number": "ACC-938210492", "issue_date": "2023-09-15"
    }),
    # 9. Vikram Verma Consistent ID Card
    ("09_ID_Card_Vikram_Verma_Consistent.png", _render_id_card, {
        "name": "Vikram Verma", "guardian": "Ramesh Verma", "dob": "10-Nov-1995",
        "address": "77 Connaught Place, New Delhi 110001", "id_number": "VOT-DL-849201", "issue_date": "2019-06-20"
    }),
    # 10. Vikram Verma Degraded Scan (Blur & Skew for Retry loop testing)
    ("10_ID_Card_Vikram_Verma_Degraded_Scan.png", _render_id_card, {
        "name": "Vikram Verma", "guardian": "Ramesh Verma", "dob": "10-Nov-1995",
        "address": "77 Connaught Place, New Delhi 110001", "id_number": "VOT-DL-849201", "issue_date": "2019-06-20"
    }),
]

print("Generating 10 custom test documents in custom_test_documents/...")
for idx, (fname, renderer, data) in enumerate(docs_spec, 1):
    img = renderer(data)
    # Apply synthetic degradation on the 10th document
    if "Degraded" in fname:
        img = img.rotate(3.5, expand=True, fillcolor=(230, 230, 230))
        img = img.filter(ImageFilter.GaussianBlur(radius=1.4))
    save_path = CUSTOM_DIR / fname
    img.save(save_path)
    print(f"[{idx}/10] Saved: {fname}")

print("\nAll 10 test documents successfully created in custom_test_documents/!")
