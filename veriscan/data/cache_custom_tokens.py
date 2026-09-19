"""Generate OCR token cache for custom_test_documents."""

import json
from pathlib import Path
from veriscan.data.cache_demo_tokens import build_tokens_from_truth

CUSTOM_DIR = Path(__file__).parent.parent.parent / "custom_test_documents"

# Document data definitions matching generate_custom_docs.py
DOCS_DATA = {
    "01_Aadhaar_Aarav_Sharma": {
        "type": "id_card",
        "fields": {
            "name": "Aarav Sharma", "guardian": "Suresh Sharma", "dob": "15/08/1999",
            "address": "42 MG Road, Bengaluru 560001", "id_number": "AADH-4920-1849"
        }
    },
    "02_Marksheet_Aarav_Sharma_Consistent": {
        "type": "marksheet",
        "fields": {
            "name": "Aarav Sharma", "guardian": "Suresh Sharma", "dob": "15-Aug-1999",
            "id_number": "ROLL-849201"
        }
    },
    "03_Utility_Bill_Aarav_Sharma_Benign_Variant": {
        "type": "address_proof",
        "fields": {
            "name": "A. Sharma", "address": "42 MG Rd, Blr 560001",
            "id_number": "CON-9402184"
        }
    },
    "04_Marksheet_Aarav_Sharma_DOB_Mismatch": {
        "type": "marksheet",
        "fields": {
            "name": "Aarav Sharma", "guardian": "Suresh Sharma", "dob": "18-Aug-1999",
            "id_number": "ROLL-849201"
        }
    },
    "05_ID_Card_Priya_Patel": {
        "type": "id_card",
        "fields": {
            "name": "Priya Patel", "guardian": "Kishore Patel", "dob": "22/01/2001",
            "address": "15 Marine Drive, Mumbai 400001", "id_number": "DL-MH-928104"
        }
    },
    "06_Application_Form_Priya_Patel_Consistent": {
        "type": "application_form",
        "fields": {
            "name": "Priya Patel", "guardian": "Kishore Patel", "dob": "2001-01-22",
            "address": "15 Marine Drive, Mumbai 400001", "id_number": "APP-2024-8192"
        }
    },
    "07_Utility_Bill_Priya_Patel_Address_Mismatch": {
        "type": "address_proof",
        "fields": {
            "name": "Priya Patil", "address": "15 Marine Drive, Mumbai 400050",
            "id_number": "CON-8472910"
        }
    },
    "08_Bank_Statement_Vikram_Verma": {
        "type": "bank_statement",
        "fields": {
            "name": "Vikram Verma", "address": "77 Connaught Place, New Delhi 110001",
            "id_number": "ACC-938210492"
        }
    },
    "09_ID_Card_Vikram_Verma_Consistent": {
        "type": "id_card",
        "fields": {
            "name": "Vikram Verma", "guardian": "Ramesh Verma", "dob": "10-Nov-1995",
            "address": "77 Connaught Place, New Delhi 110001", "id_number": "VOT-DL-849201"
        }
    },
    "10_ID_Card_Vikram_Verma_Degraded_Scan": {
        "type": "id_card",
        "is_degraded": True,
        "fields": {
            "name": "Vikram Verma", "guardian": "Ramesh Verma", "dob": "10-Nov-1995",
            "address": "77 Connaught Place, New Delhi 110001", "id_number": "VOT-DL-849201"
        }
    }
}

def generate_custom_tokens():
    CUSTOM_DIR.mkdir(exist_ok=True)
    for stem, data in DOCS_DATA.items():
        cache_file = CUSTOM_DIR / f"{stem}_tokens.json"
        is_deg = data.get("is_degraded", False)
        base_conf = 0.55 if is_deg else 0.96

        tokens = []
        line_counter = 0

        # Type header
        tokens.append({
            "text": data["type"].upper().replace("_", " "),
            "conf": base_conf,
            "bbox": [50, 30, 220, 26],
            "line_id": line_counter
        })
        line_counter += 1

        y_pos = 90
        for fn, val in data["fields"].items():
            lbl_text = fn.replace("_", " ").title() + ":"
            tokens.append({
                "text": lbl_text,
                "conf": base_conf,
                "bbox": [50, y_pos, 110, 20],
                "line_id": line_counter
            })

            val_str = str(val)
            val_words = val_str.split()
            x_val = 180
            for w in val_words:
                w_len = max(22, len(w) * 10)
                tokens.append({
                    "text": w,
                    "conf": base_conf,
                    "bbox": [x_val, y_pos, w_len, 20],
                    "line_id": line_counter
                })
                x_val += w_len + 6

            line_counter += 1
            y_pos += 35

        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(tokens, f, indent=2)
        print(f"Generated tokens for: {stem}")

if __name__ == "__main__":
    generate_custom_tokens()
