"""Synthetic mock document generator for VeriScan.

Generates 5 distinct synthetic templates with Faker (en_IN):
1. id_card
2. marksheet
3. address_proof (utility bill)
4. bank_statement
5. application_form

Every generated document includes the mandatory watermark:
"SAMPLE – NOT A REAL DOCUMENT"
Generates dev cases, held-out cases, bundled sample cases, and ground-truth truth.json.
"""

from __future__ import annotations
import os
import json
import random
import math
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

try:
    from faker import Faker
    fake = Faker("en_IN")
except ImportError:
    # Minimal fallback generator if faker still installing
    class MiniFaker:
        def first_name(self): return random.choice(["Rohit", "Aarav", "Priya", "Ananya", "Suresh", "Vikram", "Sneha", "Kavita"])
        def last_name(self): return random.choice(["Sharma", "Verma", "Patel", "Reddy", "Kumar", "Singh", "Gupta", "Deshmukh"])
        def street_name(self): return random.choice(["MG Road", "Brigade Rd", "Nehru Street", "Station Road", "Park Avenue"])
        def city(self): return random.choice(["Bengaluru", "Mumbai", "Delhi", "Chennai", "Hyderabad", "Pune", "Kolkata"])
        def state(self): return random.choice(["Karnataka", "Maharashtra", "Tamil Nadu", "Telangana", "Delhi"])
    fake = MiniFaker()

WATERMARK_TEXT = "SAMPLE – NOT A REAL DOCUMENT"

# Paths
DATA_DIR = Path(__file__).parent
SAMPLES_DIR = DATA_DIR / "samples"
HELDOUT_DIR = DATA_DIR / "heldout"
DEV_DIR = DATA_DIR / "dev"


def _draw_watermark(image: Image.Image, text: str = WATERMARK_TEXT) -> None:
    """Draw prominent diagonal watermark across the image."""
    overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    w, h = image.size

    # Choose font size proportional to width
    font_size = max(24, w // 18)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    # Draw semi-transparent diagonal text in several positions
    for y_offset in [int(h * 0.3), int(h * 0.6)]:
        draw.text((int(w * 0.1), y_offset), text, fill=(220, 50, 50, 110), font=font)

    # Convert image to RGBA if needed
    if image.mode != "RGBA":
        base = image.convert("RGBA")
    else:
        base = image
    blended = Image.alpha_composite(base, overlay)
    image.paste(blended.convert(image.mode))


def _render_id_card(data: Dict[str, str], size: Tuple[int, int] = (650, 420), heldout: bool = False) -> Image.Image:
    """Template 1: ID Card (Identity Proof)."""
    img = Image.new("RGB", size, (245, 248, 250) if not heldout else (250, 245, 240))
    draw = ImageDraw.Draw(img)
    w, h = size

    # Outer card border
    draw.rectangle([10, 10, w - 10, h - 10], outline=(30, 60, 90), width=3)
    # Header bar
    draw.rectangle([12, 12, w - 12, 65], fill=(24, 43, 73) if not heldout else (40, 20, 60))
    draw.text((25, 24), "REPUBLIC CITIZEN IDENTITY CARD", fill=(255, 255, 255))
    draw.text((w - 180, 26), "GOVT OF SAMPLES", fill=(200, 220, 240))

    # Photo box placeholder
    draw.rectangle([30, 90, 150, 230], outline=(100, 100, 100), width=2, fill=(220, 225, 230))
    draw.text((45, 150), "[PHOTO]", fill=(120, 120, 120))

    # Fields
    x_lbl = 180
    y_start = 90
    spacing = 38

    labels_values = [
        ("Full Name:", data.get("name", "")),
        ("Father's Name:", data.get("guardian_name", "")),
        ("Date of Birth:", data.get("dob", "")),
        ("ID Number:", data.get("id_number", "")),
        ("Address:", data.get("address", "")),
        ("Issue Date:", data.get("issue_date", "2021-04-15"))
    ]

    for i, (lbl, val) in enumerate(labels_values):
        cy = y_start + i * spacing
        draw.text((x_lbl, cy), lbl, fill=(50, 50, 50))
        draw.text((x_lbl + 130, cy), val, fill=(10, 15, 25))

    _draw_watermark(img)
    return img


def _render_marksheet(data: Dict[str, str], size: Tuple[int, int] = (650, 750), heldout: bool = False) -> Image.Image:
    """Template 2: Educational Marksheet / Certificate."""
    img = Image.new("RGB", size, (255, 255, 255) if not heldout else (252, 250, 245))
    draw = ImageDraw.Draw(img)
    w, h = size

    # Double border
    draw.rectangle([15, 15, w - 15, h - 15], outline=(100, 120, 140), width=2)
    draw.rectangle([20, 20, w - 20, h - 20], outline=(180, 190, 200), width=1)

    # University Header
    draw.text((w // 2 - 140, 40), "NATIONAL BOARD OF HIGHER EDUCATION", fill=(20, 40, 80))
    draw.text((w // 2 - 90, 65), "STATEMENT OF MARKS (ANNUAL)", fill=(80, 80, 80))
    draw.line([40, 95, w - 40, 95], fill=(150, 150, 150), width=1)

    # Student metadata
    fields = [
        ("Candidate Name:", data.get("name", "")),
        ("Roll Number:", data.get("id_number", "")),
        ("Father Name:", data.get("guardian_name", "")),
        ("Date of Birth:", data.get("dob", "")),
        ("Institution Code:", "COLL-BLR-0941"),
        ("Exam Year:", "2022")
    ]

    for i, (lbl, val) in enumerate(fields):
        col = i % 2
        row = i // 2
        x = 45 if col == 0 else 340
        y = 115 + row * 32
        draw.text((x, y), lbl, fill=(70, 70, 70))
        draw.text((x + 130, y), val, fill=(10, 10, 10))

    # Academic Marks Table
    ty = 230
    draw.rectangle([40, ty, w - 40, ty + 220], outline=(100, 100, 100), width=1)
    draw.rectangle([40, ty, w - 40, ty + 30], fill=(235, 240, 245))
    draw.text((55, ty + 8), "Subject", fill=(20, 20, 20))
    draw.text((280, ty + 8), "Max Marks", fill=(20, 20, 20))
    draw.text((420, ty + 8), "Marks Obtained", fill=(20, 20, 20))
    draw.text((540, ty + 8), "Grade", fill=(20, 20, 20))

    subjects = [
        ("Advanced Mathematics", "100", "88", "A+"),
        ("Computer Science", "100", "94", "O"),
        ("Physics & Applied Mechanics", "100", "82", "A"),
        ("Technical English", "100", "86", "A+"),
        ("Engineering Systems", "100", "90", "O")
    ]

    for idx, (sub, max_m, obt_m, grd) in enumerate(subjects):
        sy = ty + 35 + idx * 36
        draw.line([40, sy - 4, w - 40, sy - 4], fill=(220, 220, 220), width=1)
        draw.text((55, sy), sub, fill=(40, 40, 40))
        draw.text((300, sy), max_m, fill=(40, 40, 40))
        draw.text((450, sy), obt_m, fill=(20, 20, 20))
        draw.text((550, sy), grd, fill=(10, 80, 20))

    # Footer
    draw.text((50, 660), f"Issue Date: {data.get('issue_date', '2022-06-20')}", fill=(60, 60, 60))
    draw.text((w - 200, 660), "Controller of Examinations", fill=(60, 60, 60))

    _draw_watermark(img)
    return img


def _render_utility_bill(data: Dict[str, str], size: Tuple[int, int] = (650, 600), heldout: bool = False) -> Image.Image:
    """Template 3: Utility / Electricity Bill (Address Proof)."""
    img = Image.new("RGB", size, (255, 255, 255) if not heldout else (248, 252, 250))
    draw = ImageDraw.Draw(img)
    w, h = size

    # Utility provider header
    draw.rectangle([15, 15, w - 15, 80], fill=(230, 240, 235))
    draw.text((30, 25), "STATE ELECTRICITY DISTRIBUTION CORPORATION", fill=(20, 80, 50))
    draw.text((30, 50), "ELECTRICITY SUPPLY BILL CUM NOTICE", fill=(70, 70, 70))
    draw.text((w - 180, 45), "CUSTOMER COPY", fill=(100, 100, 100))

    # Consumer Details Box
    draw.rectangle([25, 100, w - 25, 310], outline=(150, 180, 160), width=1)
    draw.rectangle([25, 100, w - 25, 130], fill=(240, 245, 242))
    draw.text((40, 108), "CONSUMER PARTICULARS & SERVICE ADDRESS", fill=(20, 60, 30))

    fields = [
        ("Consumer Name:", data.get("name", "")),
        ("Consumer No:", data.get("id_number", "CON-8472910")),
        ("Service Address:", data.get("address", "")),
        ("Bill Date:", data.get("issue_date", "2023-11-04")),
        ("Due Date:", "2023-11-20"),
        ("Tariff Category:", "LT-1 Residential Domestic")
    ]

    for idx, (lbl, val) in enumerate(fields):
        y = 145 + idx * 26
        draw.text((45, y), lbl, fill=(70, 70, 70))
        draw.text((200, y), val, fill=(10, 10, 10))

    # Billing Summary
    draw.rectangle([25, 330, w - 25, 480], outline=(180, 180, 180), width=1)
    draw.text((40, 345), "Billing Period: 01-Oct-2023 to 31-Oct-2023", fill=(50, 50, 50))
    draw.text((40, 375), "Units Consumed: 218 kWh", fill=(50, 50, 50))
    draw.text((40, 405), "Energy Charges: Rs. 1,420.00", fill=(50, 50, 50))
    draw.text((40, 435), "Total Amount Payable: Rs. 1,650.00", fill=(10, 10, 10))

    _draw_watermark(img)
    return img


def _render_bank_statement(data: Dict[str, str], size: Tuple[int, int] = (650, 650), heldout: bool = False) -> Image.Image:
    """Template 4: Bank Statement Header."""
    img = Image.new("RGB", size, (255, 255, 255) if not heldout else (255, 250, 245))
    draw = ImageDraw.Draw(img)
    w, h = size

    # Bank Banner
    draw.rectangle([15, 15, w - 15, 75], fill=(20, 50, 100))
    draw.text((30, 28), "NATIONAL COMMERCIAL BANK", fill=(255, 255, 255))
    draw.text((30, 50), "Retail Banking & Wealth Management", fill=(200, 220, 255))
    draw.text((w - 200, 35), "ACCOUNT STATEMENT", fill=(255, 255, 255))

    # Account info grid
    draw.rectangle([25, 95, w - 25, 260], outline=(200, 210, 225), width=1, fill=(248, 250, 254))

    fields = [
        ("Account Holder:", data.get("name", "")),
        ("Account Number:", data.get("id_number", "")),
        ("Address:", data.get("address", "")),
        ("Branch IFSC:", "NCBL0004921"),
        ("Statement Date:", data.get("issue_date", "2023-08-10")),
        ("Account Type:", "Regular Savings Account")
    ]

    for idx, (lbl, val) in enumerate(fields):
        y = 110 + idx * 24
        draw.text((45, y), lbl, fill=(80, 80, 90))
        draw.text((180, y), val, fill=(15, 20, 30))

    # Transaction sample rows
    ty = 280
    draw.rectangle([25, ty, w - 25, ty + 25], fill=(225, 235, 248))
    draw.text((40, ty + 5), "Date", fill=(20, 30, 50))
    draw.text((130, ty + 5), "Description", fill=(20, 30, 50))
    draw.text((400, ty + 5), "Debit", fill=(20, 30, 50))
    draw.text((480, ty + 5), "Credit", fill=(20, 30, 50))
    draw.text((560, ty + 5), "Balance", fill=(20, 30, 50))

    txs = [
        ("01-Aug-2023", "UPI / Rent Transfer", "15,000.00", "-", "42,850.00"),
        ("04-Aug-2023", "Salary Credit NEFT", "-", "65,000.00", "107,850.00"),
        ("07-Aug-2023", "Utility Bill AutoDebit", "1,650.00", "-", "106,200.00")
    ]
    for i, tx in enumerate(txs):
        row_y = ty + 35 + i * 28
        draw.text((35, row_y), tx[0], fill=(60, 60, 60))
        draw.text((130, row_y), tx[1], fill=(40, 40, 40))
        draw.text((400, row_y), tx[2], fill=(60, 60, 60))
        draw.text((480, row_y), tx[3], fill=(20, 100, 20))
        draw.text((560, row_y), tx[4], fill=(10, 10, 10))

    _draw_watermark(img)
    return img


def _render_application_form(data: Dict[str, str], size: Tuple[int, int] = (650, 700), heldout: bool = False) -> Image.Image:
    """Template 5: Scheme / University Application Form."""
    img = Image.new("RGB", size, (255, 255, 255) if not heldout else (250, 250, 248))
    draw = ImageDraw.Draw(img)
    w, h = size

    # Header
    draw.rectangle([20, 20, w - 20, 75], outline=(50, 50, 50), width=1)
    draw.text((w // 2 - 130, 30), "NATIONAL ADMISSION PORTAL", fill=(10, 10, 10))
    draw.text((w // 2 - 160, 52), "CANDIDATE ENROLLMENT & VERIFICATION FORM", fill=(80, 80, 80))

    # Particulars Box
    draw.rectangle([20, 90, w - 20, 500], outline=(120, 120, 120), width=1)
    draw.rectangle([20, 90, w - 20, 120], fill=(240, 240, 240))
    draw.text((35, 98), "SECTION A: PERSONAL PARTICULARS", fill=(30, 30, 30))

    fields = [
        ("Applicant Name:", data.get("name", "")),
        ("Father Name:", data.get("guardian_name", "")),
        ("Date of Birth:", data.get("dob", "")),
        ("Registration No:", data.get("id_number", "")),
        ("Residential Address:", data.get("address", "")),
        ("Submission Date:", data.get("issue_date", "2024-01-12"))
    ]

    for idx, (lbl, val) in enumerate(fields):
        y = 135 + idx * 36
        draw.text((40, y), lbl, fill=(60, 60, 60))
        # Draw input box around value
        draw.rectangle([200, y - 4, w - 40, y + 22], outline=(190, 190, 190), width=1)
        draw.text((210, y), val, fill=(10, 10, 10))

    # Declaration
    draw.text((40, 530), "Declaration: I hereby declare that the particulars furnished above are true to my knowledge.", fill=(70, 70, 70))
    draw.text((w - 220, 620), "Candidate Signature", fill=(80, 80, 80))

    _draw_watermark(img)
    return img


# Degradation utilities
def apply_degradations(img: Image.Image, level: int = 1) -> Image.Image:
    """Apply realistic scan degradations: blur, rotation, noise, compression."""
    res = img.copy()

    # 1. Blur
    if level >= 1:
        res = res.filter(ImageFilter.GaussianBlur(radius=0.7 * level))

    # 2. Rotation
    angle = random.choice([-1.0, 1.0]) * (1.5 * level)
    res = res.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=(255, 255, 255))

    # 3. Add Gaussian noise
    arr = np.array(res, dtype=np.float32)
    noise = np.random.normal(0, 4.0 * level, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    res = Image.fromarray(arr)

    return res


def generate_single_case(
    case_id: str,
    case_type: str,  # "consistent", "benign", "error_single", "error_multi", "degraded"
    output_dir: Path,
    heldout: bool = False
) -> Dict[str, Any]:
    """Generate a multi-document case with ground truth."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Base profile
    first = fake.first_name()
    last = fake.last_name()
    father_first = fake.first_name()
    base_name = f"{first} {last}"
    base_father = f"{father_first} {last}"
    base_dob = f"{random.randint(1995, 2004)}-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}"
    pin = f"{random.randint(560001, 560099)}"
    city = fake.city()
    base_addr = f"No {random.randint(12, 180)}, {fake.street_name()}, {city} - {pin}"
    base_id = f"IND-{random.randint(10000000, 99999999)}"

    # Document-specific variations
    doc_data: Dict[str, Dict[str, str]] = {
        "doc_1": {  # ID Card
            "name": base_name,
            "guardian_name": base_father,
            "dob": base_dob,
            "address": base_addr,
            "id_number": base_id,
            "issue_date": "2021-04-15"
        },
        "doc_2": {  # Marksheet
            "name": base_name,
            "guardian_name": base_father,
            "dob": base_dob,
            "id_number": f"ROLL-{random.randint(100000, 999999)}",
            "issue_date": "2022-06-20"
        },
        "doc_3": {  # Utility Bill
            "name": base_name,
            "address": base_addr,
            "id_number": f"CON-{random.randint(1000000, 9999999)}",
            "issue_date": "2023-11-04"
        }
    }

    injected_changes = []

    if case_type == "benign":
        # Inject benign name initial and date format variation
        initial_name = f"{first[0]}. {last}"
        doc_data["doc_2"]["name"] = initial_name
        injected_changes.append({
            "field": "name", "docs": ["doc_1", "doc_2"],
            "type": "benign", "desc": "Initial expansion (R. Sharma vs Rohit Sharma)"
        })

        # Address abbreviation (e.g. Road -> Rd, Bengaluru -> Blr)
        abbrev_addr = base_addr.replace("Road", "Rd").replace("Street", "St").replace("Bengaluru", "Blr")
        doc_data["doc_3"]["address"] = abbrev_addr
        injected_changes.append({
            "field": "address", "docs": ["doc_1", "doc_3"],
            "type": "benign", "desc": "Address abbreviations (Rd, Blr)"
        })

    elif case_type in ["error_single", "error_multi"]:
        # Inject real discrepancy: DOB typo
        dob_parts = base_dob.split("-")
        # Change day
        alt_day = f"{(int(dob_parts[2]) % 25) + 3:02d}"
        alt_dob = f"{dob_parts[0]}-{dob_parts[1]}-{alt_day}"
        doc_data["doc_2"]["dob"] = alt_dob
        injected_changes.append({
            "field": "dob", "docs": ["doc_1", "doc_2"],
            "type": "error", "desc": f"DOB mismatch: {base_dob} vs {alt_dob}"
        })

        if case_type == "error_multi":
            # Also inject different pincode / address
            wrong_pin = f"{int(pin) + 500}"
            wrong_addr = base_addr.replace(pin, wrong_pin)
            doc_data["doc_3"]["address"] = wrong_addr
            injected_changes.append({
                "field": "address", "docs": ["doc_1", "doc_3"],
                "type": "error", "desc": f"Address mismatch: Pincode {pin} vs {wrong_pin}"
            })

    # Render images
    img1 = _render_id_card(doc_data["doc_1"], heldout=heldout)
    img2 = _render_marksheet(doc_data["doc_2"], heldout=heldout)
    img3 = _render_utility_bill(doc_data["doc_3"], heldout=heldout)

    if case_type == "degraded":
        img1 = apply_degradations(img1, level=2)
        img2 = apply_degradations(img2, level=2)
        img3 = apply_degradations(img3, level=2)

    # Save images
    p1 = output_dir / "doc_1_id_card.png"
    p2 = output_dir / "doc_2_marksheet.png"
    p3 = output_dir / "doc_3_utility_bill.png"
    img1.save(p1)
    img2.save(p2)
    img3.save(p3)

    truth = {
        "case_id": case_id,
        "case_type": case_type,
        "is_heldout": heldout,
        "watermark": WATERMARK_TEXT,
        "documents": {
            "doc_1": {"filename": "doc_1_id_card.png", "type": "id_card", "fields_truth": doc_data["doc_1"]},
            "doc_2": {"filename": "doc_2_marksheet.png", "type": "marksheet", "fields_truth": doc_data["doc_2"]},
            "doc_3": {"filename": "doc_3_utility_bill.png", "type": "address_proof", "fields_truth": doc_data["doc_3"]}
        },
        "injected_changes": injected_changes
    }

    with open(output_dir / "truth.json", "w", encoding="utf-8") as f:
        json.dump(truth, f, indent=2)

    return truth


def generate_all_datasets() -> None:
    """Generate dev dataset, held-out dataset, and 4 bundled UI demo cases."""
    print("Generating VeriScan Synthetic Datasets...")
    random.seed(42)

    # 1. Generate 4 Bundled UI Demo Cases
    bundled_specs = [
        ("case_01_consistent", "consistent"),
        ("case_02_benign", "benign"),
        ("case_03_mismatch", "error_multi"),
        ("case_04_degraded", "degraded")
    ]
    for cid, ctype in bundled_specs:
        cdir = SAMPLES_DIR / cid
        generate_single_case(cid, ctype, cdir, heldout=False)
        print(f"Generated bundled sample case: {cid} ({ctype})")

    # 2. Generate 40 Dev Set Cases
    DEV_DIR.mkdir(parents=True, exist_ok=True)
    # Mix: 20% consistent (8), 25% benign (10), 30% single error (12), 10% multi error (4), 15% degraded (6)
    types_mix = (
        ["consistent"] * 8 +
        ["benign"] * 10 +
        ["error_single"] * 12 +
        ["error_multi"] * 4 +
        ["degraded"] * 6
    )
    for idx, ctype in enumerate(types_mix):
        cid = f"dev_case_{idx + 1:02d}_{ctype}"
        cdir = DEV_DIR / cid
        generate_single_case(cid, ctype, cdir, heldout=False)

    print(f"Generated {len(types_mix)} dev set benchmark cases in {DEV_DIR}")

    # 3. Generate 12 Held-out Set Cases (Phone-photo look / alternate styling)
    HELDOUT_DIR.mkdir(parents=True, exist_ok=True)
    heldout_mix = ["consistent"] * 3 + ["benign"] * 3 + ["error_single"] * 4 + ["error_multi"] * 2
    for idx, ctype in enumerate(heldout_mix):
        cid = f"heldout_case_{idx + 1:02d}_{ctype}"
        cdir = HELDOUT_DIR / cid
        generate_single_case(cid, ctype, cdir, heldout=True)

    print(f"Generated {len(heldout_mix)} held-out benchmark cases in {HELDOUT_DIR}")


if __name__ == "__main__":
    generate_all_datasets()
