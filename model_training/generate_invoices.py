"""
generate_invoices.py  —  GST Chatbot Training
════════════════════════════════════════════════
Generates synthetic GST invoices as PNG images
with ground truth JSON annotations.

Use this when you don't have enough real invoices.
Generates 500+ realistic Indian GST invoices.

Run:
    python model_training/generate_invoices.py
"""

import os
import json
import random
import string
from datetime import datetime, timedelta

OUTPUT_DIR       = "model_training/data/synthetic"
ANNOTATIONS_FILE = "model_training/data/annotations/synthetic_annotations.json"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(ANNOTATIONS_FILE), exist_ok=True)

# ── Sample Data Pools ─────────────────────────────────────────────────────

COMPANY_NAMES = [
    "Rajan Traders Pvt Ltd", "Sri Murugan Enterprises",
    "Chennai Tech Solutions", "Kumar & Co",
    "Lakshmi Industries", "Balaji Distributors",
    "TN Export House", "Coimbatore Textiles Ltd",
    "Madurai Steel Works", "Salem Pipes & Fittings",
    "ABC Electronics", "XYZ Pharma Pvt Ltd",
    "National Hardware Store", "Global Imports Ltd",
    "Sunrise Manufacturing Co",
]

PRODUCTS = [
    ("Laptop Computer",      "84713020", 18),
    ("Mobile Phone",         "85171210", 18),
    ("Cotton Fabric",        "52081100",  5),
    ("Steel Pipes",          "73061100", 18),
    ("Pharmaceutical Drugs", "30049099",  5),
    ("Electric Motor",       "85011010", 18),
    ("Plastic Bottles",      "39232100", 18),
    ("Wooden Furniture",     "94036000", 18),
    ("Wheat Flour",          "11010000",  0),
    ("Edible Oil",           "15079010",  5),
    ("Cement",               "25232100", 28),
    ("Paints & Varnish",     "32099010", 18),
    ("Copper Wire",          "74081110", 18),
    ("Solar Panel",          "85414011",  5),
    ("Auto Parts",           "87089900", 28),
]

STATE_CODES = {
    "01": "Jammu & Kashmir", "02": "Himachal Pradesh",
    "03": "Punjab",          "04": "Chandigarh",
    "05": "Uttarakhand",     "06": "Haryana",
    "07": "Delhi",           "08": "Rajasthan",
    "09": "Uttar Pradesh",   "10": "Bihar",
    "11": "Sikkim",          "12": "Arunachal Pradesh",
    "18": "Assam",           "19": "West Bengal",
    "20": "Jharkhand",       "21": "Odisha",
    "22": "Chhattisgarh",    "23": "Madhya Pradesh",
    "24": "Gujarat",         "27": "Maharashtra",
    "29": "Karnataka",       "32": "Kerala",
    "33": "Tamil Nadu",      "36": "Telangana",
    "37": "Andhra Pradesh",
}


def random_gstin(state_code="33"):
    """Generates a realistic GSTIN."""
    pan_chars  = ''.join(random.choices(string.ascii_uppercase, k=5))
    pan_digits = ''.join(random.choices(string.digits, k=4))
    pan_alpha  = random.choice(string.ascii_uppercase)
    entity     = random.choice(string.digits + string.ascii_uppercase)
    checksum   = random.choice(string.ascii_uppercase + string.digits)
    return f"{state_code}{pan_chars}{pan_digits}{pan_alpha}{entity}Z{checksum}"


def random_invoice_number():
    prefix = random.choice(["INV", "GST", "TAX", "BILL", "REC"])
    year   = random.choice(["2024", "2025", "2026"])
    num    = random.randint(100, 9999)
    return f"{prefix}/{year}/{num:04d}"


def random_date():
    start = datetime(2024, 1, 1)
    delta = timedelta(days=random.randint(0, 730))
    return (start + delta).strftime("%d/%m/%Y")


def generate_invoice_data() -> dict:
    """Generates one invoice's data dictionary."""
    product_name, hsn_code, gst_rate = random.choice(PRODUCTS)
    state_code   = random.choice(list(STATE_CODES.keys()))
    seller_gstin = random_gstin(state_code)
    buyer_gstin  = random_gstin(random.choice(list(STATE_CODES.keys())))
    is_interstate = (seller_gstin[:2] != buyer_gstin[:2])

    qty        = random.randint(1, 100)
    unit_price = round(random.uniform(100, 50000), 2)
    taxable    = round(qty * unit_price, 2)

    if is_interstate:
        igst_amt = round(taxable * gst_rate / 100, 2)
        cgst_amt = 0.0
        sgst_amt = 0.0
    else:
        igst_amt = 0.0
        cgst_amt = round(taxable * gst_rate / 200, 2)
        sgst_amt = round(taxable * gst_rate / 200, 2)

    total = round(taxable + igst_amt + cgst_amt + sgst_amt, 2)

    return {
        "seller_name":    random.choice(COMPANY_NAMES),
        "seller_gstin":   seller_gstin,
        "buyer_name":     random.choice(COMPANY_NAMES),
        "buyer_gstin":    buyer_gstin,
        "invoice_number": random_invoice_number(),
        "invoice_date":   random_date(),
        "product":        product_name,
        "hsn_code":       hsn_code,
        "quantity":       qty,
        "unit_price":     unit_price,
        "taxable_amount": taxable,
        "gst_rate":       gst_rate,
        "cgst_rate":      gst_rate / 2 if not is_interstate else 0,
        "sgst_rate":      gst_rate / 2 if not is_interstate else 0,
        "igst_rate":      gst_rate if is_interstate else 0,
        "cgst_amount":    cgst_amt,
        "sgst_amount":    sgst_amt,
        "igst_amount":    igst_amt,
        "total_amount":   total,
        "is_interstate":  is_interstate,
        "place_of_supply":STATE_CODES.get(buyer_gstin[:2], "Tamil Nadu"),
        "state_name":     STATE_CODES.get(state_code, "Tamil Nadu"),
    }


def render_invoice_image(data: dict, save_path: str):
    """
    Renders invoice data as a PNG image using Pillow.
    Returns bounding box annotations for each field.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("Install Pillow: pip install Pillow")
        return None

    W, H = 794, 1123   # A4 at 96dpi
    img  = Image.new("RGB", (W, H), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Try to load a font; fallback to default
    try:
        font_bold  = ImageFont.truetype("arial.ttf", 14)
        font_reg   = ImageFont.truetype("arial.ttf", 12)
        font_small = ImageFont.truetype("arial.ttf", 10)
        font_title = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        font_bold  = ImageFont.load_default()
        font_reg   = font_bold
        font_small = font_bold
        font_title = font_bold

    annotations = {}   # field_name → [x1,y1,x2,y2]

    def draw_field(label, value, x, y, font=font_reg, color=(0,0,0)):
        """Draws label + value, returns bounding box of value."""
        draw.text((x, y), f"{label}: ", font=font_bold, fill=(80,80,80))
        val_x = x + 160
        draw.text((val_x, y), str(value), font=font, fill=color)
        bbox = draw.textbbox((val_x, y), str(value), font=font)
        return list(bbox)   # [x1, y1, x2, y2]

    # ── Header ────────────────────────────────────────────────
    draw.rectangle([0, 0, W, 60], fill=(41, 128, 185))
    draw.text((W//2 - 80, 18), "TAX INVOICE", font=font_title, fill=(255,255,255))

    # ── Seller Info ───────────────────────────────────────────
    y = 80
    draw.text((40, y), data["seller_name"], font=font_bold, fill=(0,0,0))
    y += 20
    annotations["seller_gstin"] = draw_field("GSTIN", data["seller_gstin"], 40, y, color=(200,0,0))
    y += 20
    draw.text((40, y), f"State: {data['state_name']}", font=font_reg, fill=(0,0,0))

    # ── Invoice Details ───────────────────────────────────────
    y += 30
    draw.line([(40, y), (W-40, y)], fill=(200,200,200), width=1)
    y += 10
    annotations["invoice_number"] = draw_field("Invoice No", data["invoice_number"], 40, y)
    annotations["invoice_date"]   = draw_field("Date",       data["invoice_date"],   400, y)
    y += 25

    # ── Buyer Info ────────────────────────────────────────────
    draw.text((40, y), "Bill To:", font=font_bold, fill=(0,0,0))
    y += 18
    draw.text((40, y), data["buyer_name"], font=font_reg, fill=(0,0,0))
    y += 18
    annotations["buyer_gstin"] = draw_field("GSTIN", data["buyer_gstin"], 40, y)
    y += 20
    annotations["place_of_supply"] = draw_field("Place of Supply", data["place_of_supply"], 40, y)

    # ── Items Table ───────────────────────────────────────────
    y += 40
    draw.rectangle([40, y, W-40, y+25], fill=(230,230,250))
    headers = ["Description", "HSN", "Qty", "Rate", "Taxable Amt"]
    xs      = [45, 250, 350, 430, 560]
    for hdr, x in zip(headers, xs):
        draw.text((x, y+5), hdr, font=font_bold, fill=(0,0,80))
    y += 25
    draw.line([(40, y), (W-40, y)], fill=(200,200,200), width=1)
    y += 10

    draw.text((xs[0], y), data["product"],          font=font_reg, fill=(0,0,0))
    annotations["hsn_code"] = list(draw.textbbox((xs[1], y), data["hsn_code"], font=font_reg))
    draw.text((xs[1], y), data["hsn_code"],         font=font_reg, fill=(0,0,0))
    draw.text((xs[2], y), str(data["quantity"]),    font=font_reg, fill=(0,0,0))
    draw.text((xs[3], y), f"{data['unit_price']:.2f}", font=font_reg, fill=(0,0,0))
    draw.text((xs[4], y), f"{data['taxable_amount']:.2f}", font=font_reg, fill=(0,0,0))

    # ── Tax Summary ───────────────────────────────────────────
    y += 50
    draw.line([(40, y), (W-40, y)], fill=(200,200,200), width=1)
    y += 15
    annotations["taxable_amount"] = draw_field(
        "Taxable Amount", f"{data['taxable_amount']:,.2f}", 400, y)
    y += 22

    if data["is_interstate"]:
        annotations["igst_amount"] = draw_field(
            f"IGST @ {data['igst_rate']}%", f"{data['igst_amount']:,.2f}", 400, y)
        y += 22
    else:
        annotations["cgst_amount"] = draw_field(
            f"CGST @ {data['cgst_rate']}%", f"{data['cgst_amount']:,.2f}", 400, y)
        y += 22
        annotations["sgst_amount"] = draw_field(
            f"SGST @ {data['sgst_rate']}%", f"{data['sgst_amount']:,.2f}", 400, y)
        y += 22

    y += 5
    draw.line([(400, y), (W-40, y)], fill=(0,0,0), width=2)
    y += 5
    annotations["total_amount"] = draw_field(
        "Total Amount", f"{data['total_amount']:,.2f}", 400, y,
        font=font_bold, color=(0,100,0))

    # ── Save ─────────────────────────────────────────────────
    img.save(save_path, "PNG", dpi=(150, 150))
    return annotations


def generate_dataset(n: int = 500):
    """Generates n synthetic invoices with annotations."""
    all_annotations = []

    print(f"Generating {n} synthetic invoices...")
    for i in range(n):
        data      = generate_invoice_data()
        img_name  = f"invoice_{i:04d}.png"
        img_path  = os.path.join(OUTPUT_DIR, img_name)

        annotations = render_invoice_image(data, img_path)

        all_annotations.append({
            "image":       img_name,
            "ground_truth": data,
            "bbox":         annotations or {},
        })

        if (i + 1) % 50 == 0:
            print(f"  ✅ {i+1}/{n} invoices generated")

    with open(ANNOTATIONS_FILE, "w") as f:
        json.dump(all_annotations, f, indent=2)

    print(f"\n✅ Dataset complete!")
    print(f"   Images     : {OUTPUT_DIR}/")
    print(f"   Annotations: {ANNOTATIONS_FILE}")
    print(f"   Total      : {n} invoices")


if __name__ == "__main__":
    generate_dataset(500)
