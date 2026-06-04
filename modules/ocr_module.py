"""
ocr_module.py
Extracts full GST invoice information from scanned images.
FIXED:
  - Auto-detects Tesseract path on Windows/Mac/Linux
  - EasyOCR fallback if Tesseract fails
  - Much stronger regex for all invoice fields
  - Better image preprocessing for receipt-style invoices
"""

import re
import cv2
import numpy as np
import os


# ─────────────────────────────────────────────
#  Auto-configure Tesseract path (Windows fix)
# ─────────────────────────────────────────────
def configure_tesseract():
    """Automatically finds and sets Tesseract path."""
    try:
        import pytesseract

        # Common Windows installation paths
        windows_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            r"C:\Users\Admin\AppData\Local\Programs\Tesseract-OCR\tesseract.exe",
            r"C:\tesseract\tesseract.exe",
        ]

        for path in windows_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                return True

        # Try running tesseract from PATH (Linux/Mac)
        import shutil
        if shutil.which("tesseract"):
            return True

        return False
    except Exception:
        return False


# ─────────────────────────────────────────────
#  Image Preprocessing
# ─────────────────────────────────────────────
def preprocess_image(image_path: str):
    """
    Enhanced preprocessing for receipt/invoice images.
    Returns multiple versions for best OCR results.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    # Upscale small images for better OCR
    h, w = img.shape[:2]
    if w < 800:
        scale = 800 / w
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_CUBIC)

    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Denoise
    denoised = cv2.fastNlMeansDenoising(gray, h=10)

    # CLAHE contrast enhancement (great for receipts)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    # Otsu thresholding
    _, otsu = cv2.threshold(
        enhanced, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return otsu, enhanced, gray


# ─────────────────────────────────────────────
#  OCR Text Extraction
# ─────────────────────────────────────────────
def extract_text_tesseract(image_path: str) -> str:
    """Extract text using Tesseract with best config."""
    try:
        import pytesseract
        configure_tesseract()

        otsu, enhanced, gray = preprocess_image(image_path)

        # Try multiple PSM modes, take the longest result
        results = []
        for psm in [6, 4, 3]:
            config = f"--oem 3 --psm {psm}"
            text = pytesseract.image_to_string(otsu, config=config)
            results.append(text)
            text2 = pytesseract.image_to_string(enhanced, config=config)
            results.append(text2)

        # Return longest extracted text
        best = max(results, key=lambda t: len(t.strip()))
        return best

    except Exception as e:
        return f"TESSERACT_ERROR: {e}"


def extract_text_easyocr(image_path: str) -> str:
    """Fallback: Extract text using EasyOCR."""
    try:
        import easyocr
        reader = easyocr.Reader(["en"], gpu=False)
        results = reader.readtext(image_path, detail=0, paragraph=True)
        return "\n".join(results)
    except ImportError:
        return "EASYOCR_NOT_INSTALLED"
    except Exception as e:
        return f"EASYOCR_ERROR: {e}"


# ─────────────────────────────────────────────
#  GST Info Extractor
# ─────────────────────────────────────────────
class OCRGSTVerifier:

    def extract_text(self, image_path: str) -> str:
        """
        Tries Tesseract first, falls back to EasyOCR.
        Returns best extracted text.
        """
        # Try Tesseract
        text = extract_text_tesseract(image_path)

        if "TESSERACT_ERROR" in text or len(text.strip()) < 20:
            # Fallback to EasyOCR
            text = extract_text_easyocr(image_path)

        if len(text.strip()) < 20:
            return "[OCR FAILED] Could not extract text. Check Tesseract installation."

        return text

    def extract_gst_info(self, text: str) -> dict:
        """
        Extracts all GST invoice fields using strong regex patterns.
        Works for printed receipts, thermal invoices, and digital invoices.
        """
        info = {}
        t = text  # original case for amounts
        tu = text.upper()  # uppercase for keyword matching

        # ── GSTIN ─────────────────────────────────────────────────────────
        # Standard GSTIN: 15 chars — 2 digits + 5 alpha + 4 digits + 1+1+Z+1
        gstin_pattern = r'\b\d{2}[A-Z]{5}\d{4}[A-Z][A-Z0-9]Z[A-Z0-9]\b'
        gstins = re.findall(gstin_pattern, tu)

        # Also catch partial/OCR-mangled GSTINs (at least 10 chars alphanumeric)
        if not gstins:
            loose_gstin = re.findall(r'\b[0-9]{2}[A-Z0-9]{10,13}\b', tu)
            gstins = loose_gstin

        info["GSTIN Numbers Found"] = list(set(gstins)) if gstins else ["Not Found"]

        # ── Invoice Number ─────────────────────────────────────────────────
        inv_patterns = [
            r'(?:invoice\s*(?:no|number|#|num)[.:\s]*)([\w\-/]+)',
            r'(?:inv\s*(?:no|#)[.:\s]*)([\w\-/]+)',
            r'(?:bill\s*(?:no|number)[.:\s]*)([\w\-/]+)',
            r'(?:receipt\s*(?:no|#)[.:\s]*)([\w\-/]+)',
        ]
        inv_found = None
        for pat in inv_patterns:
            m = re.search(pat, t, re.IGNORECASE)
            if m:
                inv_found = m.group(1).strip()
                break
        info["Invoice Number"] = inv_found if inv_found else "Not Found"

        # ── Invoice Date ───────────────────────────────────────────────────
        date_patterns = [
            r'\b(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})\b',
            r'\b(\d{4}[\/\-]\d{2}[\/\-]\d{2})\b',
            r'(\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{2,4})',
        ]
        date_found = None
        for pat in date_patterns:
            m = re.search(pat, t, re.IGNORECASE)
            if m:
                date_found = m.group(1).strip()
                break
        info["Invoice Date"] = date_found if date_found else "Not Found"

        # ── Amounts helper ─────────────────────────────────────────────────
        def clean_amount(raw: str) -> str:
            """Cleans OCR-mangled amount strings."""
            # Remove non-numeric except dot and comma
            cleaned = re.sub(r'[^\d,.]', '', raw)
            cleaned = cleaned.replace(",", "")
            if not cleaned:
                return None
            try:
                # Validate it's a reasonable amount (not a phone/pin)
                val = float(cleaned)
                if val > 100000000:  # over 10 crore — likely OCR error
                    return None
                return f"₹{val:,.2f}"
            except Exception:
                return None

        def find_amount(patterns):
            for pat in patterns:
                m = re.search(pat, t, re.IGNORECASE)
                if m:
                    result = clean_amount(m.group(1))
                    if result:
                        return result
            return "Not Found"

        # ── Taxable / Subtotal Amount ──────────────────────────────────────
        info["Taxable Amount (Subtotal)"] = find_amount([
            r'(?:taxable\s*(?:value|amount|amt))[:\s]*[₹Rs.\s]*([0-9,]+\.?\d*)',
            r'(?:subtotal|sub\s*total|sub-total)[:\s]*[₹Rs.\s]*([0-9,]+\.?\d*)',
            r'(?:basic\s*amount|base\s*amount)[:\s]*[₹Rs.\s]*([0-9,]+\.?\d*)',
            r'subtotal[^0-9]*([0-9,]+\.?\d*)',
        ])

        # ── Total Amount (extracted BEFORE CGST/SGST for fallback calc) ───
        info["Total Amount"] = find_amount([
            r'(?:grand\s*total|total\s*amount|net\s*amount)[:\s]*[₹Rs.\s]*([0-9,]+\.?\d*)',
            r'(?:amount\s*payable|payable\s*amount)[:\s]*[₹Rs.\s]*([0-9,]+\.?\d*)',
            r'^\s*[Tt]otal\s*[₹Rs.\s]*([0-9,]+\.?\d*)',
            r'total\s*[₹%@\s]*([0-9,]{5,}\.?\d*)',
        ])

        # ── Helper: extract GST rate percentage from text ──────────────────
        def find_gst_rate(keyword: str) -> float:
            """Finds the % rate associated with CGST/SGST/IGST keyword."""
            pattern = rf'{keyword}\s*@\s*([\d.]+)\s*%'
            m = re.search(pattern, t, re.IGNORECASE)
            if m:
                return float(m.group(1))
            return None

        # ── Helper: calculate tax from total when not printed ──────────────
        def calculate_tax_from_total(total_str: str, rate: float) -> str:
            """
            Back-calculates taxable amount and tax from total.
            Formula: taxable = total / (1 + rate/100)
                     tax     = total - taxable
            """
            try:
                total_val = float(re.sub(r'[^\d.]', '', total_str))
                taxable   = total_val / (1 + rate / 100)
                tax       = total_val - taxable
                return f"₹{tax:,.2f} (calculated @ {rate}%)"
            except Exception:
                return None

        # ── CGST ───────────────────────────────────────────────────────────
        cgst = find_amount([
            r'cgst\s*@\s*[\d.]+\s*%\s*[:\s]*[₹Rs.\s]*([0-9,]+\.?\d*)',
            r'cgst\s*@\s*[\d.]+\s*[:\s]*[₹Rs.\s]*([0-9,]+\.?\d*)',
            r'cgst[^0-9%\n]{0,15}([0-9,]{4,}\.?\d*)',
            r'central\s*(?:tax|gst)[^0-9]{0,15}([0-9,]+\.?\d*)',
            r'c\.g\.s\.t[^0-9]{0,10}([0-9,]+\.?\d*)',
            r'[cC][gG][sStT]\s*[@\s]*[\d.]+\s*%?[:\s₹Rs.]*([0-9,]+\.?\d*)',
        ])
        # Fallback: calculate CGST from total if rate % is mentioned
        if cgst == "Not Found":
            cgst_rate = find_gst_rate("cgst")
            total_raw = info.get("Total Amount", "Not Found")
            if cgst_rate and total_raw != "Not Found":
                calculated = calculate_tax_from_total(total_raw, cgst_rate)
                cgst = calculated if calculated else "Not Found"
        info["CGST"] = cgst

        # ── SGST ───────────────────────────────────────────────────────────
        sgst = find_amount([
            r'sgst\s*@\s*[\d.]+\s*%\s*[:\s]*[₹Rs.\s]*([0-9,]+\.?\d*)',
            r'sgst\s*@\s*[\d.]+\s*[:\s]*[₹Rs.\s]*([0-9,]+\.?\d*)',
            r'sgst[^0-9%\n]{0,15}([0-9,]{4,}\.?\d*)',
            r'state\s*(?:tax|gst)[^0-9]{0,15}([0-9,]+\.?\d*)',
            r's\.g\.s\.t[^0-9]{0,10}([0-9,]+\.?\d*)',
            r'[sS][gG][sStT]\s*[@\s]*[\d.]+\s*%?[:\s₹Rs.]*([0-9,]+\.?\d*)',
        ])
        # Fallback: calculate SGST from total if rate % is mentioned
        if sgst == "Not Found":
            sgst_rate = find_gst_rate("sgst")
            total_raw = info.get("Total Amount", "Not Found")
            if sgst_rate and total_raw != "Not Found":
                calculated = calculate_tax_from_total(total_raw, sgst_rate)
                sgst = calculated if calculated else "Not Found"
        info["SGST"] = sgst

        # ── IGST ───────────────────────────────────────────────────────────
        igst = find_amount([
            r'igst\s*@\s*[\d.]+\s*%\s*[:\s]*[₹Rs.\s]*([0-9,]+\.?\d*)',
            r'igst[^0-9%\n]{0,15}([0-9,]{4,}\.?\d*)',
            r'integrated\s*(?:tax|gst)[^0-9]{0,15}([0-9,]+\.?\d*)',
            r'i\.g\.s\.t[^0-9]{0,10}([0-9,]+\.?\d*)',
        ])
        # Fallback: calculate IGST from total if rate % is mentioned
        if igst == "Not Found":
            igst_rate = find_gst_rate("igst")
            total_raw = info.get("Total Amount", "Not Found")
            if igst_rate and total_raw != "Not Found":
                calculated = calculate_tax_from_total(total_raw, igst_rate)
                igst = calculated if calculated else "Not Found"
        info["IGST"] = igst

        # ── GST Summary (auto-calculate if all missing but total known) ────
        # When invoice only shows Total with no tax breakdown
        if (info["CGST"] == "Not Found" and
            info["SGST"] == "Not Found" and
            info["IGST"] == "Not Found"):

            total_raw = info.get("Total Amount", "Not Found")
            # Try to find any GST % mentioned anywhere in the invoice
            any_rate = re.search(r'gst\s*@?\s*([\d.]+)\s*%', t, re.IGNORECASE)
            if any_rate and total_raw != "Not Found":
                rate = float(any_rate.group(1))
                try:
                    total_val = float(re.sub(r'[^\d.]', '', total_raw))
                    taxable   = total_val / (1 + rate / 100)
                    tax_total = total_val - taxable
                    # Split equally between CGST and SGST (intra-state)
                    half_tax  = tax_total / 2
                    info["CGST"] = f"₹{half_tax:,.2f} (auto @ {rate/2}%)"
                    info["SGST"] = f"₹{half_tax:,.2f} (auto @ {rate/2}%)"
                    info["Taxable Amount (Subtotal)"] = f"₹{taxable:,.2f} (auto-calculated)"
                except Exception:
                    pass

        # Total Amount already extracted above before tax calculations

        # ── HSN Codes (4, 6, or 8 digit numbers) ──────────────────────────
        raw_nums = re.findall(r'\b(\d{4,8})\b', t)
        INDIA_PINCODES = set()  # We'll filter 6-digit numbers that look like pincodes
        hsn_codes = []
        for n in raw_nums:
            n_int = int(n)
            if 1900 <= n_int <= 2100:        # skip years
                continue
            if n.startswith("000"):           # skip 000xxx
                continue
            if len(n) == 6 and n_int > 100000 and n_int < 999999:
                # Could be pincode — only include if it appears near "HSN" keyword
                if "hsn" in t.lower():
                    hsn_codes.append(n)
                # else skip likely pincodes like 560001
                continue
            if len(n) == 10:                  # skip phone numbers
                continue
            hsn_codes.append(n)
        info["HSN Codes in Invoice"] = list(set(hsn_codes)) if hsn_codes else ["Not Found"]

        # ── Place of Supply ────────────────────────────────────────────────
        pos_match = re.search(
            r'(?:place\s*of\s*supply|state\s*of\s*supply)[:\s]*([A-Za-z\s]{3,25})',
            t, re.IGNORECASE
        )
        info["Place of Supply"] = pos_match.group(1).strip() if pos_match else "Not Found"

        # ── Seller / Company Name ──────────────────────────────────────────
        # Find first line that looks like a real company name (mostly letters, > 4 chars)
        lines = [ln.strip() for ln in t.split("\n") if ln.strip()]
        seller = "Not Found"
        for line in lines[:8]:  # Check first 8 lines
            # Skip lines that are mostly symbols/numbers/garbled
            alpha_ratio = sum(c.isalpha() or c.isspace() for c in line) / max(len(line), 1)
            if alpha_ratio > 0.6 and len(line) > 4:
                seller = line
                break
        info["Seller Name"] = seller

        return info

    def validate_gstin(self, gstin: str) -> dict:
        """Validates a GSTIN and decodes state + PAN."""
        STATE_CODES = {
            "01": "Jammu & Kashmir",   "02": "Himachal Pradesh",
            "03": "Punjab",            "04": "Chandigarh",
            "05": "Uttarakhand",       "06": "Haryana",
            "07": "Delhi",             "08": "Rajasthan",
            "09": "Uttar Pradesh",     "10": "Bihar",
            "11": "Sikkim",            "12": "Arunachal Pradesh",
            "13": "Nagaland",          "14": "Manipur",
            "15": "Mizoram",           "16": "Tripura",
            "17": "Meghalaya",         "18": "Assam",
            "19": "West Bengal",       "20": "Jharkhand",
            "21": "Odisha",            "22": "Chhattisgarh",
            "23": "Madhya Pradesh",    "24": "Gujarat",
            "27": "Maharashtra",       "29": "Karnataka",
            "32": "Kerala",            "33": "Tamil Nadu",
            "34": "Puducherry",        "36": "Telangana",
            "37": "Andhra Pradesh",
        }
        gstin = gstin.strip().upper()
        valid = bool(re.match(
            r'^\d{2}[A-Z]{5}\d{4}[A-Z][A-Z0-9]Z[A-Z0-9]$', gstin
        ))
        state_code = gstin[:2]
        state = STATE_CODES.get(state_code, "Unknown State")
        pan = gstin[2:12] if len(gstin) == 15 else "N/A"
        return {
            "GSTIN":  gstin,
            "Valid":  "✅ Valid" if valid else "❌ Invalid",
            "State":  state,
            "PAN":    pan,
        }
