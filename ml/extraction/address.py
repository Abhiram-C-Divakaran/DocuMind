import re
from typing import List
from ml.extraction.schemas import ExtractedField
from ml.ocr.engine import OCRResult

ADDRESS_KEYWORDS = [
    r"\bSTREET\b",
    r"\bST\.?\b",
    r"\bROAD\b",
    r"\bRD\.?\b",
    r"\bAVENUE\b",
    r"\bAVE\.?\b",
    r"\bBLVD\.?\b",
    r"\bBOULEVARD\b",
    r"\bLANE\b",
    r"\bLN\.?\b",
    r"\bSUITE\b",
    r"\bSTE\.?\b",
    r"\bFLOOR\b",
    r"\bFL\.?\b",
    r"\bBUILDING\b",
    r"\bBLDG\.?\b",
    r"\bBLOCK\b",
    r"\bSECTOR\b",
    r"\bZIP\b",
    r"\bPIN\s*CODE\b",
    r"\bPOSTAL\b",
    r"\bCITY\b",
    r"\bSTATE\b",
    r"\bUSA\b",
    r"\bINDIA\b",
    r"\bUK\b",
    r"\bGERMANY\b",
]


class AddressExtractor:
    def extract(self, ocr_result: OCRResult, vendor_name: Optional[str] = None) -> ExtractedField:
        """
        Extracts vendor physical address by searching header region lines
        following the vendor name for spatial indicators and address keywords.
        """
        lines = [line.strip() for line in ocr_result.full_text.splitlines() if line.strip()]
        if not lines:
            return ExtractedField(value=None, confidence=0.0, evidence="Empty OCR text")

        address_lines: List[str] = []
        start_scanning = False if vendor_name else True

        # Scan the first 12 lines
        for idx, line in enumerate(lines[:12]):
            if vendor_name and vendor_name.lower() in line.lower():
                start_scanning = True
                continue

            if not start_scanning:
                continue

            line_upper = line.upper()

            # Skip lines with financial keywords or item rows
            if any(k in line_upper for k in ["TOTAL", "SUBTOTAL", "TAX", "INVOICE", "CASHIER", "RECEIPT", "DATE"]):
                continue

            # Check if line contains address keywords or postal code patterns (5-6 digits)
            has_kw = any(re.search(pat, line_upper) for pat in ADDRESS_KEYWORDS)
            has_zip = bool(re.search(r"\b\d{5,6}\b", line))

            if has_kw or has_zip:
                address_lines.append(line)

        if not address_lines:
            return ExtractedField(
                value=None,
                confidence=0.0,
                evidence="No recognizable address keywords or postal codes found",
            )

        full_address = ", ".join(address_lines[:3])
        conf = 0.80 if len(address_lines) >= 2 else 0.65

        return ExtractedField(
            value=full_address,
            confidence=conf,
            raw_text=full_address,
            evidence=f"Matched {len(address_lines)} address lines: '{full_address}'",
        )
