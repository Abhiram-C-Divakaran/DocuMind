import re
from typing import List, Optional
from ml.extraction.schemas import ExtractedField
from ml.ocr.engine import OCRResult, OCRToken

# Common non-vendor headers and keywords to exclude
EXCLUDE_VENDOR_PATTERNS = [
    r"^\s*RECEIPT\s*$",
    r"^\s*TAX\s+INVOICE\s*$",
    r"^\s*INVOICE\s*$",
    r"^\s*CASH\s+BILL\s*$",
    r"^\s*CASH\s+MEMO\s*$",
    r"^\s*BILL\s*$",
    r"^\s*ORDER\s*#",
    r"^\s*WELCOME\s*$",
    r"^\s*CUSTOMER\s+COPY\s*$",
    r"^\s*MERCHANT\s+COPY\s*$",
    r"^\s*DUPLICATE\s*$",
    r"^\s*TEL\s*:",
    r"^\s*PHONE\s*:",
    r"^\s*FAX\s*:",
    r"^\s*GST\s*(NO|IN|ID)?\s*:",
    r"^\s*TIN\s*:",
    r"^\s*VAT\s*:",
    r"^\s*WWW\.",
    r"^\s*HTTP",
    r"^\s*DATE\s*:",
    r"^\s*TIME\s*:",
]


class VendorExtractor:
    def extract(self, ocr_result: OCRResult) -> ExtractedField:
        """
        Extracts the vendor/merchant name from the header region of the receipt.
        Analyzes early lines in the top 35% of the page, favoring prominent font sizes
        and valid alphanumeric strings.
        """
        lines = [line.strip() for line in ocr_result.full_text.splitlines() if line.strip()]
        if not lines:
            return ExtractedField(value=None, confidence=0.0, evidence="Empty text")

        candidates: List[tuple[str, float, str]] = []  # (vendor_text, score, evidence)

        # Inspect first 8 lines
        for idx, line in enumerate(lines[:8]):
            line_upper = line.upper()

            # Skip excluded patterns
            if any(re.search(pat, line_upper) for pat in EXCLUDE_VENDOR_PATTERNS):
                continue

            # Skip lines that are purely numeric, dates, or symbols
            clean = re.sub(r"[\s\-_.,/\\#]", "", line)
            if not clean or clean.isdigit() or len(clean) < 3:
                continue

            # Skip lines that look like addresses or tax lines
            if any(k in line_upper for k in ["ROAD", "STREET", "AVENUE", "LANE", "PO BOX", "PINCODE", "ZIP"]):
                continue

            # Scoring:
            # - First line gets highest positional bonus
            # - Length bonus for reasonable name length (3 to 45 chars)
            pos_weight = max(0.2, 1.0 - (idx * 0.15))
            len_weight = 1.0 if 4 <= len(line) <= 35 else 0.6
            caps_weight = 1.0 if line.isupper() else 0.85

            score = pos_weight * len_weight * caps_weight
            candidates.append((line, score, f"Line {idx + 1}: '{line}'"))

        if not candidates:
            # Fallback: take the very first non-empty line
            first_line = lines[0]
            return ExtractedField(
                value=first_line,
                confidence=0.45,
                raw_text=first_line,
                evidence="Fallback to first non-empty line",
            )

        # Select highest scoring candidate
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_name, best_score, best_evidence = candidates[0]

        # Calculate OCR confidence for this line if tokens are available
        token_conf = 0.8
        matching_tokens = [
            t.confidence for t in ocr_result.tokens if t.text in best_name
        ]
        if matching_tokens:
            token_conf = sum(matching_tokens) / len(matching_tokens)

        final_conf = min(0.95, round(best_score * 0.5 + token_conf * 0.5, 2))

        return ExtractedField(
            value=best_name,
            confidence=final_conf,
            raw_text=best_name,
            evidence=best_evidence,
        )
