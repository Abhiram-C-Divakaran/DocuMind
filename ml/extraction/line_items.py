import re
from decimal import Decimal
from typing import Any, Dict, List
from ml.extraction.financial import clean_monetary_value
from ml.ocr.engine import OCRResult

# Experimental pattern: description followed by numbers
ITEM_PATTERN = re.compile(
    r"^([A-Za-z0-9\s&'\-]{3,35})\s+(\d+)?\s*[@x]?\s*(\d+(?:\.\d{2})?)\s+(\d+(?:\.\d{2})?)$"
)


class LineItemExtractor:
    """
    Experimental tabular line item extractor for Phase 2A.
    Identifies structured items in the body section of receipts.
    Clearly marked as experimental in metadata.
    """

    def extract(self, ocr_result: OCRResult) -> List[Dict[str, Any]]:
        lines = [l.strip() for l in ocr_result.full_text.splitlines() if l.strip()]
        line_items: List[Dict[str, Any]] = []

        # Restrict to middle section of document to avoid header/footer
        if len(lines) > 6:
            candidate_lines = lines[3:-3]
        else:
            candidate_lines = lines

        for line in candidate_lines:
            # Skip lines with total/tax keywords
            line_upper = line.upper()
            if any(k in line_upper for k in ["TOTAL", "SUBTOTAL", "TAX", "BALANCE", "AMOUNT", "CASH", "CHANGE"]):
                continue

            match = ITEM_PATTERN.match(line)
            if match:
                desc = match.group(1).strip()
                qty_str = match.group(2)
                price_str = match.group(3)
                total_str = match.group(4)

                qty = Decimal(qty_str) if qty_str else Decimal(1)
                unit_price = clean_monetary_value(price_str)
                total_price = clean_monetary_value(total_str)

                if total_price:
                    line_items.append(
                        {
                            "description": desc,
                            "quantity": float(qty),
                            "unit_price": float(unit_price) if unit_price else None,
                            "total_price": float(total_price),
                            "experimental": True,
                        }
                    )

        return line_items

