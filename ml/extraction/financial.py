import re
from decimal import Decimal, InvalidOperation
from typing import Dict, List, Optional, Tuple
from ml.extraction.schemas import ExtractedField
from ml.ocr.engine import OCRResult

# Supported currencies mapping symbol/code to ISO 4217 code
CURRENCY_MAP = {
    "$": "USD",
    "USD": "USD",
    "US$": "USD",
    "₹": "INR",
    "RS": "INR",
    "RS.": "INR",
    "INR": "INR",
    "€": "EUR",
    "EUR": "EUR",
    "£": "GBP",
    "GBP": "GBP",
    "¥": "JPY",
    "JPY": "JPY",
    "C$": "CAD",
    "CAD": "CAD",
    "A$": "AUD",
    "AUD": "AUD",
    "S$": "SGD",
    "SGD": "SGD",
}

# Total amount keywords with weights
TOTAL_KEYWORDS = [
    (r"\bGRAND\s+TOTAL\b", 1.0),
    (r"\bNET\s+PAYABLE\b", 1.0),
    (r"\bAMOUNT\s+PAID\b", 0.95),
    (r"\bBALANCE\s+DUE\b", 0.95),
    (r"\bTOTAL\s+DUE\b", 0.95),
    (r"\bTOTAL\s+AMOUNT\b", 0.95),
    (r"\bFINAL\s+TOTAL\b", 0.95),
    (r"\bTOTAL\b", 0.85),
    (r"\bAMOUNT\b", 0.65),
    (r"\bPAYABLE\b", 0.65),
]

# Excluded / negative keywords for total amounts
NEGATIVE_TOTAL_KEYWORDS = [
    r"\bSUBTOTAL\b",
    r"\bSUB\s+TOTAL\b",
    r"\bTAX\b",
    r"\bGST\b",
    r"\bVAT\b",
    r"\bDISCOUNT\b",
    r"\bCHANGE\b",
    r"\bCASH\b",
    r"\bSAVINGS\b",
    r"\bSAVED\b",
    r"\bITEM\b",
]

# Tax keywords
TAX_KEYWORDS = [
    (r"\bTOTAL\s+TAX\b", 1.0),
    (r"\bSALES\s+TAX\b", 0.95),
    (r"\bTAX\s+AMOUNT\b", 0.95),
    (r"\bGST\s+AMOUNT\b", 0.95),
    (r"\bVAT\s+AMOUNT\b", 0.95),
    (r"\bCGST\b", 0.85),
    (r"\bSGST\b", 0.85),
    (r"\bIGST\b", 0.85),
    (r"\bTAX\b", 0.75),
    (r"\bGST\b", 0.75),
    (r"\bVAT\b", 0.75),
]

# Regex for monetary values:
# Handles: 1,234.56 or 1234.56 or 1 234.56 or 1234,56
MONEY_REGEX = r"([$₹€£¥]|\b[A-Z]{3}\b)?\s*(\d{1,3}(?:[,\s]\d{3})*(?:\.\d{1,2})|\d+(?:\.\d{1,2})|\d{1,3}(?:[.\s]\d{3})*(?:,\d{1,2})|\d+)"


def _iter_monetary_amounts(line: str) -> List[str]:
    """Extracts candidate monetary strings from a line while skipping percentages (e.g. '8%')."""
    results: List[str] = []
    for match in re.finditer(MONEY_REGEX, line):
        prefix = (match.group(1) or "").strip()
        amt_str = match.group(2)
        trailing = line[match.end() :].lstrip()
        if not prefix and trailing.startswith("%"):
            continue
        results.append(amt_str)
    return results


def clean_monetary_value(raw_val: str) -> Optional[Decimal]:
    """
    Cleans raw monetary strings into python Decimal:
    Handles commas, spaces, currency symbols, and dot vs comma decimals.
    """
    if not raw_val:
        return None

    # Remove all letters and currency symbols
    clean = re.sub(r"[^\d.,]", "", raw_val).strip()
    if not clean:
        return None

    try:
        # If contains both comma and dot, e.g. 1,250.50 -> standard
        if "," in clean and "." in clean:
            if clean.rfind(".") > clean.rfind(","):
                clean = clean.replace(",", "")
            else:
                clean = clean.replace(".", "").replace(",", ".")
        elif "," in clean and "." not in clean:
            # European style 1250,50 or thousand separator 1,250
            parts = clean.split(",")
            if len(parts[-1]) == 2:  # Likely cents: 1250,50
                clean = "".join(parts[:-1]) + "." + parts[-1]
            else:
                clean = clean.replace(",", "")

        val = Decimal(clean)
        # Filter out 0 or unrealistically high numbers (e.g. phone numbers misread as amounts)
        if 0 < val <= Decimal("10000000.00"):
            return val.quantize(Decimal("0.01"))
        return None
    except (InvalidOperation, ValueError):
        return None


class FinancialExtractor:
    def extract_currency(self, text: str) -> ExtractedField:
        """Finds prominent currency symbols or ISO codes."""
        found_currencies: Dict[str, int] = {}
        text_upper = text.upper()

        # Check symbols first
        for sym, code in CURRENCY_MAP.items():
            if len(sym) == 1 and sym in text:
                found_currencies[code] = found_currencies.get(code, 0) + text.count(sym)

        # Check codes/words with word boundary
        for sym, code in CURRENCY_MAP.items():
            if len(sym) > 1:
                matches = len(re.findall(r"\b" + re.escape(sym) + r"\b", text_upper))
                if matches > 0:
                    found_currencies[code] = found_currencies.get(code, 0) + matches

        if not found_currencies:
            return ExtractedField(
                value=None,
                confidence=0.0,
                evidence="No recognizable currency symbol or ISO code detected",
            )

        best_code = max(found_currencies.items(), key=lambda x: x[1])[0]
        conf = 0.90 if found_currencies[best_code] >= 2 else 0.75

        return ExtractedField(
            value=best_code,
            confidence=conf,
            raw_text=best_code,
            evidence=f"Detected currency '{best_code}' ({found_currencies[best_code]} occurrences)",
        )

    def extract_totals_and_taxes(
        self, ocr_result: OCRResult
    ) -> Tuple[ExtractedField, ExtractedField]:
        """
        Extracts Total Amount and Tax Amount using keyword matching, spatial proximity,
        and mathematical sanity checks.
        """
        lines = [l.strip() for l in ocr_result.full_text.splitlines() if l.strip()]

        total_candidates: List[Tuple[Decimal, float, str, str]] = []  # (amount, score, raw_text, evidence)
        tax_candidates: List[Tuple[Decimal, float, str, str]] = []

        # Analyze lines for totals and taxes
        for idx, line in enumerate(lines):
            line_upper = line.upper()

            # 1. Total Candidates
            for kw_pat, base_weight in TOTAL_KEYWORDS:
                if re.search(kw_pat, line_upper):
                    # Check negative keywords on the same line
                    if any(re.search(neg, line_upper) for neg in NEGATIVE_TOTAL_KEYWORDS):
                        continue

                    # Search for amounts on the same line
                    amounts = _iter_monetary_amounts(line)
                    if amounts:
                        for amt_str in amounts:
                            val = clean_monetary_value(amt_str)
                            if val:
                                total_candidates.append(
                                    (val, base_weight, amt_str, f"Line {idx+1}: '{line}'")
                                )
                    elif idx + 1 < len(lines):
                        # Search line immediately below
                        next_line = lines[idx + 1]
                        amounts_next = _iter_monetary_amounts(next_line)
                        if amounts_next:
                            for amt_str in amounts_next:
                                val = clean_monetary_value(amt_str)
                                if val:
                                    total_candidates.append(
                                        (val, base_weight * 0.85, amt_str, f"Line {idx+2} (below keyword): '{next_line}'")
                                    )

            # 2. Tax Candidates
            for kw_pat, base_weight in TAX_KEYWORDS:
                if re.search(kw_pat, line_upper):
                    amounts = _iter_monetary_amounts(line)
                    if amounts:
                        for amt_str in amounts:
                            val = clean_monetary_value(amt_str)
                            if val:
                                tax_candidates.append(
                                    (val, base_weight, amt_str, f"Line {idx+1}: '{line}'")
                                )

        # --- Resolve Total Amount ---
        extracted_total: ExtractedField
        if not total_candidates:
            # Fallback: scan for highest reasonable amount in the bottom half of the document
            fallback_candidates = []
            for line in lines[len(lines) // 2 :]:
                for amt_str in _iter_monetary_amounts(line):
                    v = clean_monetary_value(amt_str)
                    if v and v > 0:
                        fallback_candidates.append((v, amt_str, line))

            if fallback_candidates:
                # Pick max reasonable amount in lower section
                fallback_candidates.sort(key=lambda x: x[0], reverse=True)
                top_val, top_raw, top_line = fallback_candidates[0]
                extracted_total = ExtractedField(
                    value=top_val,
                    confidence=0.50,
                    raw_text=top_raw,
                    evidence=f"Fallback max value in lower half: '{top_line}'",
                )
            else:
                extracted_total = ExtractedField(
                    value=None,
                    confidence=0.0,
                    evidence="No total amount candidate found",
                )
        else:
            # Group identical candidate values and combine weights
            scores_by_val: Dict[Decimal, Tuple[float, str, str]] = {}
            for val, weight, raw, ev in total_candidates:
                if val in scores_by_val:
                    cur_w, cur_raw, cur_ev = scores_by_val[val]
                    scores_by_val[val] = (cur_w + 0.15, cur_raw, cur_ev)
                else:
                    scores_by_val[val] = (weight, raw, ev)

            # Sort by score, then value
            sorted_totals = sorted(
                scores_by_val.items(), key=lambda item: (item[1][0], item[0]), reverse=True
            )
            best_val, (best_score, best_raw, best_ev) = sorted_totals[0]
            conf = min(0.95, round(best_score * 0.9, 2))

            extracted_total = ExtractedField(
                value=best_val,
                confidence=conf,
                raw_text=best_raw,
                evidence=best_ev,
            )

        # --- Resolve Tax Amount ---
        extracted_tax: ExtractedField
        if not tax_candidates:
            extracted_tax = ExtractedField(
                value=None,
                confidence=0.0,
                evidence="No tax/VAT/GST amount detected",
            )
        else:
            # Filter tax candidates: tax must be less than total if total exists
            valid_tax = []
            for val, weight, raw, ev in tax_candidates:
                if extracted_total.value is not None:
                    if val < extracted_total.value:
                        valid_tax.append((val, weight, raw, ev))
                else:
                    valid_tax.append((val, weight, raw, ev))

            if valid_tax:
                valid_tax.sort(key=lambda x: x[1], reverse=True)
                t_val, t_weight, t_raw, t_ev = valid_tax[0]
                extracted_tax = ExtractedField(
                    value=t_val,
                    confidence=min(0.90, round(t_weight * 0.85, 2)),
                    raw_text=t_raw,
                    evidence=t_ev,
                )
            else:
                extracted_tax = ExtractedField(
                    value=None,
                    confidence=0.0,
                    evidence="Tax candidate exceeded total amount and was rejected",
                )

        return extracted_total, extracted_tax

