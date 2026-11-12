from datetime import date, datetime
import re
from typing import List, Optional, Tuple
from dateutil import parser as date_parser

from backend.app.config import settings
from ml.extraction.schemas import ExtractedField
from ml.ocr.engine import OCRResult

# Explicit date regex patterns
DATE_PATTERNS = [
    # YYYY-MM-DD, YYYY/MM/DD, YYYY.MM.DD
    (r"\b(19\d\d|20\d\d)[-/.](0?[1-9]|1[0-2])[-/.](0?[1-9]|[12]\d|3[01])\b", "YMD"),
    # DD/MM/YYYY, DD-MM-YYYY, DD.MM.YYYY
    (r"\b(0?[1-9]|[12]\d|3[01])[-/.](0?[1-9]|1[0-2])[-/.](19\d\d|20\d\d)\b", "DMY_or_MDY"),
    # DD/MM/YY, DD-MM-YY
    (r"\b(0?[1-9]|[12]\d|3[01])[-/.](0?[1-9]|1[0-2])[-/.](\d{2})\b", "SHORT_YEAR"),
    # Textual: 15 Oct 2024, 15 October 2024, Oct 15 2024, October 15, 2024
    (
        r"\b(0?[1-9]|[12]\d|3[01])\s+(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[,\s]+(19\d\d|20\d\d)\b",
        "TEXTUAL_DMY",
    ),
    (
        r"\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(0?[1-9]|[12]\d|3[01])[,\s]+(19\d\d|20\d\d)\b",
        "TEXTUAL_MDY",
    ),
]


class DateExtractor:
    def __init__(self, default_order: str = settings.DEFAULT_DATE_FORMAT_ORDER):
        self.default_order = default_order.upper()  # "DMY" or "MDY"

    def parse_candidate(self, date_str: str, pattern_type: str) -> Optional[Tuple[date, bool]]:
        """
        Parses a date string according to explicit format rules.
        Returns:
            Tuple of (parsed_date, is_ambiguous) or None
        """
        try:
            is_ambiguous = False
            parsed_dt = None

            if pattern_type == "YMD":
                # Strict YYYY-MM-DD
                clean = re.sub(r"[/.]", "-", date_str)
                parsed_dt = datetime.strptime(clean, "%Y-%m-%d").date()

            elif pattern_type == "DMY_or_MDY":
                parts = [int(p) for p in re.split(r"[-/.]", date_str)]
                p1, p2, p3 = parts[0], parts[1], parts[2]
                
                # Check for ambiguity: e.g. 05/06/2024 (could be May 6 or June 5)
                if p1 <= 12 and p2 <= 12:
                    is_ambiguous = True
                    if self.default_order == "MDY":
                        parsed_dt = date(year=p3, month=p1, day=p2)
                    else:
                        parsed_dt = date(year=p3, month=p2, day=p1)
                elif p1 > 12:  # Must be Day first: DD/MM/YYYY
                    parsed_dt = date(year=p3, month=p2, day=p1)
                else:  # Must be Month first: MM/DD/YYYY
                    parsed_dt = date(year=p3, month=p1, day=p2)

            elif pattern_type == "SHORT_YEAR":
                parts = [int(p) for p in re.split(r"[-/.]", date_str)]
                p1, p2, p3 = parts[0], parts[1], parts[2]
                year = 2000 + p3 if p3 < 70 else 1900 + p3
                if p1 <= 12 and p2 <= 12:
                    is_ambiguous = True
                    if self.default_order == "MDY":
                        parsed_dt = date(year=year, month=p1, day=p2)
                    else:
                        parsed_dt = date(year=year, month=p2, day=p1)
                elif p1 > 12:
                    parsed_dt = date(year=year, month=p2, day=p1)
                else:
                    parsed_dt = date(year=year, month=p1, day=p2)

            else:
                # Textual date parsing via dateutil
                parsed = date_parser.parse(date_str, dayfirst=(self.default_order == "DMY"))
                parsed_dt = parsed.date()

            # Date sanity check: receipts must be between 1990 and 1 year in future
            current_year = datetime.now().year
            if 1990 <= parsed_dt.year <= (current_year + 1):
                return parsed_dt, is_ambiguous

            return None
        except Exception:
            return None

    def extract(self, ocr_result: OCRResult) -> ExtractedField:
        text = ocr_result.full_text
        found_candidates: List[Tuple[date, float, str, bool]] = []  # (date, confidence, raw_snippet, is_ambiguous)

        for pat, pat_type in DATE_PATTERNS:
            for match in re.finditer(pat, text, re.IGNORECASE):
                raw_snippet = match.group(0)
                result = self.parse_candidate(raw_snippet, pat_type)
                if result:
                    parsed_date, is_ambiguous = result
                    
                    # Context bonus if preceding text mentions DATE or TIME
                    start_idx = max(0, match.start() - 25)
                    context_before = text[start_idx:match.start()].upper()
                    has_keyword = any(k in context_before for k in ["DATE", "DT", "DATED", "TIME", "ON:"])
                    
                    base_conf = 0.85 if has_keyword else 0.70
                    if is_ambiguous:
                        base_conf -= 0.15

                    found_candidates.append(
                        (parsed_date, round(base_conf, 2), raw_snippet, is_ambiguous)
                    )

        if not found_candidates:
            return ExtractedField(value=None, confidence=0.0, evidence="No valid date patterns detected")

        # Sort: prefer dates with keywords and higher confidence, then most recent valid date
        found_candidates.sort(key=lambda x: (x[1], x[0]), reverse=True)
        best_date, best_conf, best_raw, is_ambiguous = found_candidates[0]

        evidence_msg = f"Extracted '{best_raw}' -> {best_date.isoformat()}"
        if is_ambiguous:
            evidence_msg += f" (Ambiguous format resolved using preference {self.default_order})"

        return ExtractedField(
            value=best_date,
            confidence=best_conf,
            raw_text=best_raw,
            evidence=evidence_msg,
            metadata={"is_ambiguous": is_ambiguous, "format_order": self.default_order},
        )
