from datetime import date, datetime
from decimal import Decimal
from typing import Dict, List, Tuple
from backend.app.config import settings
from ml.extraction.schemas import ExtractedField


class ReceiptValidator:
    """
    Validates extracted receipt fields against business rules:
    - Date sanity (not future, reasonable past)
    - Total amount positivity and non-zero
    - Tax amount non-negativity and tax <= total
    - Missing key required fields
    - Ambiguous dates or totals
    """

    def validate(
        self,
        vendor: ExtractedField,
        address: ExtractedField,
        doc_date: ExtractedField,
        total: ExtractedField,
        currency: ExtractedField,
        tax: ExtractedField,
    ) -> Tuple[List[str], bool, float]:
        errors: List[str] = []

        # 1. Total Amount Validations
        if total.value is None:
            errors.append("Total amount was not detected.")
        else:
            if not isinstance(total.value, Decimal) or total.value <= Decimal("0"):
                errors.append(f"Invalid total amount: {total.value} (must be a positive decimal).")

        # 2. Tax Amount Validations
        if tax.value is not None:
            if not isinstance(tax.value, Decimal) or tax.value < Decimal("0"):
                errors.append(f"Invalid tax amount: {tax.value} (cannot be negative).")
            elif total.value is not None and tax.value > total.value:
                errors.append(
                    f"Inconsistent financial values: Tax amount ({tax.value}) exceeds Total amount ({total.value})."
                )

        # 3. Document Date Validations
        if doc_date.value is None:
            errors.append("Document date was not detected.")
        else:
            if isinstance(doc_date.value, date):
                today = datetime.now().date()
                if doc_date.value > today:
                    errors.append(f"Date is in the future: {doc_date.value.isoformat()}.")
                elif doc_date.value.year < 1990:
                    errors.append(f"Date is outside supported historical range: {doc_date.value.isoformat()}.")
            if doc_date.metadata.get("is_ambiguous"):
                errors.append(
                    f"Document date '{doc_date.raw_text}' is ambiguous between DD/MM and MM/DD. "
                    f"Resolved using system preference ({doc_date.metadata.get('format_order')})."
                )

        # 4. Vendor Name Validations
        if vendor.value is None:
            errors.append("Vendor name was not detected.")

        # 5. Currency Validations
        if currency.value is None and total.value is not None:
            errors.append("Currency was not explicitly detected for total amount.")

        # 6. Overall Confidence Estimation
        # Weighted average of field confidences
        weights = {
            "total_amount": 0.40,
            "document_date": 0.25,
            "vendor_name": 0.20,
            "currency": 0.10,
            "tax_amount": 0.05,
        }
        weighted_conf = (
            total.confidence * weights["total_amount"]
            + doc_date.confidence * weights["document_date"]
            + vendor.confidence * weights["vendor_name"]
            + currency.confidence * weights["currency"]
            + (tax.confidence if tax.value is not None else 0.5) * weights["tax_amount"]
        )
        overall_conf = round(weighted_conf, 2)

        # 7. Needs Manual Review Flag
        threshold = settings.OCR_CONFIDENCE_THRESHOLD
        has_critical_missing = (total.value is None) or (doc_date.value is None) or (vendor.value is None)
        has_financial_inconsistency = any("Inconsistent financial values" in e for e in errors)

        needs_review = (
            overall_conf < threshold
            or has_critical_missing
            or has_financial_inconsistency
            or len(errors) >= 2
        )

        return errors, needs_review, overall_conf

