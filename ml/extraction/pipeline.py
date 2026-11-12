from decimal import Decimal
from typing import Any, Dict
from ml.extraction.address import AddressExtractor
from ml.extraction.date import DateExtractor
from ml.extraction.financial import FinancialExtractor
from ml.extraction.line_items import LineItemExtractor
from ml.extraction.schemas import ReceiptExtractionResult
from ml.extraction.validator import ReceiptValidator
from ml.extraction.vendor import VendorExtractor
from ml.ocr.engine import OCRResult


class ReceiptExtractionPipeline:
    def __init__(self):
        self.vendor_extractor = VendorExtractor()
        self.address_extractor = AddressExtractor()
        self.date_extractor = DateExtractor()
        self.financial_extractor = FinancialExtractor()
        self.line_items_extractor = LineItemExtractor()
        self.validator = ReceiptValidator()

    def process(self, ocr_result: OCRResult) -> ReceiptExtractionResult:
        # 1. Extract vendor name
        vendor = self.vendor_extractor.extract(ocr_result)

        # 2. Extract vendor address
        address = self.address_extractor.extract(ocr_result, vendor_name=vendor.value)

        # 3. Extract document date
        doc_date = self.date_extractor.extract(ocr_result)

        # 4. Extract currency
        currency = self.financial_extractor.extract_currency(ocr_result.full_text)

        # 5. Extract totals and taxes
        total, tax = self.financial_extractor.extract_totals_and_taxes(ocr_result)

        # 6. Extract experimental line items
        line_items = self.line_items_extractor.extract(ocr_result)

        # 7. Validate and score confidence
        errors, needs_review, overall_conf = self.validator.validate(
            vendor=vendor,
            address=address,
            doc_date=doc_date,
            total=total,
            currency=currency,
            tax=tax,
        )

        # Include OCR warnings in validation errors
        for w in ocr_result.warnings:
            if w not in errors:
                errors.append(f"OCR warning: {w}")

        field_confidences = {
            "vendor_name": vendor.confidence,
            "vendor_address": address.confidence,
            "document_date": doc_date.confidence,
            "total_amount": total.confidence,
            "currency": currency.confidence,
            "tax_amount": tax.confidence,
        }

        evidence_map = {
            "vendor_name": vendor.evidence,
            "vendor_address": address.evidence,
            "document_date": doc_date.evidence,
            "total_amount": total.evidence,
            "currency": currency.evidence,
            "tax_amount": tax.evidence,
        }

        metadata = {
            "total_tokens": len(ocr_result.tokens),
            "processed_pages": ocr_result.processed_pages,
            "total_pages": ocr_result.total_pages,
            "avg_ocr_confidence": ocr_result.avg_confidence,
            "needs_manual_review": needs_review,
            "has_line_items": len(line_items) > 0,
        }

        return ReceiptExtractionResult(
            vendor_name=vendor,
            vendor_address=address,
            document_date=doc_date,
            total_amount=total,
            currency=currency,
            tax_amount=tax,
            line_items=line_items,
            overall_confidence=overall_conf,
            field_confidences=field_confidences,
            validation_errors=errors,
            raw_ocr_text=ocr_result.full_text,
            needs_manual_review=needs_review,
            evidence_map=evidence_map,
            metadata=metadata,
        )


pipeline = ReceiptExtractionPipeline()

