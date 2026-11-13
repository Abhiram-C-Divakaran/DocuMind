from pathlib import Path
from typing import Any, Dict
from ml.base import BaseExtractor, ExtractionOutput
from ml.extraction.pipeline import pipeline
from ml.ocr.engine import execute_tesseract_ocr, get_tesseract_status


class TesseractBaselineExtractor(BaseExtractor):
    """
    Functioning Tesseract OCR + rule-based receipt extraction pipeline.
    Performs PDF conversion, image deskew, contrast normalization,
    word-level tokenization with normalized bounding boxes, and regex heuristic entity parsing.
    """

    @property
    def method_name(self) -> str:
        return "TESSERACT_BASELINE"

    def check_system_dependencies(self) -> Dict[str, Any]:
        is_available, binary, msg = get_tesseract_status()
        return {
            "tesseract_installed": is_available,
            "executable_path": binary,
            "status_message": msg,
        }

    async def extract(self, file_path: Path, **kwargs: Any) -> ExtractionOutput:
        # 1. Check system dependency
        is_available, binary, msg = get_tesseract_status()
        if not is_available:
            raise RuntimeError(msg)

        # 2. Execute OCR on document
        ocr_result = execute_tesseract_ocr(file_path)

        # 3. Execute extraction pipeline on OCR result
        extraction_res = pipeline.process(ocr_result)

        # 4. Format structured data for Pydantic schema
        structured_dict = {
            "vendor_name": extraction_res.vendor_name.value,
            "vendor_address": extraction_res.vendor_address.value,
            "document_date": (
                extraction_res.document_date.value.isoformat()
                if extraction_res.document_date.value
                else None
            ),
            "total_amount": (
                str(extraction_res.total_amount.value)
                if extraction_res.total_amount.value is not None
                else None
            ),
            "currency": extraction_res.currency.value,
            "tax_amount": (
                str(extraction_res.tax_amount.value)
                if extraction_res.tax_amount.value is not None
                else None
            ),
            "line_items": extraction_res.line_items,
        }

        metadata = {
            "phase": "2A",
            "is_real_ocr": True,
            "evidence": extraction_res.evidence_map,
            "needs_manual_review": extraction_res.needs_manual_review,
            "ocr_tokens": [
                {
                    "text": t.text,
                    "norm_bbox": list(t.norm_bbox),
                    "page_num": t.page_num,
                }
                for t in ocr_result.tokens[:1000]
            ],
            **extraction_res.metadata,
        }

        return ExtractionOutput(
            extraction_method=self.method_name,
            is_placeholder=False,
            structured_data=structured_dict,
            raw_text=extraction_res.raw_ocr_text,
            confidence=extraction_res.overall_confidence,
            field_confidences=extraction_res.field_confidences,
            validation_errors=extraction_res.validation_errors,
            metadata=metadata,
        )
