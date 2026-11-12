import shutil
from pathlib import Path
from typing import Any
from ml.base import BaseExtractor, ExtractionOutput


class TesseractBaselineExtractor(BaseExtractor):
    """
    Tesseract OCR + Regex/Heuristic rules baseline interface for receipt/invoice extraction.
    In Phase 1, validates environment capability (Pillow / Tesseract binary availability)
    and serves as the baseline contract without running full pipeline or fabricating metrics.
    """

    @property
    def method_name(self) -> str:
        return "TESSERACT_BASELINE"

    def check_system_dependencies(self) -> dict:
        has_binary = shutil.which("tesseract") is not None
        return {
            "tesseract_binary_found": has_binary,
            "ready_for_phase_2": has_binary,
        }

    async def extract(self, file_path: Path, **kwargs: Any) -> ExtractionOutput:
        deps = self.check_system_dependencies()
        return ExtractionOutput(
            extraction_method=self.method_name,
            is_placeholder=True,
            structured_data=None,
            raw_text=None,
            confidence=None,
            validation_errors=[
                "Tesseract baseline pipeline interface initialized. Full OCR + regex extraction "
                "baseline will be calibrated in Phase 2 on SROIE benchmark."
            ],
            metadata={
                "file_name": file_path.name,
                "environment_status": deps,
                "phase": 1,
                "baseline_type": "OCR_HEURISTIC",
            }
        )

